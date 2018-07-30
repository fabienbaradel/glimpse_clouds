import torch
import torch.nn as nn
import random
import ipdb
from model.backbone.inflated_resnet import inflated_resnet
import torch.nn.functional as F
from torch.autograd import Variable
from model.spatial_transformer import zoom_ST
import numpy as np
from utils.cuda import CUDA


class GlimpseClouds(nn.Module):
    def __init__(self, nb_classes=60, nb_glimpses=3, nb_workers=3,
                 options={},
                 **kwargs):
        super(GlimpseClouds, self).__init__()

        # Settings
        self.nb_classes = nb_classes
        self.nb_workers = nb_workers
        self.nb_glimpses = nb_glimpses
        self.glimpse_clouds = options['glimpse_clouds']
        self.global_model = not options['glimpse_clouds']
        self.pose_predictions = options['pose_predictions']

        # # CNN-based model
        self.cnn = inflated_resnet()
        self.D = 2048

        # Avgpool
        self.avgpool_1x7x7 = nn.AvgPool3d((1, 7, 7))
        self.avgpool_1x14x14 = nn.AvgPool3d((1, 14, 14))
        self.avgpool_Tx7x7 = None
        self.avgpool_14x14 = nn.AvgPool2d((14, 14))
        self.avgpool_7x7 = nn.AvgPool2d((7, 7))

        # Pose regression
        self.fc_pose = nn.Linear(int(self.D / 2.), 25 * 2 * 2)

        # Recurrent Attention - Zoomer
        # RNN
        self.rnn_zoomer_size = int(self.D / 4.)
        self.rnn_zoomer = nn.GRU(input_size=self.D + int(self.D / 4.),  # cnn features + global hidden state of the task
                                 hidden_size=self.rnn_zoomer_size,
                                 batch_first=True)  # (batch, seq, feature)
        # MLP
        self.mlp_glimpse_location = nn.Linear(in_features=self.rnn_zoomer_size + int(self.D / 2.),
                                              # from hidden state of the zoomer and the human position features
                                              out_features=4  # (x,y,scale_x,scale_y)
                                              )

        # Embedding of the location
        self.mlp_embedding_location = nn.Sequential(nn.Linear(4, int(self.D / 8)),
                                                    nn.ReLU(),
                                                    nn.Linear(int(self.D / 8), self.D)
                                                    )

        # Workers and MemNet
        self.temperature = 0.5
        self.cosine_similiraity = nn.CosineSimilarity()
        self.list_worker, self.list_fc = nn.ModuleList(), nn.ModuleList()
        self.list_first_attention, self.list_attention_worker = nn.ParameterList(), []
        for _ in range(self.nb_workers):
            # RNN and FC layer
            worker = nn.GRU(input_size=self.D, hidden_size=int(self.D / 4.), num_layers=1,
                            batch_first=True)  # (batch, seq, feature)
            fc = nn.Linear(in_features=int(self.D / 4.), out_features=self.nb_classes)
            first_attention = nn.Parameter(torch.abs(torch.randn(self.nb_glimpses)))

            self.list_worker.append(worker)
            self.list_fc.append(fc)
            self.list_first_attention.append(first_attention)

        # Distance
        self.do_pretraining_distance = True

        # Freeze the CNN if glimpse clouds
        if options['glimpse_clouds']:
            for child in self.cnn.children():
                for param in child.parameters():
                    param.requires_grad = False

        # Similarity matrix
        # self.inv_covariance_gpu = nn.Parameter(torch.eye(self.D).type(torch.FloatTensor), requires_grad=False)
        self.inv_covariance_gpu = None

        # Classifier of the global model
        self.fc_global_model = nn.Linear(self.D, self.nb_classes)

    def forward_global_model(self, fm):
        B, D, T, W, H = fm.size()
        # Pooler
        self.avgpool_Tx7x7 = nn.AvgPool3d((T, W, H)) if self.avgpool_Tx7x7 is None else self.avgpool_Tx7x7

        # GAP pooling
        final_representation = self.avgpool_Tx7x7(fm)
        final_representation = final_representation.view(B, D)

        # Linear classifier
        logits = self.fc_global_model(final_representation)

        return logits

    def forward_pose_predictions(self, fm):
        B, D, T, W, H = fm.size()

        # Pooling
        features = self.avgpool_1x14x14(fm)
        features = features.view(B, D, T)

        # Transpose
        features = features.transpose(1, 2)  # (B, T, D)

        # Regress
        pose = self.fc_pose(features)
        pose = F.tanh(pose)  # to make sure the pose is between -1 and 1

        return pose

    def compute_similarity_matrix(self, all_v):
        # Shape
        B, T, C, D = all_v.size()

        if T == 1:
            return None

        # Distance matrix
        distance_matrix = torch.zeros((B, T, self.nb_glimpses, self.nb_glimpses))
        distance_matrix = distance_matrix.cuda() if CUDA else distance_matrix  # (B, T-1, 3, 3)

        # New features
        new_features = all_v[:, T - 1]  # (B,C,2048)
        previous_features = all_v[:, :(T - 1)]  # (B,T,C,2048)

        # Pretraining
        if self.do_pretraining_distance:
            # Mahalanobis distance
            previous_features = previous_features.contiguous()
            previous_features_cpu = previous_features.view(-1, self.D)  # (B*T,D)
            previous_features_cpu = previous_features_cpu.transpose(0, 1)  # (D,B)
            previous_features_cpu = previous_features_cpu.detach().cpu().numpy()
            covariance_cpu = np.cov(previous_features_cpu)  # (D,D) -> (2048,2048)
            inv_covariance_cpu = np.linalg.inv(covariance_cpu)  # *** numpy.linalg.linalg.LinAlgError: Singular matrix
            # inv_covariance_gpu = Variable(torch.from_numpy(inv_covariance_cpu).type(torch.FloatTensor),
            #                               requires_grad=True).repeat(B, 1, 1)  # (8, 2048,2048)
            inv_covariance_gpu = Variable(torch.from_numpy(inv_covariance_cpu).type(torch.FloatTensor),
                                          requires_grad=True)  # (2048,2048)
            inv_covariance_gpu = inv_covariance_gpu.cuda() if CUDA else inv_covariance_gpu
            self.inv_covariance_gpu = inv_covariance_gpu
            self.do_pretraining_distance = False
        for t in range(T - 1):
            for z_1 in range(self.nb_glimpses):
                for z_2 in range(self.nb_glimpses):
                    x = previous_features[:, t, z_1]  # (B,2048)
                    y = new_features[:, z_2]  # (B,2048)
                    x_y = (x - y).view(B, 1, 2048)  # (B,1,2048)

                    # No Mahalanobis here !
                    # mahalanobis_dist = torch.matmul(x_y, x_y.view(B, 2048, 1))

                    # Mahalanobis
                    mahalanobis_dist = torch.matmul(x_y, self.inv_covariance_gpu)
                    mahalanobis_dist = torch.matmul(mahalanobis_dist, x_y.view(B, 2048, 1))

                    # Tricks
                    mahalanobis_dist = mahalanobis_dist.squeeze()
                    mahalanobis_dist = torch.clamp(mahalanobis_dist, 0, 100000)
                    distance_matrix[:, t, z_1, z_2] = mahalanobis_dist

        # Normalization
        max_dist, min_dist = torch.max(distance_matrix), torch.min(distance_matrix)
        distance_matrix = (distance_matrix - min_dist) / (max_dist - min_dist)
        similarity_matrix = 1 - distance_matrix

        # return torch.ones_like(similarity_matrix)
        return similarity_matrix

    def get_worker_input(self, similarity_matrix, all_v, w, t):
        B, T, *_ = all_v.size()

        # ipdb.set_trace()
        # Attention weights
        if t == 0:
            attention_W = self.list_first_attention[w]  # (C)
            attention_W = attention_W.repeat(B, 1)  # (B, C)
        else:
            # Retrieve previous attention
            previous_attention_w = self.list_attention_worker[w]
            previous_attention_w = torch.stack(previous_attention_w, 1)  # (B,T,C,1)
            # And compute the new one given the similarity matrix
            attention_W = previous_attention_w * similarity_matrix[:, :T - 1]  # (B,t-1,C,C)
            attention_W = torch.mean(attention_W, -2)  # (B,t,C)
            exponential_time_decay = torch.exp(-torch.arange(0, t) / 4.0).repeat(attention_W.size(0), 1).view(B, t, 1)
            exponential_time_decay = exponential_time_decay.cuda() if CUDA else exponential_time_decay
            attention_W = attention_W * exponential_time_decay  # (B,t,C)
            attention_W = torch.sum(attention_W, 1)  # (B,C)

        # Softmax
        attention_W = attention_W.unsqueeze(2)  # (B,C,1)
        attention_W = F.softmax(attention_W / self.temperature, 1)

        # Store
        self.list_attention_worker[w].append(attention_W)

        # V tild
        v_tild = torch.sum(all_v[:, t] * attention_W, 1)  # (B,D)

        return v_tild

    def forward_glimpse_clouds(self, final_fm, pose_fm):
        # Size of the feature maps
        B, D, T, W, H = final_fm.size()

        # For storing attention weights of the workers
        self.list_attention_worker = [[] for _ in range(self.nb_glimpses)]

        # List of attention points
        list_v = []
        list_attention_points_glimpses = []

        # Init the hidden state of the zoomer
        h = torch.zeros(1, B, self.rnn_zoomer_size)
        h = h.cuda() if CUDA else h

        # Init the hidden state of the workers
        list_r = [torch.zeros(1, B, int(D / 4.)) for _ in range(self.nb_workers)]
        list_r = [x.cuda() if CUDA else x for x in list_r]

        # Loop over time
        list_logits = []
        for t in range(T):
            # Extract the feature maps and the pose features
            final_fm_t, pose_fm_t = final_fm[:, :, t], pose_fm[:, :, t]  # (B, 2048, 7, 7) - (B, 1024, 14, 14)
            c = self.avgpool_14x14(pose_fm_t).view(B, int(D / 2.))  # (B, 1024)

            # Hidden state of th workers
            r_all_workers = list_r[0]
            for r_w in list_r[1:]:
                r_all_workers = r_all_workers + r_w
            r_all_workers = r_all_workers.transpose(0, 1)  # (B, 1, D/4)

            # Loop over the glimpses
            for g in range(self.nb_glimpses):
                # Input of the RNN zoomer
                input_loc_params = torch.cat([c, h.view(B, int(D / 4.))], 1)  # (B, 1536)

                # Estimate (x,y,scale_x,scale_y) of the glimpse
                loc = self.mlp_glimpse_location(input_loc_params)  # (B, 4)
                # ipdb.set_trace()
                loc_xy = F.tanh(loc[:, :2])  # to make sure it is between -1 and 1
                loc_zooms = F.sigmoid(loc[:, 2:] + 3.)  # to make sure it is between 0 and 1 - +3 for starting with a zoom ~ 1

                # Extract the corresponding features map with Spatial Transformer
                Z = zoom_ST(final_fm_t, loc_xy, loc_zooms, W, H, CUDA)  # (B, 2048, 7, 7)

                # Get the visual and location features and finally append
                z = self.avgpool_7x7(Z).view(B, D)  # (B, 2048)
                v = z * self.mlp_embedding_location(loc)  # (B, 2048)

                # Store glimpse features and attention points
                list_v.append(v)
                list_attention_points_glimpses.append(torch.cat([loc_xy, loc_zooms], 1))

                # Update the zoomer
                _, h = self.rnn_zoomer(torch.cat([v.view(B, 1, D), r_all_workers], 2), h)

            # Compute the similarity matrix
            all_v = torch.stack(list_v, 1).view(B, t + 1, self.nb_glimpses, D)  # (B,t,C,D)
            similarity_matrix = self.compute_similarity_matrix(all_v)

            # Create the input for each worker
            list_v_tild = []
            # Distribute the features over the workers
            for w in range(self.nb_workers):
                # Get the input for the worker
                input_worker = self.get_worker_input(similarity_matrix, all_v, w, t)
                list_v_tild.append(input_worker)

                # Catch the workers and its previous hidden state
                rnn, hidden = self.list_worker[w], list_r[0]

                # Run the rnn
                out, hidden = rnn(input_worker.unsqueeze(1), hidden)

                # Update the list of hidden state
                list_r[w] = hidden

                # And finally classify
                fc = self.list_fc[w]
                logits = fc(out.view(B, int(D / 4.)))
                list_logits.append(logits)

        # Stack
        all_logits = torch.stack(list_logits, 1)  # (B,T,60)

        # Average the logits
        logits = torch.mean(all_logits, 1)  # (B, 60)

        # Stack attention points
        attention_points_glimpses = torch.stack(list_attention_points_glimpses, 1).view(B, T, self.nb_glimpses, 4)

        return logits, attention_points_glimpses

    def forward(self, x):

        # Extract the input
        clip, skeleton = x['clip'], x['skeleton']

        # Extract feature maps
        final_fm, pose_fm = self.cnn.get_feature_maps(clip)

        # Predict the pose
        pose = self.forward_pose_predictions(pose_fm) if self.pose_predictions else None

        # Several way possible for training the network
        if self.global_model:
            logits = self.forward_global_model(final_fm)
            attention_points = None
        else:
            logits, attention_points = self.forward_glimpse_clouds(final_fm, pose_fm)
            # ipdb.set_trace()
            # print(attention_points[0])

        return logits, pose, attention_points
