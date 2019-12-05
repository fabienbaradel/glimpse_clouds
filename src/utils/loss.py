import torch
import torch.nn as nn
import ipdb
from utils.cuda import CUDA


def is_one_person(pose_t):
    return pose_t.detach().numpy().sum() == 0


def decompose_pose_t(pose, t, check_presence_2_persons=True):
    """
    Decompose the pose tensor
    Return the poses and a tensor indicating presence or not of a second person 
    """

    # Infer dimension
    B, *_ = pose.size()

    # Take pose at timestep
    pose_t = pose[:, t].view(B, -1)

    # Decompose
    pose_t_1, pose_t_2 = pose_t[:, :50], pose_t[:, 50:]

    # Assume the first person is always present
    sum_pose_t_2 = pose_t_2.sum(1)  # (B)

    # Constant
    constant = torch.ones(1)[0] * -50
    constant = constant.cuda() if CUDA else constant

    if check_presence_2_persons:
        # Create a binary vector (1 if there is a 2nd person - 0 else)
        list_presence_2_persons = []
        for b in range(B):
            if torch.equal(sum_pose_t_2[b], constant):  # due to normalization
                presence = torch.zeros(1)[0]
            else:
                # print("2 persons")
                presence = torch.ones(1)[0]

            # Append
            presence = presence.cuda() if CUDA else presence
            list_presence_2_persons.append(presence)

        # Stack
        presence_2_persons = torch.stack(list_presence_2_persons, 0)
    else:
        presence_2_persons = None

    return pose_t_1, pose_t_2, presence_2_persons


def pose_l2_loss(pose, pose_hat):
    """ groundtruth: (B, T, 2, 25, 2)
        pred: (B, T, 100)
    """
    # Do not compute the loss if no prediction
    if pose_hat is None:
        return init_loss()

    B, T, P, J, XY = pose.size()
    pose_hat = pose_hat.view(B, T, P, J, XY)

    loss = init_loss()

    for t in range(T):
        # pose at time t
        pose_t_1, pose_t_2, presence_2_persons = decompose_pose_t(pose, t)
        pose_t_1_hat, pose_t_2_hat, _ = decompose_pose_t(pose_hat, t, False)

        # Loop over the batch
        for b in range(B):
            # 2 persons
            if presence_2_persons[b]:
                # Compute all the possibilities
                mse_loss_1_1 = mse_loss(pose_t_1[b], pose_t_1_hat[b])
                mse_loss_1_2 = mse_loss(pose_t_1[b], pose_t_2_hat[b])
                mse_loss_2_1 = mse_loss(pose_t_2[b], pose_t_1_hat[b])
                mse_loss_2_2 = mse_loss(pose_t_2[b], pose_t_2_hat[b])

                # Select the optimal one (which prediction match the groundtruth
                if torch.gt(mse_loss_1_2, mse_loss_1_1):
                    loss = loss + mse_loss_1_1 + mse_loss_2_2
                else:
                    loss = loss + mse_loss_1_2 + mse_loss_2_1
            # 1 person only
            else:
                loss = loss + mse_loss(pose_t_1[b], pose_t_1_hat[b])

    return loss / float(B)


def loss_regularizer_glimpses(pose, attention_points, alpha_attraction_humans=1.0, alpha_encourage_diversity=1.0):
    """ pose : (B, T, 2, 25, 2) 
        attention_points: (B, T, G, 4) where G is the number of glimpses (here G=3)
    """
    loss = init_loss()

    # Infer dimension
    B, T, G, _ = attention_points.size()

    # Loop over timesteps
    for t in range(T):
        # Pose at time t
        pose_t_1, pose_t_2, presence_2_persons = decompose_pose_t(pose, t)  # 2x (B,50)

        # Compute the L2 norm for each pose
        for b in range(B):
            # 1 person
            if not presence_2_persons[b]:
                # Repeat G times
                pose_t_b = pose_t_1[b].view(25, 2).repeat(G, 1, 1)  # (G, 25, 2)
                # Number of points
                K = 25
            # 2 persons:
            else:
                # Cat pose
                pose_t_1_b = pose_t_1[b].view(25, 2).repeat(G, 1, 1)  # (G, 25, 2)
                pose_t_2_b = pose_t_2[b].view(25, 2).repeat(G, 1, 1)  # (G, 25, 2)
                pose_t_b = torch.cat([pose_t_1_b, pose_t_2_b], 1)
                # Number of points
                K = 50

            # Attention points
            attention_points_t_b = attention_points[b, t, :, :2].view(G, 1, 2).repeat(1, K, 1)  # (G, 25 or 50, 2)

            # Attraction to humans
            loss = loss + alpha_attraction_humans * loss_attraction_to_humans(pose_t_b, attention_points_t_b)

            # Encourage diversity over glimpses
            loss = loss + alpha_encourage_diversity * loss_encourage_divesity(attention_points_t_b)

    return loss / float(B)


def loss_attraction_to_humans(pose_t_b, attention_points_t_b):
    """ Make sure that the glimpses are not too far from humans """
    # Compute all the distances
    dist = torch.pow((pose_t_b - attention_points_t_b), 2)
    # take the min for each glimpse
    dist_g, _ = torch.min(dist, dim=1)
    # Sum up
    dist_g_sum = torch.sum(dist_g)

    return dist_g_sum


def loss_encourage_divesity(attention_points):
    """ input : (G, 25, 2) """

    # Infer the number of glimpse
    G, *_ = attention_points.size()

    # Expand dim
    attention_points_1 = attention_points.unsqueeze(0).expand(G, -1, -1, -1)  # (G,G,25,2)
    attention_points_2 = attention_points.unsqueeze(1).expand(-1, G, -1, -1)  # (G,G,25,2)

    # Distance L2
    dist_btw_glimpses = mse_loss(attention_points_1, attention_points_2)

    return 1./(1.+dist_btw_glimpses)


def mse_loss(input, target):
    return torch.sum(torch.pow((input - target), 2)) / input.data.nelement()


def init_loss():
    return torch.zeros(1)[0].cuda() if CUDA else torch.zeros(1)[0]
