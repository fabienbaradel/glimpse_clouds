from __future__ import print_function
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import random
from PIL import Image
import numpy as np
import ipdb
import pickle
from pycocotools import mask as maskUtils
import lintel
import time
from torch.utils.data.dataloader import default_collate
from random import shuffle
import torch.nn.functional as F


class NTU(data.Dataset):
    """
    A general dataset with fake label
    * root is the directory where are the extracted frames
    """

    def __init__(self, root, w=224, h=224, t=8, num_classes=60,
                 avi_dir='avi_256x256_30',
                 skeleton_dir='skeleton',
                 usual_transform=False,
                 common_suffix='_rgb.avi',
                 train=True,
                 dataset='train'):
        # Usual settings
        self.train = train
        self.root = root
        self.w = w
        self.h = h
        self.t = t
        self.avi_dir = avi_dir
        self.num_classes = num_classes
        self.usual_transform = usual_transform
        self.avi_dir_full = os.path.join(self.root, self.avi_dir)
        self.video_prefix = 'super_video'
        self.common_suffix = common_suffix
        self.skeleton_dir = skeleton_dir
        self.skeleton_dir = os.path.join(self.root, self.skeleton_dir)
        self.dataset = dataset
        self.dict_video_length_fn = os.path.join(self.avi_dir_full, 'dict_id_length.pickle')
        self.minus_len = 2
        self.video_label_pickle = ''

        # Retrieve the real shape of the super_video
        self.retrieve_w_and_h_from_dir()

        # NTU seetings
        self.max_len_clip = 4 * self.real_fps  # sec by fps -> num of frames
        self.split = 'CS'  # TODO for CV
        self.original_w, self.original_h = 1920, 1080

        # NB crops
        self.nb_crops = 5 if self.dataset == 'test' else 1

        # ID fo training subjects
        self.person_id_training = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        if self.dataset == 'train':
            self.person_id_to_keep = self.person_id_training
        else:
            self.person_id_to_keep = list(range(1, 40))
            self.person_id_to_keep = [p for p in self.person_id_to_keep if p not in self.person_id_training]

        # Get the videos
        self.list_video, self.dict_video_length = self.get_videos()

    @staticmethod
    def load_pickle(file):
        # file = '/Users/fabien/Datasets/EPIC_KITCHENS_2018/annotations/EPIC_train_action_labels.pkl'

        with open(file, mode='rb') as f:
            df = pickle.load(f, encoding='latin1')

        return df

    @staticmethod
    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    def get_videos(self):
        # Open the pickle file
        with open(self.dict_video_length_fn, 'rb') as file:
            dict_video_length = pickle.load(file)

        # List of videos (all datatset)
        list_all_videos = self.listdir_nohidden(self.avi_dir_full)
        list_video = [v.split(self.common_suffix)[0] for v in list_all_videos if
                      self.get_person_id(v) in self.person_id_to_keep and self.common_suffix in v]

        return list_video, dict_video_length

    def get_video_fn(self, id):
        # Video location
        video_location = os.path.join(self.avi_dir_full, '{}{}'.format(id, self.common_suffix))
        return video_location

    def get_length(self, id):
        return self.dict_video_length['/{}'.format(id)]

    @staticmethod
    def get_label_from_id(video_id):
        action_id = int(video_id.split('A')[1][:3]) - 1
        return action_id

    @staticmethod
    def get_person_id(video_id):
        try:
            person_id = int(video_id.split('P')[1][:3])
            return person_id
        except:
            return None

    def get_2D_skeleton(self, skeleton_file, timesteps, P=2):
        # Read the full content of a file

        with open(skeleton_file, mode='r') as file:
            content = file.readlines()
        content = [c.strip() for c in content]

        # Nb of frames
        T = int(content[0])

        # Init the numpy array
        np_xy_coordinates = np.zeros((T, P, 25, 2)).astype(np.float32)

        # Loop over the frames
        i = 1
        for t in range(T):
            # Number of person detected
            nb_person = int(content[i])

            # Loop over the number of person
            for p in range(nb_person):
                i = i + 2
                for j in range(25):
                    # Catch the line of j
                    i = i + 1
                    content_j = content[i]

                    # Slit the line
                    list_content_j = content_j.split(' ')
                    list_content_j = [float(c) for c in list_content_j]
                    xy_coordinates = list_content_j[5:7]

                    # Add in the numpy array
                    try:
                        np_xy_coordinates[t, p, j] = xy_coordinates
                    except Exception as e:
                        pass
                        # print(e)  # 3 persons e.g

            i += 1

        # How many person in maximum
        one_person = np.sum(np_xy_coordinates[:, 1]) == 0.
        nb_person = 1 if one_person else 2

        # Extract only the interesting frames
        np_xy_coordinates = np_xy_coordinates[timesteps]

        # Normalize to 0-1
        np_xy_coordinates[:, :, :, 0] /= float(self.original_w)
        np_xy_coordinates[:, :, :, 1] /= float(self.original_h)

        # Replace NaN by 0
        np_xy_coordinates = np.nan_to_num(np_xy_coordinates)

        return np_xy_coordinates, nb_person

    def get_target(self, id):
        label = self.get_label_from_id(id)
        label = torch.LongTensor([label])
        return label[0]

    def retrieve_w_and_h_from_dir(self):
        _, w_h, fps, *_ = self.avi_dir.split('_')
        w, h = w_h.split('x')
        self.real_w, self.real_h, self.real_fps = int(w), int(h), int(fps)
        self.ratio_real_crop_w, self.ratio_real_crop_h = self.real_w / self.w, self.real_h / self.h

    def get_video_and_length(self):
        # Open the pickle file
        with open(self.dict_video_length_fn, 'rb') as file:
            dict_video_length = pickle.load(file)

        # Loop in each super_video dir to get th right super_video file
        list_video = []
        for video_id, length in dict_video_length.items():
            # Video id
            real_id = int(video_id.split('/')[1])
            list_video.append(real_id)

        return list_video, dict_video_length

    def time_sampling(self, video_len):
        # update the video_len on some dataset
        video_len = video_len - self.minus_len

        # Check that the super_video is not too long
        diff = self.max_len_clip - video_len

        # Change the start and adapt the length of the super_video
        if diff >= 0:
            start = 0
        else:
            start = random.sample(range(abs(diff)), 1)[0]

        video_len_up = video_len - start

        # Size of the sub-seq
        len_subseq = video_len_up / float(self.t)

        # Sample over each bin and add the start time
        if self.dataset != 'train' and self.nb_crops == 1:
            timesteps = [int((len_subseq / 2.0) + t * len_subseq + start) for t in range(self.t)]
        else:
            timesteps = [int(random.sample(range(int(len_subseq)), 1)[0] + t * len_subseq + start) for t in
                         range(self.t)]

        return timesteps

    def video_transform(self, np_clip, np_skeleton):

        # Random crop
        _, _, h, w = np_clip.shape
        w_min, h_min = random.sample(range(w - self.w), 1)[0], random.sample(range(h - self.h), 1)[0]
        # clip
        np_clip = np_clip[:, :, h_min:(self.h + h_min), w_min:(self.w + w_min)]

        # skeleton
        # 0-1 -> w,h
        np_skeleton[:, :, :, 0] *= self.real_w
        np_skeleton[:, :, :, 1] *= self.real_h
        # minus
        np_skeleton[:, :, :, 0] = np.clip(np_skeleton[:, :, :, 0] - w_min, 0, self.w)
        np_skeleton[:, :, :, 1] = np.clip(np_skeleton[:, :, :, 1] - h_min, 0, self.h)

        if self.usual_transform:
            # Div by 255
            np_clip /= 255.

            # Normalization
            np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)  # mean
            np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)  # std

            # Normalization of the skeleton to -1:1
            np_skeleton[:, :, :, 0] /= self.w
            np_skeleton[:, :, :, 0] *= 2
            np_skeleton[:, :, :, 0] -= 1
            np_skeleton[:, :, :, 1] /= self.h
            np_skeleton[:, :, :, 1] *= 2
            np_skeleton[:, :, :, 1] -= 1

        return np_clip, np_skeleton

    def extract_frames(self, video_file, timesteps):

        with open(video_file, 'rb') as f:
            encoded_video = f.read()

            decoded_frames = lintel.loadvid_frame_nums(encoded_video,
                                                       frame_nums=timesteps,
                                                       width=self.real_w,
                                                       height=self.real_h)
            try:
                np_clip = np.frombuffer(decoded_frames, dtype=np.uint8)
                np_clip = np.reshape(np_clip,
                                     newshape=(self.t, self.real_h, self.real_w, 3))
                np_clip = np_clip.transpose([3, 0, 1, 2])
                np_clip = np.float32(np_clip)
            except Exception as e:
                np_clip = decoded_frames
        return np_clip

    @staticmethod
    def load_masks(file):
        with open(file, 'rb') as f:
            masks = pickle.load(f, encoding='latin-1')
        return (masks['segms'], masks['boxes'])

    def store_dict_video_into_pickle(self, dict_video_label, dataset):
        with open(self.video_label_pickle, 'wb') as file:
            pickle.dump(dict_video_label, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dict_video_label of {} saved! -> {}".format(dataset, self.video_label_pickle))

    def starting_point(self, id):
        return 0

    def __getitem__(self, index):
        """
          Args:
              index (int): Index
        """

        try:
            # Get the super_video dir
            id = self.list_video[index]

            # video length
            length = self.get_length(id)

            # create random target
            torch_target = self.get_target(id)

            # get as many crops as you want
            list_np_clip, list_np_skeleton = [], []
            for c in range(self.nb_crops):
                # From the timesteps sample some timesteps to extract
                timesteps = self.time_sampling(length)

                np_clip = self.extract_frames(self.get_video_fn(id), timesteps)

                # Get the skeleton data
                skeleton_fn = os.path.join(self.skeleton_dir, str(id) + '.skeleton')
                np_skeleton, nb_person = self.get_2D_skeleton(skeleton_fn, timesteps)

                # Data processing on the super_video
                np_clip, np_skeleton = self.video_transform(np_clip, np_skeleton)

                # Append
                if self.train:  # because the train set is something called 'training' or 'train'... 'tr' is always present!
                    break
                else:
                    list_np_clip.append(np_clip)
                    list_np_skeleton.append(np_skeleton)

            if not self.train:
                np_clip = np.stack(list_np_clip)
                np_skeleton = np.stack(list_np_skeleton)

            # Torch world
            torch_clip = torch.from_numpy(np_clip)
            torch_skeleton = torch.from_numpy(np_skeleton)
            torch_nb_person = torch.from_numpy(np.asarray([nb_person]))

            # ID
            np_uint8_id = np.fromstring(str(id), dtype=np.uint8)
            torch_id = torch.from_numpy(np_uint8_id)
            torch_id = F.pad(torch_id, (0, 100))[:100]

            return {"target": torch_target,
                    "clip": torch_clip,
                    "id": torch_id,
                    "skeleton": torch_skeleton,
                    "nb_person": torch_nb_person
                    }
        except Exception as e:
            return None

    def __len__(self):
        return len(self.list_video)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of videos: {}\n'.format(self.__len__())
        return fmt_str


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
