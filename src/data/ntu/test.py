import argparse
from data.ntu.dataloader import NTU
import torch
import ipdb
import time
import sys
from utils.meter import AverageMeter
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from data.ntu.dataloader import my_collate
import os
from matplotlib.patches import Circle

def main(args):
    # NTU Dataset
    dataset = NTU(root=args.root,
                  w=args.width,
                  h=args.height,
                  t=args.time,
                  dataset='train',
                  train=True,
                  avi_dir=args.avi_dir,
                  usual_transform=False, )

    # Pytorch dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=args.cuda,
                                             collate_fn=my_collate)
    # Loop
    data_time = AverageMeter()
    start_data = time.time()
    for i, dict_input in enumerate(dataloader):
        duration_data = time.time() - start_data
        data_time.update(duration_data)

        # Get the data
        clip, skeleton = dict_input['clip'], dict_input['skeleton'] # (B, C, T, 224, 224), (B, T, 2, 25, 2)
        # Show
        show_one_img(clip[0, :, 0], skeleton[0, 0])

        print("{}/{} : {time.val:.3f} ({time.avg:.3f}) sec/batch".format(i + 1, len(dataloader),
                                                                         time=data_time))
        sys.stdout.flush()
        start_data = time.time()


def show_one_img(torch_img, torch_skeleton):
    np_img, np_skeleton = torch_img.numpy(), torch_skeleton.numpy()

    C, H, W = np_img.shape

    # Rescale skeleton from -1,1 to 0,255
    if np.max(np_skeleton) < 1.0:
        np_skeleton[:, :, 0] +=1
        np_skeleton[:, :, 0] /=2
        np_skeleton[:, :, 0] *= float(W)
        np_skeleton[:, :, 1] +=1
        np_skeleton[:, :, 1] /=2
        np_skeleton[:, :, 1] *= float(H)
    # Uint8 img
    img_array = np_img.astype('uint8')
    img_array = img_array.transpose([1, 2, 0])

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(img_array)

    # Show the skeleton points
    for j in range(25):
        circ = Circle((np_skeleton[0,j,0], np_skeleton[0,j,1]), 2)
        ax.add_patch(circ)

    # Show the image
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Loader')
    parser.add_argument('--root', metavar='DIR',
                        default='/Users/fabien/Datasets/NTU-RGB-D',
                        help='path to avi dir')
    parser.add_argument('--avi-dir', metavar='AVI',
                        default='avi_256x256_30',
                        help='path to avi dir')
    parser.add_argument('--width', default=224, type=int,
                        metavar='W', help='Width of the images')
    parser.add_argument('--height', default=224, type=int,
                        metavar='H', help='Height of the images')
    parser.add_argument('--time', default=8, type=int,
                        metavar='H', help='Number of timesteps to extract from a super_video')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 3)')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='cuda mode')

    args = parser.parse_args()

    main(args)
