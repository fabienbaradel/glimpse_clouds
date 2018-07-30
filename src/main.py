import argparse
import ipdb
import inference.inference as inference
from sys import platform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glimpse Clouds')
    parser.add_argument('--dir', metavar='DIR',
                        default='/tmp/my_resume',
                        help='path to the resume dir')
    parser.add_argument('--root', metavar='ROOT',
                        default='/storage/Datasets/NTU-RGB-D-Action-Recognition',  # oggy
                        help='path to dataset')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 3)')
    parser.add_argument('-t', '--time', default=3, type=int,
                        metavar='T', help='number of timesteps per super_video for train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--global-model', dest='global_model',
                        action='store_true',
                        help='Train the model with Global Average Pooling')
    parser.add_argument('--glimpse-clouds', dest='glimpse_clouds',
                        action='store_true',
                        help='Train the model with Glimpse Clouds')
    parser.add_argument('--pose-predictions', dest='pose_predictions',
                        action='store_true',
                        help='Regress the pose from the penultimate features maps')

    # Args
    args, _ = parser.parse_known_args()

    # Transform to dict
    options = vars(args)

    options['global_model'] = True
    # options['glimpse_clouds'] = True
    # options['pose_predictions'] = True
    options[
        'root'] = '/Users/fabien/Datasets/NTU-RGB-D' if platform == "darwin" else '/storage/Datasets/NTU-RGB-D-Action-Recognition'
    options['workers'] = 0 if platform == "darwin" else 4

    # Infer
    inference.inference(options)
