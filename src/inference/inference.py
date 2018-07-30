import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.meter import *
import ipdb
# from model import models
from data.ntu.dataloader import my_collate
from model.glimpse_clouds import *
from utils.dataset import get_datasets_n_loaders
from inference.train_val import *
from model.glimpse_clouds import GlimpseClouds
from utils.save_restore import *
from utils.cuda import CUDA
from sys import platform

torch.backends.cudnn.enabled = False  # BUg in CUDNN ---> https://github.com/pytorch/pytorch/issues/4107


def inference(options):
    print("\n* DIR ===> {}".format(options['dir']))
    print("* DATA ===> {}\n".format(options['root']))

    # Create the model
    model = GlimpseClouds(num_classes=60,
                          options=options)

    # Make a Data Parallel and CUDA model if possible
    model = model.cuda() if CUDA else model
    if options['global_model']:
        # it seems that if I data parallel the input_var to my model goes anyway to the GPU even if I want to stay CPU based -- working oly for the global model
        model = torch.nn.DataParallel(model)

        # Trainable params
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Print number of parameters
    nb_total_params = count_nb_params(model.parameters())
    nb_trainable_params = count_nb_params(filter(lambda p: p.requires_grad, model.parameters()))
    ratio = float(nb_trainable_params / nb_total_params) * 100.
    print("* Total nb params (nb trainable params) : {} ({}) - {ratio:.2f}% of the weights are trainable".format(
        print_number(nb_total_params),
        print_number(nb_trainable_params),
        ratio=ratio
    ))

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    metric = AccuracyMeter()

    # Optimizer
    optimizer = torch.optim.Adam(trainable_params, options['lr'])

    # Optionally resume from a checkpoint
    model, optimizer, best_metric_val, epoch = load_from_dir(model, optimizer, options)

    # Datasets and Dataloader
    train_set, test_set, train_loader, test_loader = get_datasets_n_loaders(options, CUDA)

    print('\n*** Train set of size {}  -  Test set of size {} ***\n'.format(print_number(len(train_set.list_video)),
                                                                           print_number(len(test_set.list_video))))

    # Print-freq
    print_freq = 1 if platform == "darwin" else 10

    # Training or evaluation
    if not options['evaluate']:
        # Init metric
        best_metric_val, best_epoch_val = 0, 0
        for epoch in range(options['epochs']):
            # Train one epoch
            loss_train, metric_train = train(train_loader, model, criterion, optimizer, metric, epoch, options,
                                             cuda=CUDA, print_freq=print_freq)

            # Write in log
            write_to_log('train', options['dir'], epoch + 1, [loss_train, metric_train])

            # Save the model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric_val': best_metric_val,
                'optimizer': optimizer.state_dict(),
            }, True, options['dir'])

    else:
        # Evaluate
        loss_test, metric_test = val(test_loader, model, criterion, None, metric, epoch, options, cuda=CUDA,
                                     NB_CROPS=test_loader.dataset.nb_crops, print_freq=print_freq)

        # Write in log
        write_to_log('test', options['dir'], epoch + 1, [loss_test, metric_test])

        pass
