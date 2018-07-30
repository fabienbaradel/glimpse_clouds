import os
import math
import torch
import numpy as np
import numbers
import ipdb


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(AverageMeter):
    def __init__(self, topk=[1], all_dataset=True):
        super(AverageMeter, self).__init__()
        self.topk = topk
        self.all_dataset = all_dataset
        self.reset()

    def reset(self):
        self.val = [0. for _ in self.topk]
        self.avg = [0. for _ in self.topk]
        self.sum = [0. for _ in self.topk]
        self.count = 0.

    def add(self, output, target):
        maxk = max(self.topk)
        n = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        self.count += n
        for k in range(len(self.topk)):
            topk = self.topk[k]
            correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
            self.val[k] = (correct_k[0] / float(n)) * 100.
            self.sum[k] += self.val[k] * float(n)
            self.avg[k] = self.sum[k] / float(self.count)

        # Idx of correct preds
        correct_np = correct.numpy().squeeze()  # (1, B)
        list_idx_correct_preds = list(np.where(correct_np == 1)[0])

        return list_idx_correct_preds

    def value(self, k=0):
        return self.val[k], self.avg[k], None


def get_time_to_print(time_sec):
    hours = math.trunc(time_sec / 3600)
    time_sec = time_sec - hours * 3600
    mins = math.trunc(time_sec / 60)
    time_sec = time_sec - mins * 60
    secs = math.trunc(time_sec % 60)
    string = '%02d:%02d:%02d' % (hours, mins, secs)
    return string


def write_to_log(dataset, resume, epoch, metrics):
    file_full_name = os.path.join(resume, dataset + '_log')
    with open(file_full_name, 'a+') as f:
        f.write('Epoch=%03d, Loss=%.4f, Metric=%.4f\n' % (epoch, metrics[0], metrics[1]))


def count_nb_params(enum_params):
    nb_params = 0
    for parameter in enum_params:
        nb_param_w = 1
        for s in parameter.size():
            nb_param_w *= s
        nb_params += nb_param_w
    return nb_params


def print_number(number):
    """ print a ' every 3 number starting from the left (e.g 23999 -> 23'999)"""
    len_3 = round(len(str(number)) / 3.)

    j = 0
    number = str(number)
    for i in range(1, len_3 + 1):
        k = i * 3 + j
        number = number[:-k] + '\'' + number[-k:]
        j += 1

    # remove ' if it is at the begining
    if number[0] == '\'':
        return number[1:]
    else:
        return number
