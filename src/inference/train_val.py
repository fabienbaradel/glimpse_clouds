from utils.meter import *
import time
import torch
import ipdb
import sys
import torch.nn as nn
# from utils.others import *
import matplotlib
from utils.loss import *


def make_variable_all_input(dict_input, cuda=False):
    dict_input_var = {}
    for k, v in dict_input.items():
        var = torch.autograd.Variable(v)
        dict_input_var[k] = var.cuda() if cuda else var
    return dict_input_var


def forward_backward(model, options, input_var, criterion, optimizer=None, cuda=False):
    # compute output
    output, pose, attention_points = model(input_var)  # (B, 60) - (B, T, 100)

    # pose L2 loss
    loss_pose = pose_l2_loss(input_var['skeleton'], pose)

    # activity loss
    loss_activity = criterion(output, input_var['target'])

    # attraction loss
    if options['glimpse_clouds']:
        loss_attraction = loss_regularizer_glimpses(input_var['skeleton'], attention_points)
    else:
        loss_attraction = init_loss()

    # Full loss
    # ipdb.set_trace()
    # print('pose: ', loss_pose, 'activity: ', loss_activity, 'attraction: ', loss_attraction)
    loss = 0.1 * loss_pose + loss_activity + loss_attraction

    # backward if needed
    if optimizer is not None:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output, loss


def update_metric_loss(input, output, metric, loss, losses):
    # loss
    losses.update(loss.detach(), input['clip'].size(0))

    # metrics
    target = input['target']

    target = target.cpu()
    preds = output.view(-1, output.size(-1)).data.cpu()

    list_idx_correct_preds = metric.add(preds, target)

    metric_val, metric_avg, _ = metric.value()

    return metric_val, metric_avg, list_idx_correct_preds


def train(train_loader, model, criterion, optimizer, metric, epoch, options, cuda=False, print_freq=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric.reset()

    # switch to train mode
    model.train()

    end = time.time()
    print("")
    for i, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Make Variables
        input_var = make_variable_all_input(input, cuda=cuda)

        # compute output
        output, loss = forward_backward(model,
                                        options,
                                        input_var,
                                        criterion,
                                        optimizer,
                                        cuda)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            # measure accuracy and record loss
            metric_val, metric_avg, *_ = update_metric_loss(input, output, metric, loss, losses)

            time_done = get_time_to_print(batch_time.avg * (i + 1))
            time_remaining = get_time_to_print(batch_time.avg * len(train_loader))

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) [{done} => {remaining}]\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric {metric_val:.3f} ({metric_avg:.3f})'.format(
                epoch, i + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, metric_val=metric_val,
                metric_avg=metric_avg,
                done=time_done, remaining=time_remaining)
            )
            sys.stdout.flush()

    return losses.avg, metric_avg


def val(val_loader, model, criterion, optimizer, metric, epoch, options, cuda=False, print_freq=1, NB_CROPS=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric.reset()

    # switch to train mode
    model.eval()

    end = time.time()
    print("")
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # Make Variables
            input_var = make_variable_all_input(input, cuda=cuda)

            output, loss = None, None
            for j in range(NB_CROPS):
                input_var_j = {'clip': input_var['clip'][:, j],
                               'skeleton': input_var['skeleton'][:, j],
                               'target': input_var['target']
                               }
                # compute output
                output_j, loss_j = forward_backward(model,
                                                    options,
                                                    input_var_j,
                                                    criterion,
                                                    None,
                                                    cuda)
                # Append
                output = output_j if output is None else output + output_j
                loss = loss_j if loss is None else loss + loss_j

            # Div
            output /= NB_CROPS
            loss /= NB_CROPS

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                # measure accuracy and record loss
                metric_val, metric_avg, *_ = update_metric_loss(input, output, metric, loss, losses)

                time_done = get_time_to_print(batch_time.avg * (i + 1))
                time_remaining = get_time_to_print(batch_time.avg * len(val_loader))

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) [{done} => {remaining}]\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {metric_val:.3f} ({metric_avg:.3f})'.format(
                    epoch, i + 1, len(val_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, metric_val=metric_val,
                    metric_avg=metric_avg,
                    done=time_done, remaining=time_remaining)
                )
                sys.stdout.flush()

    return losses.avg, metric_avg
