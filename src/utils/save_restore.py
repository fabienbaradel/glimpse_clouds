import torch
import shutil
import os
import ipdb

def save_checkpoint(state, is_best, resume, filename='checkpoint.pth.tar'):
    full_filename = os.path.join(resume, filename)
    torch.save(state, full_filename)
    if is_best:
        full_filename_best = os.path.join(resume, 'model_best.pth.tar')
        shutil.copyfile(full_filename, full_filename_best)


def load_from_dir(model, optimizer, options):
    ''' load from ckpt found in the dir'''
    best_metric_val, epoch = 0, 0
    if options['dir']:
        if os.path.isdir(options['dir']):
            ckpt_resume = os.path.join(options['dir'], 'model_best.pth.tar')
            if os.path.isfile(ckpt_resume):
                print("\n=> loading checkpoint '{}'".format(ckpt_resume))
                checkpoint = torch.load(ckpt_resume)
                epoch = checkpoint['epoch']
                try:
                    best_metric_val = checkpoint['best_metric_val']
                except:
                    best_metric_val = 0.0
                # Remove the fc_classifier
                updated_params = {}
                model_dict = model.state_dict()
                for k, v in checkpoint['state_dict'].items():
                    # Delete the module at the begining
                    if 'module' in k and not options['global_model']:
                        k = k.replace('module.', '')
                    # Train classifier fom scratch
                    if "fc_classifier" in k:
                        pass
                    # Look if the size if the same
                    if k in list(model_dict.keys()):
                        v_new_size, v_old_size = v.size(), model_dict[k].size()
                        if v_old_size == v_new_size:
                            updated_params[k] = v

                # Load
                new_params = model.state_dict()
                new_params.update(updated_params)
                model.load_state_dict(new_params)

                # Optim
                updated_params = {}
                new_params = optimizer.state_dict()
                for k, v in checkpoint['state_dict'].items():
                    if k not in list(new_params.keys()):
                        updated_params[k] = v

                new_params.update(updated_params)
                optimizer.load_state_dict(new_params)

                # Epoch
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(ckpt_resume, checkpoint['epoch']))
            else:
                print("\n=> no checkpoint found at '{}'".format(options['dir']))
                epoch = 0
        else:
            os.makedirs(options['dir'])

    return model, optimizer, best_metric_val, epoch


