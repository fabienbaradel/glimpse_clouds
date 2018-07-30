from data.ntu.dataloader import NTU, my_collate
import torch
import ipdb

def get_datasets_n_loaders(options, cuda=False):
    # NTU Dataset
    train_set = NTU(root=options['root'],
                    t=options['time'],
                    dataset='train',
                    train=True,
                    usual_transform=True)
    val_set = NTU(root=options['root'],
                  t=options['time'],
                  dataset='test',
                  train=False,
                  usual_transform=True)

    # Pytorch dataloader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=options['batch_size'],
                                               num_workers=options['workers'],
                                               pin_memory=cuda,
                                               collate_fn=my_collate)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=options['batch_size'],
                                             num_workers=options['workers'],
                                             pin_memory=cuda,
                                             collate_fn=my_collate)

    return train_set, val_set, train_loader, val_loader
