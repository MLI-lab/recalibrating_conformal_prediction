from pathlib import Path, PosixPath
import shutil
import json
from typing import Callable, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset




class ImageNetEncodings(Dataset):
    """ImageNet encodings from a convolutional network"""

    def __init__(self, pathname: Path, train = True, target_transform: Callable = None):
        """
        Args:
            pathname (pathlib.Path): Path to the npz file.
            train (bool): Extract train (True) or test (False) set from the file.
            target_transform (callable): Operation on the int class index.
        """
        self.samples, self.targets = np_encoding_loader(pathname.resolve(), train=train)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



def np_encoding_loader(filename, train=True):
    data = np.load(filename)
    if train:
        return data['X_train'], data['y_train']
    else:
        return data['X_test'], data['y_test']



def get_linear_model(checkpoint, num_filters=2048, num_classes=1000):
    """initiate model and directly load state from checkpoint"""

    model=nn.Linear(num_filters, num_classes)
    print('Loading from the checkpoint at {}...'.format(checkpoint))
    weights_update(
        model=model,
        checkpoint=torch.load(checkpoint)
    )

    return model


def weights_update(model, checkpoint):

    with torch.no_grad():
        # get the current model state dict
        model_dict = model.state_dict()

        # Pytorch lightning (PL) saves model state dict with `model.` prefix. Remove this prefix.
        checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        checkpoint_state_dict = {k.replace('model.', ''): v for k, v in checkpoint_state_dict.items()}

        # PL also saves other PL Module parameters in the state dict. Only consider model parameters.
        pretrained_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_dict}

        overridden_params = list(pretrained_dict.keys())
        if len(overridden_params) < 10:
            print('The following model parameters will be overridden from the checkpoint state:\t' + '\t'.join(overridden_params))
        else:
            print('{} paramaters will be overridden from the checkpoint state'.format(len(overridden_params)))

        # update the model from the proper state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)



def cifar_match_state_dict(state_dict):

    # fc --> linear and downsample --> shortcut
    checkpoint_state_dict = {k.replace('fc.', 'linear.').replace('downsample.', 'shortcut.'): v for k, v in state_dict.items()}

    return checkpoint_state_dict



class CandidateDataset(Dataset):
    """Candidate dataset"""
    
    def __init__(self, pathname, transform=None, train=True):
        """
        Args:
            pathname (pathlib.Path): Path to the npz file.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (boolean): Extract train (True) or test (False) set from the file.
        """
        self.samples, self.targets = np_loader(pathname.resolve(), train=train)
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.samples[index], self.targets[index]
        sample = Image.fromarray(np.moveaxis(sample, 0, -1))
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        # TODO: Target transform.
        
        return sample, target
    
def np_loader(filename, train=True):
    data = np.load(filename)
    if train:
        samples = data['X_train'].transpose(0, 3, 1, 2)
        targets = data['y_train']
    else:
        samples = data['X_test'].transpose(0, 3, 1, 2)
        targets = data['y_test']
    return samples, targets


def convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.
    """
    assert (
        torch.max(targets).item() < classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = torch.zeros(
        (targets.shape[0], classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.decay_rate ** (epoch // args.decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: Union[None, str] = None, fmt: str = 'f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


def save_checkpoint(state, is_best, outdir=None, filename='checkpoint.pth.tar'):
    outdir = outdir if outdir is not None else Path.cwd()
    torch.save(state, outdir / filename)
    if is_best:
        shutil.copyfile(filename, outdir / 'model_best.pth.tar')


def save_config(args):
    config_file = args.outpath / 'config.json'
    
    config_dict = {k:(str(v) if isinstance(v, PosixPath) else v) for k,v in args.__dict__.items()}
    
    with open(config_file, 'w') as fn:
        json.dump(config_dict, fn, indent=2)