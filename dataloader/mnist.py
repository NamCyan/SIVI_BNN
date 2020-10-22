import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle


########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0, tasknum=10):

    data = {}
    #taskcla = []
    size = [1, 28, 28]
    # Pre-load
    # MNIST
    mean = torch.Tensor([0.1307])
    std = torch.Tensor([0.3081])
    dat = {}
    dat['train'] = datasets.MNIST('./data/', train=True, download=True)
    dat['test'] = datasets.MNIST('./data/', train=False, download=True)

    for s in ['train', 'test']:
        if s == 'train':
            arr = dat[s].data.view(dat[s].data.shape[0], -1).float()
            label = torch.LongTensor(dat[s].targets)
        else:
            arr = dat[s].data.view(dat[s].data.shape[0], -1).float()
            label = torch.LongTensor(dat[s].targets)

        arr = (arr / 255 - mean) / std
        if(s == 'train'):
            data['train'] = {}
            data['valid'] = {}
            data['train']['x'] = arr.view(-1, size[0], size[1], size[2])[0:50000]
            data['train']['y'] = label[0:50000]
            data['valid']['x'] = arr.view(-1, size[0], size[1], size[2])[50000:]
            data['valid']['y'] = label[50000:]
        else:
            data['test'] = {}
            data['test']['x'] = arr.view(-1, size[0], size[1], size[2])
            data['test']['y'] = label



    return data, 28*28