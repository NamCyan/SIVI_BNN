import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0,pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,32,32]
    nclass = 10

    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    dat={}
    dat['train']=datasets.CIFAR10('./data/CIFAR_10/',train=True,download=True,
                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10('./data/CIFAR_10/',train=False,download=True,
                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

    data ={}
    for s in ['train','test']:
        data[s]= {}
        data[s]['x']= []
        data[s]['y']= []
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:     
            data[s]['x'].append(image)
            data[s]['y'].append(target.numpy()[0])
        data[s]['x']=torch.stack(data[s]['x']).view(-1,size[0],size[1],size[2])
        data[s]['y']=torch.LongTensor(np.array(data[s]['y'],dtype=int)).view(-1)

    r=np.arange(data['train']['x'].size(0))
    r=np.array(shuffle(r,random_state=seed),dtype=int)
    ntrain=int((1-pc_valid)*len(r))
    ivalid=torch.LongTensor(r[ntrain:])
    itrain=torch.LongTensor(r[:ntrain])
    data['valid']={}
    data['valid']['x']=data['train']['x'][ivalid].clone()
    data['valid']['y']=data['train']['y'][ivalid].clone()
    data['train']['x']=data['train']['x'][itrain].clone()
    data['train']['y']=data['train']['y'][itrain].clone()

    return data, size, nclass