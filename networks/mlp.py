import torch
#from semi_imp_bnn.bayes_layer import BayesianLinear
from utils import *
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, inputdim, layer_size, outputdim, droprate = None):
        super(Net, self).__init__()
        self.droprate =droprate
        self.inputdim = inputdim
        if self.droprate is not None:
            self.drop = torch.nn.Dropout(droprate)
        hid1, hid2 = layer_size
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(inputdim, hid1)
        self.fc2 = torch.nn.Linear(hid1, hid2)
        self.fc3 = torch.nn.Linear(hid2, outputdim)


    def forward(self,x):
        h=x.view(x.size(0),-1)
        if self.droprate is not None:
            h=self.drop(self.relu(self.fc1(h)))
            h=self.drop(self.relu(self.fc2(h)))
        else:
            h=self.relu(self.fc1(h))
            h=self.relu(self.fc2(h))
        mu = self.fc3(h)
        return mu
    
    def weight_init(self):
        for (_,layer) in self.named_children():
            #self.xavier_init(layer)
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    layer.weight= torch.nn.Parameter(torch.eye(self.inputdim))
                    layer.bias= torch.nn.Parameter(torch.Tensor(layer.bias * -3.143/-0.0137))

