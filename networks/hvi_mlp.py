import torch
#from semi_imp_bnn.bayes_layer import BayesianLinear
from utils import *
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, inputdim, layer_size, outputdim, droprate = None):
        super(Net, self).__init__()
        self.droprate =droprate
        
        if self.droprate is not None:
            self.drop = torch.nn.Dropout(droprate)
        hid1, hid2 = layer_size
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(inputdim, hid1)
        self.fc2 = torch.nn.Linear(hid1, hid2)
        self.fc3 = torch.nn.Linear(hid2, outputdim)
        self.fc4 = torch.nn.Linear(hid2, 3)
        self.fc5 = torch.nn.Linear(3, outputdim)


    def forward(self,x):
        h=x.view(x.size(0),-1)
        if self.droprate is not None:
            h=self.drop(self.relu(self.fc1(h)))
            h=self.drop(self.relu(self.fc2(h)))
        else:
            h=self.relu(self.fc1(h))
            h=self.relu(self.fc2(h))
        mu = self.fc3(h)
        rho = self.relu(self.fc4(h))
        rho = self.fc5(rho)
        return mu, rho

