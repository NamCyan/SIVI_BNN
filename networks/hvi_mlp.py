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
        #self.weight_init()
        print("fc1 w: {}".format(self.fc1.weight))
        

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

    def xavier_init(self,w):
        if isinstance(w, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(w.weight)
            w.bias.data.fill_(0.01)

    def weight_init(self):
        for (_,layer) in self.named_children():
            #self.xavier_init(layer)
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    layer.weight= torch.nn.Parameter(torch.Tensor(layer.weight* -3.143/-0.0137))
                    layer.bias= torch.nn.Parameter(torch.Tensor(layer.bias * -3.143/-0.0137))

