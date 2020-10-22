from bayes_layer import SIVI_bayes_layer
import torch
import torch.nn.functional as F
from utils import *

class Net(torch.nn.Module):
    def __init__(self, inputdim, layer_size, outputdim, prior_gmm=True, SIVI_by_col= False, SIVI_input_dim=10, SIVI_layer_size=20, semi_unit= False, droprate= None, local_rep= True, ratio=0.5):
        super(Net, self).__init__()
        hid1, hid2 = layer_size
        self.outputdim = outputdim
        self.local_rep = local_rep
        self.SIVI_by_col = SIVI_by_col
        self.prior_gmm = prior_gmm
        
        self.fc1 = SIVI_bayes_layer(inputdim, hid1, prior_gmm= prior_gmm,SIVI_by_col=SIVI_by_col, SIVI_input_dim=SIVI_input_dim, SIVI_layer_size=SIVI_layer_size, semi_unit= semi_unit, droprate= droprate,ratio=ratio)
        self.fc2 = SIVI_bayes_layer(hid1, hid2, prior_gmm=prior_gmm,SIVI_by_col=SIVI_by_col, SIVI_input_dim=SIVI_input_dim, SIVI_layer_size=SIVI_layer_size, semi_unit= semi_unit, droprate= droprate,ratio=ratio)
        self.fc3 = SIVI_bayes_layer(hid2, outputdim, prior_gmm=prior_gmm, SIVI_by_col=SIVI_by_col, SIVI_input_dim=SIVI_input_dim, SIVI_layer_size=SIVI_layer_size, semi_unit= semi_unit, droprate= droprate,ratio=ratio)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        h = x.view(x.size(0),-1)
        h = self.relu(self.fc1(h, self.local_rep))
        h = self.relu(self.fc2(h, self.local_rep))
        y = self.fc3(h, self.local_rep)
        return y


    def loss_forward(self, x,y, N_M, no_sample):

        axis =1
        if(self.SIVI_by_col):
            axis = 0

        sig1, sig2, sig3 = torch.log1p(torch.exp(self.fc1.weight_rho)), torch.log1p(
            torch.exp(self.fc2.weight_rho)), torch.log1p(torch.exp(self.fc3.weight_rho))

        
        output = F.log_softmax(self.forward(x), dim=1)
        
        log_likelihood = F.nll_loss(output, y, reduction='sum')
        log_pw = self.get_log_pw()
        log_qw = self.get_log_qw(no_sample)

        loss =  log_likelihood + (log_qw - log_pw)/N_M

        return loss

    def get_log_pw(self):
        log_pw = self.fc1.get_log_pw() + self.fc2.get_log_pw()
        return log_pw

    def get_log_qw(self, no_sample):
        log_qw = self.fc1.get_log_qw(no_sample) + self.fc2.get_log_qw(no_sample) + self.fc3.get_log_qw(no_sample)
        return log_qw

    def pred_sample(self, x, y, no_sample):
        out = torch.zeros([len(x),self.outputdim])
  
        for i in range(no_sample):
            output = F.log_softmax(self.forward(x), dim=1)
            out += output

        return out/no_sample

if __name__ == "__main__":
    bnn = Net(28*28, [400,400], 10, SIVI_by_col=True)
    t = torch.Tensor(10, 28*28).uniform_(-0.5,0.5)
    out = bnn(t)
    print(out)
    print(F.softmax(out, dim=1))