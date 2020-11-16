import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2d
from bayes_layer import BayesianLinear


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(nn.Module):
    def __init__(self, inputsize, output_dim, prior_gmm= True, pi= 0.5, sig_gau1=1, sig_gau2=np.exp(-6), sample=True, ratio= 0.25):
        super().__init__()
        self.output_dim = output_dim
        self.sample = sample
        ncha,size,_=inputsize
        d1, d2 = 32, 128
        self.conv1 = BayesianConv2d(ncha,d1, kernel_size=7, padding=0, prior_gmm= prior_gmm, pi= pi, sig_gau1= sig_gau1, sig_gau2=sig_gau2, ratio=ratio)
        #self.bn1 = torch.nn.BatchNorm2d(d1)
        s = compute_conv_output_size(size,7, padding=0) # 32,22,22
        s = s//2 # 11
        self.conv2 = BayesianConv2d(d1,d2,kernel_size=5, padding=0, prior_gmm= prior_gmm, pi= pi, sig_gau1= sig_gau1, sig_gau2=sig_gau2, ratio=ratio)
        #self.bn2 = torch.nn.BatchNorm2d(d2)
        s = compute_conv_output_size(s,5, padding=0) # 128,7,7
        s = s//2 # 3
        self.bfc1= BayesianLinear(s*s*d2,output_dim, prior_gmm= prior_gmm, pi= pi, sig_gau1= sig_gau1, sig_gau2=sig_gau2, ratio = ratio)

        self.MaxPool = torch.nn.MaxPool2d(2)  
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        h= self.lrelu(self.conv1(x, sample=self.sample))
        h= self.MaxPool(h)
        h= self.lrelu(self.conv2(h, sample=self.sample))
        h= self.MaxPool(h)
        h=h.view(x.size(0),-1)
        y = self.bfc1(h, sample=self.sample)
        
        return y

    def loss_forward(self, x,y, N_M, no_sample):
        total_qw, total_pw, total_log_likelihood = 0., 0., 0.
        out = torch.zeros([x.shape[0],self.output_dim])
        for i in range(no_sample):
            output = F.log_softmax(self.forward(x), dim= 1)       
            total_qw += self.get_qw()
            total_pw += self.get_pw()
            out += output
            
        total_qw = total_qw/no_sample
        total_pw = total_pw/no_sample
        total_log_likelihood = F.nll_loss(out/no_sample, y, reduction='sum')
        
        loss = (total_qw - total_pw)/N_M + total_log_likelihood
        return loss, out/no_sample

    def pred_sample(self, x,y, no_sample):
        output = torch.zeros([len(x),self.output_dim])
        for i in range(no_sample):
            output_ = F.log_softmax(self.forward(x), dim= 1)
            output += output_
        return output/no_sample

    def get_pw(self):
        log_pw = 0
        for (_, layer) in self.named_children():
            if isinstance(layer, BayesianLinear)==False and isinstance(layer, BayesianConv2d)==False:
                continue
            log_pw += layer.p_w

        return log_pw

    def get_qw(self):
        log_qw = 0
        for (_, layer) in self.named_children():
            if isinstance(layer, BayesianLinear)==False and isinstance(layer, BayesianConv2d)==False:
                continue
            log_qw += layer.q_w

        return log_qw

if __name__ == "__main__":
    inp = torch.Tensor(2,3,32,32).uniform_(0.1,0.1).cuda()

    inputsize = [3,32,32]
    output_dim = 10
    SIVI_input_dim, SIVI_layer_size = 10, 20
    net = Net(inputsize, output_dim, SIVI_input_dim, SIVI_layer_size).cuda()
    out = net(inp)
    print(out)