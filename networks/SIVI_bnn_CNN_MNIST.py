import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import SIVI_BayesianConv2d
from bayes_layer import SIVI_bayes_layer


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class Net(nn.Module):
    def __init__(self, inputsize, output_dim, SIVI_input_dim, SIVI_layer_size, prior_gmm= True, local_rep= False, droprate= None, ratio= 0.25):
        super().__init__()
        self.local_rep = local_rep
        self.output_dim = output_dim
        
        ncha,size,_=inputsize
        d1, d2 = 32, 128
        self.conv1 = SIVI_BayesianConv2d(ncha,d1, kernel_size=7, padding=0, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, droprate= droprate, ratio=ratio)
        #self.bn1 = torch.nn.BatchNorm2d(d1)
        s = compute_conv_output_size(size,7, padding=0) # 32,22,22
        s = s//2 # 11
        self.conv2 = SIVI_BayesianConv2d(d1,d2,kernel_size=5, padding=0, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, droprate= droprate, ratio=ratio)
        #self.bn2 = torch.nn.BatchNorm2d(d2)
        s = compute_conv_output_size(s,5, padding=0) # 128,7,7
        s = s//2 # 3
        self.bfc1= SIVI_bayes_layer(s*s*d2,output_dim, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, droprate= droprate, ratio = ratio)

        self.MaxPool = torch.nn.MaxPool2d(2)  
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, sample=True, train= True):
        h= self.lrelu(self.conv1(x, sample=sample, train= train))
        h= self.MaxPool(h)
        h= self.lrelu(self.conv2(h, sample=sample, train= train))
        h= self.MaxPool(h)
        h=h.view(x.size(0),-1)
        y = self.bfc1(h, local_rep=self.local_rep, sample=sample, train= train)
        
        return y

#     def __init__(self, inputsize, output_dim, SIVI_input_dim, SIVI_layer_size, prior_gmm= True, local_rep= False, droprate= None, ratio= 0.25):
#         super().__init__()
#         self.local_rep = local_rep
#         self.output_dim = output_dim
        
#         ncha,size,_=inputsize
        
#         self.conv1 = SIVI_BayesianConv2d(ncha,32, kernel_size=3, padding=1, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, ratio=ratio)
#         s = compute_conv_output_size(size,3, padding=1) # 32
#         self.conv2 = SIVI_BayesianConv2d(32,32,kernel_size=3, padding=1, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, ratio=ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 32
#         s = s//2 # 16
#         self.conv3 = SIVI_BayesianConv2d(32,64,kernel_size=3, padding=1, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, ratio=ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 16
#         self.conv4 = SIVI_BayesianConv2d(64,64,kernel_size=3, padding=1, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, ratio=ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 16
#         s = s//2 # 8
#         self.conv5 = SIVI_BayesianConv2d(64,128,kernel_size=3, padding=1, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, ratio=ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 8
#         self.conv6 = SIVI_BayesianConv2d(128,128,kernel_size=3, padding=1, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, ratio=ratio)
#         s = compute_conv_output_size(s,3, padding=1) # 8
# #         self.conv7 = SIVI_BayesianConv2d(128,128,kernel_size=3, padding=1, ratio)
# #         s = compute_conv_output_size(s,3, padding=1) # 8
#         s = s//2 # 4
#         self.bfc1 = SIVI_bayes_layer(s*s*128,256, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, droprate= droprate, ratio = ratio)
        
#         self.bfc2= SIVI_bayes_layer(256,output_dim, SIVI_input_dim= SIVI_input_dim, SIVI_layer_size = SIVI_layer_size, prior_gmm= prior_gmm, droprate= droprate, ratio = ratio)

#         self.drop1 = nn.Dropout(0.25)
#         self.drop2 = nn.Dropout(0.5)
#         self.MaxPool = torch.nn.MaxPool2d(2)  
#         self.relu = torch.nn.ReLU()

#     def forward(self, x, sample=True, train= True):
#         h=self.relu(self.conv1(x,sample=sample, train= train))
#         h=self.relu(self.conv2(h,sample=sample, train= train))
#         h=self.drop1(self.MaxPool(h))
#         h=self.relu(self.conv3(h,sample=sample, train= train))
#         h=self.relu(self.conv4(h,sample=sample, train= train))
#         h=self.drop1(self.MaxPool(h))
#         h=self.relu(self.conv5(h,sample=sample, train= train))
#         h=self.relu(self.conv6(h,sample=sample, train= train))
# #         h=self.relu(self.conv7(h,sample=sample, train= train))
#         h=self.drop1(self.MaxPool(h))
#         h=h.view(x.size(0),-1)
#         h = self.drop2(self.relu(self.bfc1(h, local_rep=self.local_rep, sample=sample, train= train)))
#         y = self.bfc2(h, local_rep=self.local_rep, sample=sample, train= train)
        
#         return y

    def loss_forward(self, x,y, N_M, no_sample):     
        output = F.log_softmax(self.forward(x), dim=1)
    
        log_likelihood = F.nll_loss(output, y, reduction='sum')
        log_pw = self.get_log_pw()
        log_qw = self.get_log_qw(no_sample)

        loss =  log_likelihood + (log_qw - log_pw)/N_M
        return loss

    def get_log_pw(self):
        log_pw = 0
        for (_, layer) in self.named_children():
            if isinstance(layer, SIVI_bayes_layer)==False and isinstance(layer, SIVI_BayesianConv2d)==False:
                continue
            log_pw += layer.get_log_pw()

        return log_pw

    def get_log_qw(self, no_sample):
        log_qw = 0
        for (_, layer) in self.named_children():
            if isinstance(layer, SIVI_bayes_layer)==False and isinstance(layer, SIVI_BayesianConv2d)==False:
                continue
            log_qw += layer.get_log_qw(no_sample)

        return log_qw
    
    def pred_sample(self, x, y, semi_sample, w_sample, test= False):
        if not test:
            semi_sample, w_sample = 1, 1

        out = torch.zeros([len(x),self.output_dim])
  
        for i in range(semi_sample):
            output = F.log_softmax(self.forward(x, train= True), dim=1)
            out += output
            for j in range(w_sample-1):
                output = F.log_softmax(self.forward(x, train= False), dim=1)
                out += output

        return out/(semi_sample*w_sample)

if __name__ == "__main__":
    inp = torch.Tensor(2,3,32,32).uniform_(0.1,0.1).cuda()

    inputsize = [3,32,32]
    output_dim = 10
    SIVI_input_dim, SIVI_layer_size = 10, 20
    net = Net(inputsize, output_dim, SIVI_input_dim, SIVI_layer_size).cuda()
    out = net(inp)
    print(out)