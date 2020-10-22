from bayes_layer import SIVI_bayes_layer
import torch
import torch.nn.functional as F
from utils import *

class Net(torch.nn.Module):
    def __init__(self, inputdim, layer_size, outputdim, prior_gmm=True, SIVI_by_col= False, SIVI_input_dim=10, SIVI_layer_size=20, semi_unit= False, droprate= None, local_rep=True, tau= 1.0, ratio=0.5):
        super(Net, self).__init__()
        hid1, hid2 = layer_size
        self.local_rep = local_rep
        self.SIVI_by_col = SIVI_by_col
        self.prior_gmm = prior_gmm
        self.tau = tau
        self.fc1 = SIVI_bayes_layer(inputdim, hid1, prior_gmm= prior_gmm,SIVI_by_col=SIVI_by_col, SIVI_input_dim=SIVI_input_dim, SIVI_layer_size=SIVI_layer_size, semi_unit= semi_unit, droprate= droprate,ratio=ratio)
        self.fc2 = SIVI_bayes_layer(hid1, outputdim, prior_gmm=prior_gmm,SIVI_by_col=SIVI_by_col, SIVI_input_dim=SIVI_input_dim, SIVI_layer_size=SIVI_layer_size, semi_unit= semi_unit, droprate= droprate,ratio=ratio)
    
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        h = x.view(x.size(0),-1)
        h = self.relu(self.fc1(h, self.local_rep))
        y = self.fc2(h, self.local_rep)
        return y


    def loss_forward(self, x,y, N_M, no_sample):

        tau = self.tau
        # axis =1
        # if self.SIVI_by_col:
        #     axis = 0

        # sig1, sig2 = torch.log1p(torch.exp(self.fc1.weight_rho)), torch.log1p(torch.exp(self.fc2.weight_rho))

        # w01, w02, log_likelihood, log_pw = 0,0,0,0
        # for i in range(no_sample):
        #     output = torch.transpose(self.forward(x),0,1)[0]
        
        #     if i == 0:
        #         w01 = self.fc1.weight_n_bias
        #         w02 = self.fc2.weight_n_bias
            
        #         log_likelihood = log_gauss(y,output,torch.Tensor([np.sqrt(1/tau)]),rho=False).sum()
        #         log_pw = self.get_log_pw()
                
        #         q_w1 = torch.sum(-0.5*(w01-self.fc1.weight_mu)**2/sig1**2,axis=axis, keepdim=True)
        #         q_w2 = torch.sum(-0.5*(w02-self.fc2.weight_mu)**2/sig2**2,axis=axis, keepdim=True)
        #     else:
        #         q_w1 = torch.cat((q_w1, torch.sum(-0.5*(w01 - self.fc1.weight_mu) ** 2 / sig1 ** 2, axis=axis, keepdim=True)), axis=axis)
        #         q_w2 = torch.cat((q_w2, torch.sum(-0.5*(w02 - self.fc2.weight_mu) ** 2 / sig2 ** 2, axis=axis, keepdim=True)), axis=axis)
  
        # q_w1 = (torch.logsumexp(q_w1,dim=axis)).sum() - torch.log(sig1).sum() - (0.5*sig1.size()[-axis]*np.log(2*np.pi) + np.log(no_sample))*sig1.size()[axis-1]
        # q_w2 = (torch.logsumexp(q_w2,dim=axis)).sum() - torch.log(sig2).sum() - (0.5*sig2.size()[-axis]*np.log(2*np.pi) + np.log(no_sample))*sig2.size()[axis-1]
        
        output = torch.transpose(self.forward(x),0,1)[0]
            
        log_likelihood = log_gauss(y,output,torch.Tensor([np.sqrt(1/tau)]),rho=False).sum()
        log_pw = self.get_log_pw()
        log_qw = self.get_log_qw(no_sample)
            
        loss = -log_likelihood + (log_qw - log_pw)/N_M
        
        return loss

    def get_log_pw(self):
        log_pw = self.fc1.get_log_pw() + self.fc2.get_log_pw()
        return log_pw

    def get_log_qw(self, no_sample):
        log_qw = self.fc1.get_log_qw(no_sample) + self.fc2.get_log_qw(no_sample)
        return log_qw

    def pred_sample(self, x, y, no_sample, normalization_info):
        output_mean = normalization_info['output mean'].cuda()
        output_std = normalization_info['output std'].cuda()
        
        out = torch.zeros(len(y)).cuda() #use to calculate rmse
        output_sample = torch.empty((0,)) #use to calculate loglikelihood

        for i in range(no_sample):
            output = torch.transpose(self.forward(x),0,1)
            output_sample = torch.cat([output_sample,output])
            out += output[0]*output_std + output_mean
        
        output_sample = output_sample* output_std + output_mean
        targets = y* output_std + output_mean
        
        rmse_part = (out/no_sample - targets)**2
        llh_part = torch.logsumexp(-0.5 * self.tau * (torch.unsqueeze(targets,0) - output_sample)**2, 0)
        return rmse_part, llh_part

if __name__ == "__main__":
    bnn = Net(28*28, 50, 1, SIVI_by_col=False).cuda()
    t = torch.Tensor(10, 28*28).uniform_(-0.5,0.5).cuda()
    out = bnn(t)
    print(out.transpose(0,1))