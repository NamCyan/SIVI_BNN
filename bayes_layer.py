import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks import hvi_mlp ,mlp
from torch.nn.modules.utils import _single, _pair, _triple
from utils import *

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

################################################################################
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu.cuda()
        self.rho = rho.cuda()
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.mu.size()).cuda()
        return self.mu + self.sigma * epsilon

################################################################################
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_gmm= True, pi= 0.5, sig_gau1=1, sig_gau2=np.exp(-6), ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #prior gmm 
        self.prior_gmm = prior_gmm
        self.pi = pi
        self.sig_gau1 = sig_gau1
        self.sig_gau2 = sig_gau2

        self.q_w = 0
        self.p_w = 0

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        gain = 1  # Var[w] + sigma^2 = 2/fan_in

        total_var = 2 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)

        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        #nn.init.uniform_(self.bias_mu, -0, 0)

        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(rho_init, rho_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(rho_init, rho_init))
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    def forward(self, input, sample=False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        #prior
        pi = self.pi
        sig_gau1 = self.sig_gau1
        sig_gau2 = self.sig_gau2
        self.p_w = log_prior_w(weight,Pi= pi, sigma1= sig_gau1, sigma2= sig_gau2 ,gmm= self.prior_gmm).sum() + log_prior_w(bias,Pi= pi, sigma1= sig_gau1, sigma2= sig_gau2 ,gmm= self.prior_gmm).sum()
        #variational dist
        self.q_w = log_gauss(weight, self.weight_mu, self.weight_rho, rho=True).sum() + log_gauss(bias, self.bias_mu, self.bias_rho, rho=True).sum()
        
        return F.linear(input, weight, bias)

################################################################################
class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, transposed= False, output_padding= 0, groups=1, bias=True, prior_gmm= True, pi = 0.5,  sig_gau1=1, sig_gau2=np.exp(-6), ratio= 0.25):
        super(BayesianConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.prior_gmm = prior_gmm

        self.pi = pi
        self.sig_gau1 = sig_gau1
        self.sig_gau2 = sig_gau2

        self.p_w = 0
        self.q_w = 0

        _, fan_out = _calculate_fan_in_and_fan_out(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        total_var = 2 / fan_out
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)
        
        #self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(0, 0), requires_grad=bias)
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, (in_channels // groups)*kernel_size*kernel_size + 1).uniform_(rho_init, rho_init))
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, (in_channels // groups)*kernel_size*kernel_size + 1).uniform_(-bound, bound))
        
    def forward(self, net_input, sample= True):
        if sample:
            self.weight_n_bias = Gaussian(self.weight_mu, self.weight_rho).sample()
        else:
            self.weight_n_bias = self.weight_mu

        weight = self.weight_n_bias[:,0:-1].reshape(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        bias = self.weight_n_bias[:,-1]

        pi = self.pi
        sig_gau1 = self.sig_gau1
        sig_gau2 = self.sig_gau2

        self.p_w = log_prior_w(self.weight_n_bias,Pi= pi, sigma1= sig_gau1, sigma2= sig_gau2 ,gmm= self.prior_gmm).sum()
        self.q_w = log_gauss(self.weight_n_bias, self.weight_mu, self.weight_rho, rho=True).sum()

        return F.conv2d(net_input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

################################################################################
class SIVI_bayes_layer(nn.Module):
    def __init__(self, in_features, out_features, prior_gmm= True, SIVI_input_dim = 10, SIVI_layer_size = 20, SIVI_by_col = False, semi_unit = False, droprate= None,ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.SIVI_by_col = SIVI_by_col
        self.prior_gmm = prior_gmm

        fan_in = in_features

        total_var = 2 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)

        # bound = 0.1
        # rho_init = -3

        if(SIVI_by_col):
            self.SIVI = mlp.Net(SIVI_input_dim,[SIVI_layer_size,SIVI_layer_size], out_features, droprate)
            if semi_unit:
                rho_init = np.log(np.e - 1)
                self.semi_mu = torch.Tensor(in_features+1, SIVI_input_dim).uniform_(0, 0)
                self.semi_rho = torch.Tensor(in_features+1, SIVI_input_dim).uniform_(rho_init, rho_init)
            else:
                self.semi_mu = nn.Parameter(torch.Tensor(in_features+1, SIVI_input_dim).uniform_(-bound, bound))
                self.semi_rho = nn.Parameter(torch.Tensor(in_features+1, SIVI_input_dim).uniform_(rho_init, rho_init))
        else:
            self.SIVI= mlp.Net(SIVI_input_dim,[SIVI_layer_size,SIVI_layer_size], in_features+1, droprate)
            if semi_unit:
                rho_init = np.log(np.e - 1)
                self.semi_mu = torch.Tensor(out_features, SIVI_input_dim).uniform_(0, 0)
                self.semi_rho = torch.Tensor(out_features, SIVI_input_dim).uniform_(rho_init, rho_init)
            else:
                self.semi_mu = nn.Parameter(torch.Tensor(out_features, SIVI_input_dim).uniform_(-bound, bound))
                self.semi_rho = nn.Parameter(torch.Tensor(out_features, SIVI_input_dim).uniform_(rho_init, rho_init))

        #self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features+1).uniform_(rho_init, rho_init))
        self.weight_rho = torch.Tensor(out_features, in_features+1).uniform_(rho_init, rho_init)
        
        self.semi_input = Gaussian(self.semi_mu, self.semi_rho)


    def forward(self, net_input, local_rep = True, train= True, sample= True):
        if train:
            semi_input = self.semi_input.sample()
        
            if self.SIVI_by_col:
                self.weight_mu = torch.transpose(self.SIVI(semi_input),0,1)
            else:
                self.weight_mu = self.SIVI(semi_input)

        sigma = torch.log1p(torch.exp(self.weight_rho))
        epsilon = torch.distributions.Normal(0, sigma[0][0]).sample(self.weight_rho.size()).cuda()
        self.weight_n_bias = self.weight_mu + epsilon
        #self.weight_n_bias = Gaussian(self.weight_mu, self.weight_rho).sample()
        
        if local_rep:
            one = torch.ones(net_input.size()[0],1).cuda()
            net_input = torch.cat([net_input,one],axis=1)
            
            mu_preact = torch.mm(net_input,torch.transpose(self.weight_mu,0,1))
            weight_sigma =  torch.log1p(torch.exp(self.weight_rho))
            sigma_preact = torch.sqrt(torch.mm(net_input**2,torch.transpose(weight_sigma,0,1)**2)+ 1e-6)
            
            eps = torch.distributions.Normal(0, 1).sample(mu_preact.size()).cuda()
            out = sigma_preact*eps + mu_preact
            
        else:
            weight = self.weight_n_bias[:,0:-1]
            bias = self.weight_n_bias[:,-1]
            out = F.linear(net_input, weight, bias)
          
        #self.log_p_w = torch.sum(log_prior_w(self.weight_n_bias, gmm= self.prior_gmm))
        #self.log_q_w = self.get_log_qw()
        return out

    def get_log_pw(self):
        return torch.sum(log_prior_w(self.weight_n_bias, gmm= self.prior_gmm))
        
    def get_log_qw(self, no_sample):
        axis = 1
        if self.SIVI_by_col:
            axis = 0
        
        sig = torch.log1p(torch.exp(self.weight_rho))
        N, M = sig.shape

        semi_input = torch.empty((0,))
        sigma = torch.cat([sig]*no_sample,0)
        weight = torch.cat([self.weight_n_bias]*no_sample,0)

        for i in range(no_sample-1):
            semi_input = torch.cat([semi_input,self.semi_input.sample()],0)

        weight_mu = self.SIVI(semi_input)
        weight_mu = torch.cat([self.weight_mu,weight_mu],0)
 
        q_w = torch.sum(-0.5*(weight-weight_mu)**2/sigma**2,axis=axis, keepdim=True)
        #q_w = torch.sum(log_gauss(weight,weight_mu, sigma,rho= False),axis=axis, keepdim=True)
        q_w = q_w.reshape(no_sample,self.weight_mu.shape[0])
        
        log_q_w = (torch.logsumexp(q_w,0)).sum() - torch.log(sig).sum() - (0.5*M*np.log(2*np.pi) + np.log(no_sample))*N
        #log_q_w = (torch.logsumexp(q_w,0)).sum() - np.log(no_sample)*N

        return log_q_w

            
        # for i in range(no_sample):
        #     if i == 0:
        #         q_w = torch.sum(-0.5*(self.weight_n_bias-self.weight_mu)**2/sig**2,axis=axis, keepdim=True)

        #     else:
        #         semi_input = self.semi_input.sample()

        #         if self.SIVI_by_col:
        #             weight_mu = torch.transpose(self.SIVI(semi_input),0,1)
        #         else:
        #             weight_mu = self.SIVI(semi_input)

        #         q_w = torch.cat((q_w, torch.sum(-0.5*(self.weight_n_bias - weight_mu) ** 2 / sig ** 2, axis=axis, keepdim=True)), axis=axis)
        
        # log_q_w = (torch.logsumexp(q_w,dim=axis)).sum() - torch.log(sig).sum() - (0.5*sig.size()[-axis]*np.log(2*np.pi) + np.log(no_sample))*sig.size()[axis-1]
        # return log_q_w

    # def sample_wmu(self, no_sample):
    #     wmu = []
    #     for i in range(no_sample):
    #         semi_input = self.semi_input.sample()

    #         if self.SIVI_by_col:
    #             self.weight_mu = torch.transpose(self.SIVI(semi_input),0,1)
    #         else:
    #             self.weight_mu = self.SIVI(semi_input)
            
    #         wmu.append(self.weight_mu)
    #     return wmu

################################################################################
class SIVI_BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, SIVI_input_dim,SIVI_layer_size, kernel_size,
                 stride=1, padding=0, dilation=1, transposed= False, output_padding= 0, groups=1, bias=True, droprate= None, prior_gmm= True, ratio= 0.25):
        super(SIVI_BayesianConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.prior_gmm = prior_gmm

        _, fan_out = _calculate_fan_in_and_fan_out(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        total_var = 2 / fan_out
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)

        self.SIVI= mlp.Net(SIVI_input_dim,[SIVI_layer_size,SIVI_layer_size], (in_channels // groups)*kernel_size*kernel_size + 1, droprate= droprate)
        self.semi_mu = nn.Parameter(torch.Tensor(out_channels, SIVI_input_dim).uniform_(-bound, bound))
        self.semi_rho = nn.Parameter(torch.Tensor(out_channels, SIVI_input_dim).uniform_(rho_init, rho_init))

        
        #self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(0, 0), requires_grad=bias)
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, (in_channels // groups)*kernel_size*kernel_size + 1).uniform_(rho_init, rho_init))
        self.semi_input = Gaussian(self.semi_mu, self.semi_rho)
    
    def forward(self, net_input, sample= True, train= True):
        if train:
            semi_input = self.semi_input.sample()
    
            self.weight_mu = self.SIVI(semi_input)

        self.weight_n_bias = Gaussian(self.weight_mu, self.weight_rho).sample()
        weight = self.weight_n_bias[:,0:-1].reshape(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        bias = self.weight_n_bias[:,-1]

        return F.conv2d(net_input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def get_log_pw(self):
        return torch.sum(log_prior_w(self.weight_n_bias, gmm= self.prior_gmm))

    def get_log_qw(self, no_sample):
        axis = 1

        sig = torch.log1p(torch.exp(self.weight_rho))
        N, M = sig.shape

        sigma = torch.cat([sig]*no_sample,0)
        weight = torch.cat([self.weight_n_bias]*no_sample,0)
        
        semi_input = torch.empty((0,))
        for i in range(no_sample-1):
            semi_input = torch.cat([semi_input,self.semi_input.sample()],0)

        weight_mu = self.SIVI(semi_input)
        weight_mu = torch.cat([self.weight_mu,weight_mu],0)
 
        q_w = torch.sum(-0.5*(weight-weight_mu)**2/sigma**2,axis=axis, keepdim=True)
        q_w = q_w.reshape(no_sample,self.weight_mu.shape[0])
        
        log_q_w = (torch.logsumexp(q_w,0)).sum() - torch.log(sig).sum() - (0.5*M*np.log(2*np.pi) + np.log(no_sample))*N
        return log_q_w

        # for i in range(no_sample):
        #     if i == 0:
        #         q_w = torch.sum(-0.5*(self.weight_n_bias-self.weight_mu)**2/sig**2,axis=axis, keepdim=True)

        #     else:
        #         semi_input = self.semi_input.sample()
        #         weight_mu = self.SIVI(semi_input)

        #         q_w = torch.cat((q_w, torch.sum(-0.5*(self.weight_n_bias - weight_mu) ** 2 / sig ** 2, axis=axis, keepdim=True)), axis=axis)
        
        # log_q_w = (torch.logsumexp(q_w,dim=axis)).sum() - torch.log(sig).sum() - (0.5*sig.size()[-axis]*np.log(2*np.pi) + np.log(no_sample))*sig.size()[axis-1]
        # return log_q_w

################################################################################
class _BayesianConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, transposed, output_padding, groups, bias, ratio):
        super(_BayesianConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        _, fan_out = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2 / fan_out
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)


        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.bias = nn.Parameter(torch.Tensor(out_channels).uniform_(0, 0), requires_grad=bias)

        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).uniform_(rho_init, rho_init))

        self.weight = Gaussian(self.weight_mu, self.weight_rho)

################################################################################
# class BayesianConv2D(_BayesianConvNd):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True, ratio=0.25):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(BayesianConv2D, self).__init__(in_channels, out_channels, kernel_size,
#                                              stride, padding, dilation, False, _pair(0), groups, bias, ratio)
#     def forward(self, input, sample=False):
#         if sample:
#             weight = self.weight.sample()

#         else:
#             weight = self.weight.mu

#         return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
################################################################################
class HVI_BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_gmm= True, SIVI_input_dim = 10, SIVI_layer_size = 20, SIVI_by_col= False, semi_unit= False, droprate= None,ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_gmm = prior_gmm

        fan_in = in_features

        total_var = 2 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)

        # bound = 0.1
        # rho_init = -3
        
        self.SIVI = hvi_mlp.Net(SIVI_input_dim,[SIVI_layer_size,SIVI_layer_size], in_features+1, droprate)

        self.mu_0 = nn.Parameter(torch.Tensor(out_features, SIVI_input_dim).uniform_(-bound, bound))
        self.rho_0 = nn.Parameter(torch.Tensor(out_features, SIVI_input_dim).uniform_(rho_init, rho_init))

        self.epsilon = Gaussian(self.mu_0, self.rho_0)


    def forward(self, net_input, local_rep = True, train= True, sample= True):
        if train:
            self.epsilon_0 = self.epsilon.sample()

            self.weight_mu, self.weight_rho = self.SIVI(self.epsilon_0 )

        self.weight_n_bias = Gaussian(self.weight_mu, self.weight_rho).sample()
        
        if local_rep:
            one = torch.ones(net_input.size()[0],1).cuda()
            net_input = torch.cat([net_input,one],axis=1)
            
            mu_preact = torch.mm(net_input,torch.transpose(self.weight_mu,0,1))
            weight_sigma =  torch.log1p(torch.exp(self.weight_rho))
            sigma_preact = torch.sqrt(torch.mm(net_input**2,torch.transpose(weight_sigma,0,1)**2)+ 1e-6)
            
            eps = torch.distributions.Normal(0, 1).sample(mu_preact.size()).cuda()
            out = sigma_preact*eps + mu_preact
            
        else:
            weight = self.weight_n_bias[:,0:-1]
            bias = self.weight_n_bias[:,-1]
            out = F.linear(net_input, weight, bias)
          
        #self.log_p_w = torch.sum(log_prior_w(self.weight_n_bias, gmm= self.prior_gmm))
        #self.log_q_w = self.get_log_qw()
        return out

    def get_log_pw(self):
        return torch.sum(log_prior_w(self.weight_n_bias, gmm= self.prior_gmm))
        
    def get_log_qw(self, no_sample):
        axis = 1

        weight = torch.cat([self.weight_n_bias]*no_sample,0)
        
        epsilon = torch.empty((0,))
        for i in range(no_sample-1):
            epsilon = torch.cat([epsilon,self.epsilon.sample()],0)

        weight_mu, weight_rho = self.SIVI(epsilon)
        weight_mu = torch.cat([self.weight_mu,weight_mu],0)
        weight_rho = torch.cat([self.weight_rho,weight_rho],0)
 
        q_w = torch.sum(log_gauss(weight,weight_mu, weight_rho, rho= True),axis=axis, keepdim=True)
        q_w = q_w.reshape(no_sample,self.weight_mu.shape[0])
        
        log_q_w = (torch.logsumexp(q_w,0)).sum() - np.log(no_sample)*self.weight_mu.shape[0]

        return log_q_w
