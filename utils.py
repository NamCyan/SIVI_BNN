import torch
import numpy as np
from copy import deepcopy
import math
#-------------------------
def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

#-------------------------
def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

#-------------------------------
def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def get_model(model):
    return deepcopy(model.state_dict())
#---------------------------------------------
class logger(object):
    def __init__(self, file_name='pmnist2', resume=False, path='./result_data/csvdata/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))
#-------------------------
def sample_gauss(mu, sig):
    epsilon = torch.distributions.Normal(0,1).sample(mu.shape)
    z = mu + epsilon*sig
    return z

def sample_model(inputdim, model, no_sample, prior_noise):
    mu, sig = prior_noise
    epsilon = torch.distributions.Normal(0, 1).sample(torch.Size([no_sample, inputdim]))*sig + mu
    return model(epsilon)

#----------------------
def gaussian_dist(var, mu, sig, rho= True):
    sig_ = sig
    if(rho):
        sig_ = torch.log1p(torch.exp(sig_))
    gauss = (1/(sig_*np.sqrt(2*np.pi)))*torch.exp(-0.5*(var-mu)**2/sig_**2)
    return gauss

def log_gauss(var, mu ,sig, rho=True):
    log2_pi = torch.Tensor([np.log(2*np.pi)]).cuda()
    if(rho):
        sig = torch.log1p(torch.exp(sig))
  
    return  -0.5*log2_pi - torch.log(sig) - 0.5* ((var - mu)**2 / sig**2)
#------------------------
def log_prior_w(w, Pi= 0.5, sigma1= 1, sigma2= np.exp(-6), gmm = True):
    if(gmm):
        sig1 = torch.Tensor(w.shape).uniform_(sigma1,sigma1).cuda()
        sig2 = torch.Tensor(w.shape).uniform_(sigma2,sigma2).cuda()
        mu = torch.Tensor(w.shape).uniform_(0,0).cuda()
        gau1 = log_gauss(w,mu,sig1, rho= False) + np.log(Pi)
        gau2 = log_gauss(w,mu,sig2, rho= False) + np.log(1-Pi)
        gau = torch.cat([gau1.unsqueeze(0),gau2.unsqueeze(0)], 0)
        #p = Pi*torch.exp(log_gauss(w,mu,sig1, rho= False)) + (1-Pi)*torch.exp(log_gauss(w,mu,sig2, rho= False))
        p = torch.logsumexp(gau,dim=0)
   
    else:
        log2_pi = torch.Tensor([np.log(2*np.pi)]).cuda()
        p = -0.5*log2_pi - 0.5*(w**2)
        
    return p
