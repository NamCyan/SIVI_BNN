from utils import *
import torch
import dataloader.mnist as data
import torch.nn.functional as F
import time
import numpy as np
from approach.bnn import Appr as appr


cnn = False
if cnn:
    from networks.baseline_cnn import Net
else:
    from networks.mlp_bnn import Net


tstart = time.time()
# Seed
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
else:
    print('[CUDA unavailable]'); sys.exit()

#########################init hyper parameters
#Approach hyper
lr = 0.0001
nepochs = 600
sbatch = 256
train_sample = 10
test_sample = 10
lr_min = 2e-6
lr_factor = 3
lr_patience = 5
optim = 'Adam' 

#model hyper
hid_layer = 400

prior_gmm = False
pi = 0.25
sig_gau1 = np.exp(0)
sig_gau2 = np.exp(-6)


###################load data
print("Load data...")
dat, input_dim, output_dim = data.get()
xtrain = dat['train']['x'].cuda()
ytrain = dat['train']['y'].cuda()

xvalid = dat['valid']['x'].cuda()
yvalid = dat['valid']['y'].cuda()

xtest = dat['test']['x'].cuda()
ytest = dat['test']['y'].cuda()
print("Data loaded!")
print('*' * 200)

##################### model infor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if cnn:
    model = Net(input_dim, output_dim, prior_gmm= prior_gmm, pi= pi, sig_gau1=sig_gau1, sig_gau2=sig_gau2,sample= True).cuda()
else:
    model = Net(input_dim, [hid_layer, hid_layer], output_dim, prior_gmm= prior_gmm, pi= pi, sig_gau1=sig_gau1, sig_gau2=sig_gau2,sample= True).cuda()
Appr = appr(model, optim= optim, nosample= train_sample, test_sample= test_sample, nepochs= nepochs, sbatch= sbatch, lr=lr, lr_min=lr_min, lr_factor=lr_factor, lr_patience=lr_patience)
# model info
print_model_report(model)
print_optimizer_config(Appr.optimizer)
print('\nParameter:')
for name, param in model.named_parameters():
    if param.requires_grad:
        print("\t"+name)
print('*' * 200)

#####################TRAIN
print('TRAINING')

Appr.train(xtrain,ytrain,xvalid,yvalid,xtest,ytest)

print('*' * 200)
print('Tesing')

test_loss, test_acc = Appr.eval(xtest, ytest)
print("Test: loss= {:.3f}, acc={:.3f}%".format(test_loss, 100 * test_acc), end='\n')