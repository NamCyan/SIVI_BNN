import torch
import numpy as np
from utils import *
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


"""
Tune hyperparameters:
    prior_w: pi, sigma1, sigma2
    learning_rate, batch_size
    layer_size: implicit and inference model
"""



###############################################



##################init
"""
better exp:
  relu -> leaky


"""
tstart = time.time()
# Seed
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
else:
    print('[CUDA unavailable]'); sys.exit()

##################################################
"""
#train 97.x, valid 96.x, test 96.x
lr = 0.001
nepochs = 150
sbatch = 256
lr_min=2e-6
lr_factor=3
lr_patience=5
nosample = 10
hid_layer = 100
optim = 'Adam'
SIVI_layer_size = 100
SIVI_input_dim = 40
semi_unit = False
prior_gmm = False
semi_by_col = False

#best: 98.5, valid 97.5, test 97.4
lr = 0.001
nepochs = 300
sbatch = 256
lr_min=2e-6
lr_factor=3
lr_patience=5
nosample = 10
hid_layer = 100
optim = 'Adam'
SIVI_layer_size = 400
SIVI_input_dim = 100
semi_unit = False
prior_gmm = False
semi_by_col = False
"""


lr_min = 2e-6
lr_factor=3
lr_patience=5
optim = 'Adam'

semi_unit = False
droprate = None
experiment = 'uci'
local_rep = True
prior_gmm = False #prior is gau mixture (True) or gau unit 0,1 (False)
semi_by_col = False #if true: semi followed by cols else followed by rows

###################load data
print("Load data...")
if experiment == 'mnist':
    print('MNIST DATASET')
    from dataloader import mnist as data
    from networks.SIVI_bnn import Net
    from approach.sivi_bnn import Appr as appr
    dat, input_dim = data.get() #read data
    #hyper parameters
    nepochs = 300
    lr = 0.0001
    hid_layer = 400
    SIVI_layer_size = 400
    SIVI_input_dim = 100
    sbatch = 256
    nosample = 10
    output_dim =10
elif experiment == 'uci':
    from dataloader import uci as data
    from networks.SIVI_bnn_UCI import Net
    from approach.sivi_bnn_uci import Appr as appr
    data_type = 'BOSTON'

    if data_type == 'BOSTON':
        nepochs, n_splits, tau = 40, 20, [0.1,0.15,0.2]
    elif data_type == 'CONCRETE':
        nepochs, n_splits, tau = 40, 20, [0.025,0.05,0.075]
    elif data_type == 'ENERGY':
        nepochs, n_splits, tau = 40, 20, [0.25,0.5,0.75]
    elif data_type == 'KININ8NM:
        nepochs, n_splits, tau = 40, 20, [150,200,250]
    elif data_type == 'NAVAL':
        nepochs, n_splits, tau = 40, 20, [30000,40000,50000]
    elif data_type == 'POWERPLANT':
        nepochs, n_splits, tau = 40, 20, [0.05,0.1,0.15]
    elif data_type == 'PROTEIN':
        nepochs, n_splits, tau = 20, 5, [0.025,0.05,0.075]
    elif data_type == 'WINE':
        nepochs, n_splits, tau = 40, 20, [2.5,3.0,3.5]
    elif data_type == 'YACHT':
        nepochs, n_splits, tau = 40, 20, [0.25,0.5,0.75]

    if data_type == 'PROTEIN' or data_type == 'YEAR':
        hid_layer = 100
    else:
        hid_layer = 50

    print('UCI DATASET: '+ data_type)
    dat, input_dim, normalization_info= data.get(data_type, split_id) #read data
    #hyper parameters
    lr = 0.001
    SIVI_layer_size = 100
    SIVI_input_dim = 40
    sbatch = 5
    nosample = 100
    output_dim =1
    

xtrain = dat['train']['x'].cuda()
ytrain = dat['train']['y'].cuda()

xvalid = dat['valid']['x'].cuda()
yvalid = dat['valid']['y'].cuda()

xtest = dat['test']['x'].cuda()
ytest = dat['test']['y'].cuda()

print(xvalid.shape, yvalid.shape)
print('Done!')
print('\n'+ '*'*200)
#####################
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#init model and apply approach
if experiment == 'mnist':
    model = Net(input_dim, [hid_layer,hid_layer], output_dim, prior_gmm=prior_gmm,SIVI_by_col=semi_by_col, SIVI_layer_size=SIVI_layer_size, SIVI_input_dim=SIVI_input_dim, semi_unit= semi_unit, droprate= droprate, local_rep= local_rep).cuda()
elif experiment == 'uci':
    model = Net(input_dim, [hid_layer,hid_layer], output_dim, prior_gmm=prior_gmm,SIVI_by_col=semi_by_col, SIVI_layer_size=SIVI_layer_size, SIVI_input_dim=SIVI_input_dim, semi_unit= semi_unit, droprate= droprate, local_rep= local_rep,tau= tau[0]).cuda()
    for split_id in range(n_splits):
      
Appr = appr(model, optim= optim, nosample= nosample, nepochs= nepochs, sbatch= sbatch, lr=lr, lr_min=lr_min, lr_factor=lr_factor, lr_patience=lr_patience)

# model info
print_model_report(model)
print_optimizer_config(Appr.optimizer)

print("Parameters:")
cnt_param = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        cnt_param += 1
        print("\t"+ name)
print('Number of parameters: '+ str(cnt_param))
print('*'*200)

##################### TRAIN
print('TRAINING')
if experiment == 'mnist':
    Appr.train(xtrain,ytrain,xvalid,yvalid,xtest, ytest)
else:
    Appr.train(xtrain,ytrain,xvalid,yvalid,xtest, ytest, normalization_info)

print('*'*200)
###################### TEST
print('TESTING')
if experiment == 'mnist':
    test_loss, test_acc = Appr.eval(xtest, ytest)
    print("Test: loss= {:.3f}, acc={:5.1f}".format(test_loss,100*test_acc),end= '')
else:
    test_loss, test_rmse, test_llh = Appr.eval(xtest, ytest,normalization_info)
    print("Test: loss= {:.3f}, rmse={:5.1f}, llh={:.3f}".format(test_loss,test_rmse,test_llh),end= '')
