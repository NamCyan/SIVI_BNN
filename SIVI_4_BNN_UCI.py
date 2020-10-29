import torch
import numpy as np
from utils import *
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

from dataloader import uci as data
from networks.SIVI_bnn_UCI import Net
from approach.sivi_bnn_uci import Appr as appr


########################################
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


lr_min = 1e-7
lr_factor=3
lr_patience=5
optim = 'Adam'

semi_unit = False
semi_by_col = False #if true: semi followed by cols else followed by rows

###################load data
print("Load data...")

data_type = 'BOSTON'
print('UCI DATASET: '+ data_type)

if data_type == 'BOSTON':
    nepochs, n_splits, tau = 40, 20, [0.1,0.15,0.2]
elif data_type == 'CONCRETE':
    nepochs, n_splits, tau = 40, 20, [0.025,0.05,0.075]
elif data_type == 'ENERGY':
    nepochs, n_splits, tau = 40, 20, [0.25,0.5,0.75]
elif data_type == 'KIN8NM':
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
output_dim =1
######################Tune parameters

nepochs = 200
droprate = [0.005,0.01,0.05,0.1]
lr = [0.001, 0.0001]
sbatch = 32
z_sample = 1
train_sample = 100
test_sample = 100
w_sample = 1
SIVI_layer_size = 100
SIVI_input_dim = 50

local_rep = True
prior_gmm = False #prior is gau mixture (True) or gau unit 0,1 (False)

tau.append(1)

##############################################

for l in lr:
  for dr in droprate:
    for t in tau:
      test_rs = []
      for split_id in range(n_splits):
          dat, input_dim, normalization_info= data.get(data_type, split_id) #read data

          xtrain = dat['train']['x'].cuda()
          ytrain = dat['train']['y'].cuda()

          xvalid = dat['valid']['x'].cuda()
          yvalid = dat['valid']['y'].cuda()

          xtest = dat['test']['x'].cuda()
          ytest = dat['test']['y'].cuda()
          if split_id == 0:
              print('TRAIN:' + str(xtrain.shape))
              print('VALID:' + str(xvalid.shape))
              print('TEST:' + str(xtest.shape))

          print('\n'+ '*'*200)
          #####################
          torch.set_default_tensor_type('torch.cuda.FloatTensor')

          #init model and apply approach
          model = Net(input_dim, [hid_layer,hid_layer], output_dim, prior_gmm=prior_gmm,SIVI_by_col=semi_by_col, SIVI_layer_size=SIVI_layer_size, SIVI_input_dim=SIVI_input_dim, semi_unit= semi_unit, droprate= dr, local_rep= local_rep,tau= t).cuda()  
          Appr = appr(model, optim= optim, train_sample= train_sample, test_sample= test_sample, w_sample= w_sample, nepochs= nepochs, sbatch= sbatch, lr=l, lr_min=lr_min, lr_factor=lr_factor, lr_patience=lr_patience)
          if split_id == 0:
              # model info
              print_model_report(model)
              print_optimizer_config(Appr.optimizer)

              print("Parameters:")
              cnt_param = 0
              for name, param in model.named_parameters():
                  if param.requires_grad:
                      cnt_param += 1
                      print("\t"+ name)
              print('\tTotal: '+ str(cnt_param))
              print('*'*200)
              print('Approach param: optim= {}, train_sample= {}, test_sample= {}, w_sample= {}, nepochs= {}, sbatch= {}, lr= {}'.format(optim, train_sample, test_sample, w_sample, nepochs, sbatch, l),end='\n')
              print('Model param: prior_gmm= {}, SIVI_layer_size= {}, SIVI_input_dim= {}, droprate= {}, local_rep= {}, tau= {}'.format(prior_gmm, SIVI_layer_size, SIVI_input_dim, dr, local_rep, t),end='\n')
              print("tau= {}, lr = {}, droprate = {}\n".format(t,l,dr))
          print('Load split: '+ str(split_id))
          print('Done!')
          ##################### TRAIN
          print('TRAINING')

          Appr.train(xtrain,ytrain,xvalid,yvalid,xtest, ytest, normalization_info)

          print('*'*200)
          ###################### TEST
          print('TESTING')

          test_loss, test_rmse, test_llh = Appr.eval(xtest, ytest,normalization_info)
          print("Test: loss= {:.3f}, rmse={:.3f}, llh={:.3f}".format(test_loss,test_rmse,test_llh),end= '\n')
          test_rs.append((test_rmse, test_llh))
          break

      mean_rmse, mean_llh = np.mean(test_rs,0)
      max_rmse, max_llh = np.max(test_rs,0)
      min_rmse, min_llh = np.min(test_rs,0)
      print(test_rs)
      print("*"*200)

      f = open("result/uci/" +data_type+ ".txt", "a")
      if t == tau[0]:
          f.write("*Data: {} \nTune parameters: droprate= {}, lr= {}, SIVI_layer_size= {}, SIVI_input_dim= {}, sbatch= {}, train_sample={}, test_sample= {}, w_sample= {}, local_rep={}, prior_gmm={}, n_epochs= {}, tau={}\n rs: rmse: mean={}, max={}, min= {}; llh: mean={}, max={}, min={} \n".format(data_type,dr,l,SIVI_layer_size, SIVI_input_dim, sbatch, train_sample, test_sample, w_sample, str(local_rep), str(prior_gmm),nepochs,t,mean_rmse, max_rmse, min_rmse, mean_llh, max_llh, min_llh))
      else:
          f.write("\nTune parameters: droprate= {}, lr= {}, SIVI_layer_size= {}, SIVI_input_dim= {}, sbatch= {}, train_sample={}, test_sample= {}, w_sample= {}, local_rep={}, prior_gmm={}, n_epochs= {}, tau={}\n rs: rmse: mean={}, max={}, min= {}; llh: mean={}, max={}, min={} \n".format(dr,l,SIVI_layer_size, SIVI_input_dim, sbatch, train_sample, test_sample, w_sample, str(local_rep), str(prior_gmm),nepochs,t,mean_rmse, max_rmse, min_rmse, mean_llh, max_llh, min_llh))

      f.close()
