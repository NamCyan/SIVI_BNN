import torch
import numpy as np
from utils import *
import torch.nn.functional as F
import time
import csv
import matplotlib.pyplot as plt
from dataloader import mnist as data
from approach.sivi_bnn_mnist import Appr as appr


cnn = False
if cnn:
    from networks.SIVI_bnn_CNN_MNIST import Net
else:
    from networks.SIVI_bnn_MNIST import Net

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


lr_min = 1e-7
lr_factor=3
lr_patience=5
optim = 'Adam'


semi_unit = False
semi_by_col = False #if true: semi followed by cols else followed by rows

###################load data
print("Load data...")
print('MNIST DATASET')

dat, input_dim, output_dim = data.get() #read data
xtrain = dat['train']['x'].cuda()
ytrain = dat['train']['y'].cuda()

xvalid = dat['valid']['x'].cuda()
yvalid = dat['valid']['y'].cuda()

xtest = dat['test']['x'].cuda()
ytest = dat['test']['y'].cuda()

print("Train: "+ str(xtrain.shape))
print("Valid: "+ str(xvalid.shape))
print("Test: "+ str(xtest.shape))
print('Done!')
print('\n'+ '*'*200)

hid_layer = 400

##########################Tune parameters
nepochs = 600
lr = 0.0001
droprate = None
sbatch = 256
train_sample = 10
test_sample = 50
test_w_sample = 1
SIVI_input_dim = 100
SIVI_layer_size = 400

local_rep = False
prior_gmm = False #prior is gau mixture (True) or gau unit 0,1 (False)

for lr in [0.0001]:
#####################init model and apply approach
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if cnn:
        model = Net(input_dim, output_dim, prior_gmm=prior_gmm, SIVI_layer_size=SIVI_layer_size, SIVI_input_dim=SIVI_input_dim, droprate= droprate, local_rep= local_rep).cuda()
    else:
        model = Net(input_dim, [hid_layer,hid_layer], output_dim, prior_gmm=prior_gmm,SIVI_by_col=semi_by_col, SIVI_layer_size=SIVI_layer_size, SIVI_input_dim=SIVI_input_dim, semi_unit= semi_unit, droprate= droprate, local_rep= local_rep).cuda()

    Appr = appr(model, optim= optim, train_sample= train_sample, test_sample= test_sample, w_sample= test_w_sample, nepochs= nepochs, sbatch= sbatch, lr=lr, lr_min=lr_min, lr_factor=lr_factor, lr_patience=lr_patience)

    #report model info
    print_model_report(model)
    print_optimizer_config(Appr.optimizer)

    print("Parameters:")
    cnt_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            cnt_param += 1
            print("\t"+ name)
    print('\tTotal: '+ str(cnt_param))
    print('-'*200)
    print("Approach parameters: \n\toptim= {} \n\tlr= {} \n\ttrain_sample= {} \n\ttest_sample= {} \n\ttest_w_sample= {} \n\tnepochs= {} \n\tsbatch= {}".format(optim, lr, train_sample, test_sample, test_w_sample, nepochs, sbatch), end='\n')
    print("Model parameters: \n\thid_layer= {} \n\tSIVI_layer_size= {} \n\tSIVI_input_dim= {} \n\tprior_gmm= {} \n\tlocal_rep= {} \n\tdroprate= {}".format(hid_layer, SIVI_layer_size, SIVI_input_dim, prior_gmm, local_rep, droprate),end='\n')
    print("-"*200)
    ##################### TRAIN
    print('TRAINING')
    Appr.train(xtrain,ytrain,xvalid,yvalid,xtest, ytest)

    print('*'*200)
    ###################### TEST
    print('TESTING')
    train_loss, train_acc = Appr.eval(xtrain, ytrain, test= True)
    valid_loss, valid_acc = Appr.eval(xvalid, yvalid, test= True)
    test_loss, test_acc = Appr.eval(xtest, ytest, test= True)
    print("Test: loss= {:.3f}, acc={:.3f}".format(test_loss,100*test_acc),end= '\n')

    if cnn:
        f = open("result/mnist/mnist_cnn.txt", "a")
        f.write("*Tune parameters: droprate= {}, lr= {}, SIVI_layer_size= {}, SIVI_input_dim= {}, sbatch= {}, train_sample={}, test_sample= {}, test_w_sample= {}, local_rep={}, prior_gmm={}, n_epochs= {}\n result: rain_acc= {}, valid_acc= {},test_acc= {}\n".format(droprate,lr,SIVI_layer_size, SIVI_input_dim, sbatch, train_sample, test_sample, test_w_sample, str(local_rep), str(prior_gmm),nepochs,train_acc,valid_acc,test_acc))
        f.close()
        with open('result/mnist/mnist_cnn.csv', mode='a') as rs_file:
            rs = csv.writer(rs_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rs.writerow([str(droprate),str(lr),str(SIVI_layer_size),str(SIVI_input_dim),str(train_sample),str(test_sample),str(test_w_sample),str(local_rep),str(prior_gmm),str(nepochs),str(sbatch),str(train_acc),str(valid_acc),str(test_acc)])
    else:
        f = open("result/mnist/mnist.txt", "a")
        f.write("*Tune parameters: droprate= {}, lr= {}, SIVI_layer_size= {}, SIVI_input_dim= {}, sbatch= {}, train_sample={}, test_sample= {}, test_w_sample= {}, local_rep={}, prior_gmm={}, n_epochs= {}\n result: rain_acc= {}, valid_acc= {},test_acc= {}\n".format(droprate,lr,SIVI_layer_size, SIVI_input_dim, sbatch, train_sample, test_sample, test_w_sample, str(local_rep), str(prior_gmm),nepochs,train_acc,valid_acc,test_acc))
        f.close()
        with open('result/mnist/mnist.csv', mode='a') as rs_file:
            rs = csv.writer(rs_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rs.writerow([str(droprate),str(lr),str(SIVI_layer_size),str(SIVI_input_dim),str(train_sample),str(test_sample),str(test_w_sample),str(local_rep),str(prior_gmm),str(nepochs),str(sbatch),str(train_acc),str(valid_acc),str(test_acc)])
