import torch
import numpy as np
from utils import *
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from networks.SIVI_bnn_CNN import Net
from approach.sivi_bnn_cnn import Appr as appr


dataset = 'CIFAR10'
if dataset == 'CIFAR10':
    from dataloader import cifar_10 as data
elif dataset == 'CIFAR100':
    from dataloader import cifar_100 as data

##################init
tstart = time.time()
# Seed
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
else:
    print('[CUDA unavailable]'); sys.exit()

##################################################

lr_min = 1e-7
lr_factor=3
lr_patience=5
optim = 'Adam'


###################load data
print("Load data...")
print(dataset + ' DATASET')

dat, input_dim, output_dim = data.get() #read data
xtrain = dat['train']['x'].cuda()
ytrain = dat['train']['y'].cuda()

xvalid = dat['valid']['x'].cuda()
yvalid = dat['valid']['y'].cuda()

xtest = dat['test']['x'].cuda()
ytest = dat['test']['y'].cuda()

print(xvalid.shape, yvalid.shape)
print('Done!')
print('\n'+ '*'*200)


##########################Tune parameters
nepochs = 600
lr = 0.0001
droprate = None
sbatch = 256
train_sample = 10
test_sample = 10
test_w_sample = 2
SIVI_input_dim = 100
SIVI_layer_size = 400

local_rep = False
prior_gmm = False #prior is gau mixture (True) or gau unit 0,1 (False)
for lr in [0.0001, 0.0001]:
#####################init model and apply approach
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = Net(input_dim, output_dim, prior_gmm=prior_gmm, SIVI_layer_size=SIVI_layer_size, SIVI_input_dim=SIVI_input_dim, droprate= droprate, local_rep= local_rep).cuda()
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
    print("Approach parameters: \n\toptim ={} \n\tlr= {} \n\ttrain_sample= {} \n\ttest_sample= {} \n\ttest_w_sample= {} \n\tnepochs= {} \n\tsbatch= {}".format(optim, lr, train_sample, test_sample, test_w_sample, nepochs, sbatch), end='\n')
    print("Model parameters: \n\tSIVI_layer_size= {} \n\tSIVI_input_dim= {} \n\tprior_gmm= {} \n\tlocal_rep= {} \n\tdroprate= {}".format(SIVI_layer_size, SIVI_input_dim, prior_gmm, local_rep, droprate),end='\n')
    print("-"*200)
    ##################### TRAIN
    print('TRAINING')
    Appr.train(xtrain,ytrain,xvalid,yvalid,xtest, ytest)

    print('*'*200)
    ###################### TEST
    print('TESTING')
    train_loss, train_acc = Appr.eval(xtrain, ytrain)
    valid_loss, valid_acc = Appr.eval(xvalid, yvalid)
    test_loss, test_acc = Appr.eval(xtest, ytest)
    print("Test: loss= {:.3f}, acc={:.3f}".format(test_loss,100*test_acc),end= '')
    f = open("result/cifar/"+ dataset+".txt", "a")
    f.write("*Tune parameters: droprate= {}, lr= {}, SIVI_layer_size= {}, SIVI_input_dim= {}, sbatch= {}, train_sample={}, test_sample= {}, test_w_sample= {}, local_rep={}, prior_gmm={}, n_epochs= {}\n result: train_acc= {}, valid_acc= {}, test_acc= {}\n".format(droprate,lr,SIVI_layer_size, SIVI_input_dim, sbatch, train_sample, test_sample, test_w_sample, str(local_rep), str(prior_gmm),nepochs, train_acc, valid_acc, test_acc))
    f.close()