import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn

from networks.SIVI_bnn_UCI import Net
from bayes_layer import SIVI_bayes_layer

class Appr(object):
  
    def __init__(self,model,optim = 'Adam', z_sample = 10, nosample=10, test_sample= 10, nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100, args=None, log_name=None, split=False):
        self.model=model
        self.model_old=model


        #file_name = log_name
        #self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')      
        self.z_sample = z_sample
        self.nosample = nosample
        self.test_sample = test_sample
        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.optim = optim
        self.optimizer=self._get_optimizer()


        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        if self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, xtrain, ytrain, xvalid, yvalid, xtest, ytest,normalization_info):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            

            num_batch = xtrain.size(0)
            
            self.train_epoch(xtrain,ytrain)
            
            clock1=time.time()
            train_loss,train_rmse, train_llh=self.eval(xtrain,ytrain,normalization_info)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, rmse={:.3f}, llh = {:.3f} |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,train_rmse, train_llh),end='')
            # Valid
            valid_loss,valid_rmse, valid_llh=self.eval(xvalid,yvalid,normalization_info)
            print(' Valid: loss={:.3f}, rmse={:.3f}, llh={:.3f} |'.format(valid_loss,valid_rmse,valid_llh),end='')
            test_loss,test_rmse,test_llh=self.eval(xtest,ytest,normalization_info)
            print(' Test: loss= {:.3f}, rmse={:.3f}, llh={:.3f} |'.format(test_loss,test_rmse,test_llh),end='')
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        lr = self.lr_min
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # self.logger.save()
        return

    def train_epoch(self,x,y):
        self.model.train()


        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            # print("Batch: "+str(i))
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = x[b]
            targets = y[b]
            
            # Forward current model
            mini_batch_size = len(targets)
            N_M = len(r) / mini_batch_size

            loss = self.model.loss_forward(images, targets, N_M, no_sample= self.nosample, z_sample = self.z_sample)
            

            self.optimizer.zero_grad()
            loss.backward()
            if self.optim == 'SGD' or self.optim == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,x,y,normalization_info):
        total_loss=0
        total_rmse=0
        total_num=0

        tau = self.model.tau
        output_mean = normalization_info['output mean'].cuda()
        output_std = normalization_info['output std'].cuda()

        batchs =self.sbatch
        batchs = 256
        
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        logexpsum_pred = torch.empty((0,))
        rmse = torch.empty((0,))
        # Loop batches
        with torch.no_grad():
            for i in range(0, len(r), batchs):
                if i + batchs <= len(r):
                    b = r[i:i + batchs]
                else:
                    b = r[i:]
                images = x[b]
                targets = y[b]

                # Forward
                mini_batch_size = len(targets)
                N_M = len(r) / mini_batch_size

                loss = self.model.loss_forward(images, targets, N_M, self.test_sample, self.z_sample)
                rmse_part, llh_part = self.model.pred_sample(images, targets, self.test_sample, normalization_info)
                
                logexpsum_pred = torch.cat([logexpsum_pred,llh_part])
                rmse = torch.cat([rmse,rmse_part])

                total_loss += loss.data.cpu().numpy()
                total_num += len(b)

            log_likelihood = (logexpsum_pred - np.log(self.test_sample) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)).mean().data.cpu().numpy()
            total_rmse = torch.sqrt(rmse.mean()).data.cpu().numpy()

        return total_loss/total_num,total_rmse, log_likelihood





