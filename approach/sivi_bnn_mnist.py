import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn

from networks.SIVI_bnn_MNIST import Net
from bayes_layer import SIVI_bayes_layer

class Appr(object):
  
    def __init__(self,model,optim = 'Adam', train_sample=10, test_sample = 10, w_sample= 10, nepochs=100, sbatch=256, lr=0.001,lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100):
        self.model=model
        self.model_old=model
        self.valid_rs = []


        #file_name = log_name
        #self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')   
        self.train_sample = train_sample
        self.test_sample = test_sample
        self.w_sample = w_sample
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

    def train(self, xtrain, ytrain, xvalid, yvalid, xtest, ytest):
        best_loss = np.inf
        best_acc = -np.inf
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
            # train_loss,train_acc=self.eval(xtrain,ytrain)
            # clock2=time.time()
            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
            #     e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
            #     1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(xvalid,yvalid, test= False)
            clock2=time.time()
            print('Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms, Valid: loss={:.3f}, acc={:5.1f}% |'.format(e+1,1000*self.sbatch*(clock1-clock0)/num_batch,1000*self.sbatch*(clock2-clock1)/xvalid.size(0),valid_loss,100*valid_acc),end='')
            # test_loss,test_acc=self.eval(xtest,ytest)
            # print(' Test: acc={:5.1f}% |'.format(100*test_acc),end='')
            self.valid_rs.append((valid_loss,valid_acc))
            # Adapt lr
            # if valid_acc > best_acc:
            #     best_model = utils.get_model(self.model)
            #     best_acc = valid_acc
                
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
      
            if len(r) % mini_batch_size == 0:
                number_of_batchs = len(r) // mini_batch_size
            else:
                number_of_batchs = len(r) // mini_batch_size + 1

            loss = self.model.loss_forward(images, targets, number_of_batchs, self.train_sample, i+1)

            self.optimizer.zero_grad()
            loss.backward()
            if self.optim == 'SGD' or self.optim == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,x,y, test= False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        with torch.no_grad():
            for i in range(0, len(r), self.sbatch):
                if i + self.sbatch <= len(r):
                    b = r[i:i + self.sbatch]
                else:
                    b = r[i:]
                images = x[b]
                targets = y[b]

                # Forward
                mini_batch_size = len(targets)
                if len(r) % mini_batch_size == 0:
                    number_of_batchs = len(r) // mini_batch_size
                else:
                    number_of_batchs = len(r) // mini_batch_size + 1
                    
                loss = self.model.loss_forward(images, targets, number_of_batchs, self.train_sample, i+1)
                output = self.model.pred_sample(images, targets, self.test_sample, self.w_sample, test = test)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                total_loss += loss.data.cpu().numpy()
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(b)
  
        return total_loss/total_num,total_acc/total_num





