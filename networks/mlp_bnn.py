from bayes_layer import BayesianLinear
import torch
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, inputdim, layer_size, outputdim, re_wKL = 1, prior_gmm= True, pi= 0.5, sig_gau1=1, sig_gau2=1, sample= True, ratio=0.5):
        super(Net, self).__init__()
        self.outputdim = outputdim
        self.re_wKL = re_wKL
        hid1, hid2 = layer_size
        ncha, size, _ =inputdim
        self.fc1 = BayesianLinear(size*size, hid1, prior_gmm= prior_gmm, pi= pi, sig_gau1=sig_gau1, sig_gau2=sig_gau2,ratio=ratio)
        self.fc2 = BayesianLinear(hid1, hid2, prior_gmm= prior_gmm, pi= pi, sig_gau1=sig_gau1, sig_gau2=sig_gau2, ratio=ratio)
        self.fc3 = BayesianLinear(hid2, outputdim, prior_gmm= prior_gmm, pi= pi, sig_gau1=sig_gau1, sig_gau2=sig_gau2, ratio=ratio)
        self.relu = torch.nn.ReLU()
        self.sample = sample

    def forward(self,x):
        h=x.view(x.size(0),-1)
        h=self.relu(self.fc1(h,self.sample))
        h=self.relu(self.fc2(h,self.sample))
        y = self.fc3(h,self.sample)
        return y

    def loss_forward(self, x,y, N_M, no_sample,mb_index,train= True):
        total_qw, total_pw, total_log_likelihood = 0., 0., 0.
        re_wKL = self.re_wKL
        out = torch.zeros([x.shape[0],self.outputdim])

        if train:
            no_sample = 1
            
        for i in range(no_sample):
            output = F.log_softmax(self.forward(x), dim= 1)       
            total_qw += self.get_qw()
            total_pw += self.get_pw()
            out += output
            
        total_qw = total_qw/no_sample
        total_pw = total_pw/no_sample
        total_log_likelihood = F.nll_loss(out/no_sample, y, reduction='sum')
        if re_wKL == "adaptive":
            re_wKL = np.power(2.0,N_M-mb_index) / (np.power(2.0,N_M)-1)
            #print(re_wKL)
        else:
            re_wKL = re_wKL/N_M

        loss = re_wKL*(total_qw - total_pw) + total_log_likelihood
        return loss, out/no_sample

    # def pred_sample(self, x,y, no_sample):
    #     output = torch.zeros([len(x),self.outputdim])
    #     for i in range(no_sample):
    #         output_ = F.log_softmax(self.forward(x), dim= 1)
    #         output += output_
    #     return output/no_sample

    def get_pw(self):
        return self.fc1.p_w + self.fc2.p_w + self.fc3.p_w

    def get_qw(self):
        return self.fc1.q_w + self.fc2.q_w + self.fc3.q_w

if __name__ == "__main__":
    bnn = Net(28*28, [400,400], 10, sample=False).cuda()
    t = torch.Tensor(10, 28*28).uniform_(-0.5,0.5).cuda()
    out = bnn(t)
    print(F.softmax(out, dim=1))