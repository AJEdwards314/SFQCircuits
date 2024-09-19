import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import itertools
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import functional as F, init
import math
torch.autograd.set_detect_anomaly(True)

testData = torch.tensor([[[1,1]],[[1,0]],[[0,1]],[[0,0]]]).to(torch.float)
testResults = torch.tensor([[0],[1],[1],[0]]).to(torch.float)

inFeatures = 2
outFeatures = 1


#TODO fan in
class Adder(nn.Module):
    def __init__(self,numOut=2):
        super(Adder,self).__init__()
        self.numOut = numOut
    def forward(self,X):
        output = X.size()[0]*F.avg_pool1d(X,self.numOut)
        return output

#TODO fan out
class Propo(nn.Module):
    def __init__(self):
        super(Propo,self).__init__()
    def forward(self,X,outFeatures):
        output = X[0].clone()
        for i in range(outFeatures-1):
            output = torch.cat((output,X[0].clone()))
        return output

#TODO graphChecker
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.syn1 = ModifiedLinear(inFeatures,inFeatures, maximum=1, minimum=-1)
        self.syn1 = nn.Linear(inFeatures, inFeatures)
        self.neuro1 = nn.Sigmoid()
        self.fanIn = Adder()
        #self.fanOut = nn.ReplicationPad1d((1,0))
        self.fanOut = Propo()
        self.syn2 = nn.Linear(inFeatures, inFeatures)
        #self.syn2 = ModifiedLinear(inFeatures,inFeatures, maximum=1, minimum=-1)
        self.neuro2 = nn.Sigmoid()
        self.syn3 = nn.Linear(inFeatures, outFeatures)
        #self.syn3 = ModifiedLinear(inFeatures,outFeatures, maximum=1, minimum=-1)
        self.neuro3 = nn.Sigmoid()
    def forward(self,X):
        inter = self.syn1(X)
        inter = self.neuro1(inter)
        inter = self.fanIn(inter)
        inter = self.fanOut(inter,2)
        inter = self.syn2(inter)
        inter = self.neuro2(inter)
        inter = self.syn3(inter)
        inter = self.neuro3(inter)
        return inter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

numEpochs = 10000
net = Net().to(device)
loss_hist = []
loss = nn.MSELoss()
optimizer = torch.optim.NAdam(net.parameters(), lr=1e-2)
for epoch in range(numEpochs):
    if epoch%100 == 0:
        print("Epoch {}".format(epoch))
    for i in range(len(testData)):
        data = testData[i].to(device)
        results = testResults[i].to(device)
        optimizer.zero_grad()
        net.train()
        iterResults = net(data)
        loss_val = loss(iterResults, results)
        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.detach().numpy())
        if epoch%100 == 0:
            print("loss {}: {}".format(i,loss_hist[-1]))
            print("Expected: {}----Result: {}".format(results, iterResults.data))
        for i in net.parameters():
            i.data.clamp_(1,-1)
            if epoch%1000 == 0:
                print(i.data)
plt.plot(loss_hist)
plt.savefig("loss_values.png")
