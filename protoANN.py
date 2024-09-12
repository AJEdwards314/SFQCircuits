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


testData = torch.tensor([[1,1],[1,0],[0,1],[0,0]]).to(torch.float)
testResults = torch.tensor([0,1,1,0]).to(torch.float)

inFeatures = 2
outFeatures = 1

#stolen code from torch.nn.........add a limiter to forward
#https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L50
class ModifiedLinear(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        minimum = None,
        maximum = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if minimum:
            self.min = minimum
        if maximum:
            self.max = maximum
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        self.weight=nn.Parameter(self.weight/self.weight.clamp(min=self.min, max=self.max))
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    

#TODO fan in
class Adder(nn.Module):
    def __init__(self):
        super(Adder,self).__init__()
    def forward(self,X):
        output = X[0]
        for i in range(1,len(X)):
            output+=X[i]
        return output.unsqueeze(dim=0)

#TODO fan out
class Propo(nn.Module):
    def __init__(self):
        super(Propo,self).__init__()
    def forward(self,X,outFeatures):
        output = X.clone()
        for i in range(outFeatures-1):
            output = torch.cat((output,X))
        return output

#TODO graphChecker
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.syn1 = ModifiedLinear(inFeatures,inFeatures, maximum=1, minimum=-1)
        #self.neuro1 = nn.GraphCheck()
        self.fanIn = Adder()
        self.fanOut = Propo()
        self.syn2 = ModifiedLinear(inFeatures,inFeatures, maximum=1, minimum=-1)
        #self.neuro2 = nn.GraphCheck()
        self.syn3 = ModifiedLinear(inFeatures,outFeatures, maximum=1, minimum=-1)
        #self.neuro3 = nn.GraphCheck()
    def forward(self,X):
        inter = self.syn1(X)
        #inter = self.neuro1(inter)
        inter = self.fanIn(inter)
        inter = self.fanOut(inter,2)
        inter = self.syn2(inter)
        #inter = self.neuro2(inter)
        inter = self.syn3(inter)
        #inter = self.neuro3(inter)
        return inter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

numEpochs = 1000
net = Net().to(device)
loss = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
for epoch in range(numEpochs):
    for i in range(len(testData)):
        data = testData[i].to(device)
        results = testResults[i].to(device)
        net.train()
        iterResults = net(data)
        loss_val = loss(iterResults, results)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        print(iterResults)
