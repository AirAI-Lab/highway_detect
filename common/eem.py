import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, laplace

class SEM(nn.Module):
    def __init__(self, ch_out, reduction=None):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)

def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n,sigma)
    gaussiankernel = torch.from_numpy(g)
    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    return laplacekernel.float()

class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, groups, reduction):
        super(EEM, self).__init__()
        self.groups = groups
        self.gk = gaussiankernel(ch_in, int(ch_in/groups), kernel, kernel-2, 0.9)
        self.lk = laplaceiankernel(ch_in, int(ch_in/groups), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(int(ch_out/2), ch_out, kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out)
        )
        self.sem1 = SEM(ch_out, reduction=reduction)
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)
    def forward(self, x):
        DoG = F.conv2d(x, self.gk.to(x.device), padding='same',groups=self.groups)
        LoG = F.conv2d(DoG, self.lk.to(x.device), padding='same',groups=self.groups)
        DoG = self.conv1(DoG-x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG*LoG)
        tot1 = self.sem1(tot)
        x1 = self.sem2(x)
        return self.prelu(x+x1+tot+tot1)
