#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/30 

from torch import Tensor
from torch import nn

from data import NUM_CLASSES

class CNN(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(1,  8, 5, stride=2, padding=5//2)
    #self.bn1   = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=5//2)
    #self.bn2   = nn.BatchNorm2d(16)
    self.act   = nn.ReLU()
    self.drop  = nn.Dropout(0.5)
    self.fc    = nn.Linear(16 * 7 * 7, NUM_CLASSES)
  
  def forward(self, x:Tensor):
    x = x               # [B, 1, 28, 28]
    x = self.conv1(x)   # [B, 8, 14, 14]
    x = self.act(x)
    #x = self.bn1(x)
    x = self.conv2(x)   # [B, 16, 7, 7]
    x = self.act(x)
    #x = self.bn2(x)
    x = self.drop(x)
    x = x.flatten(start_dim=1)
    x = self.fc(x)
    return x
