#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import os
from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_mnist
from model import CNN
from test import get_clean_acc, get_clean_racc, device
from train import LOG_PATH

TRAIN_STEPS = 10000
BATCH_SIZE = [
  1,
  2,
  4,
  8,
  16,
  32,
  64,
  128,
  256,
]
LR = [
  0.1,
  0.05,
  0.01,
  0.005,
  0.001,
  0.0005,
  0.0001,
  0.00005,
  0.00001,
]


def tester(args):
  ''' Data '''
  dataloader = get_mnist(is_train=False, batch_size=args.batch_size)

  ''' Model '''
  model = CNN().to(device)
  model.eval()
  
  ''' Grid Test '''
  shape = (len(BATCH_SIZE), len(LR))
  acc_m  = np.zeros(shape)
  racc_m = np.zeros(shape)
  for i, bs in enumerate(BATCH_SIZE):
    for j, lr in enumerate(LR):
      fp = os.path.join(LOG_PATH, f'mnist_S={TRAIN_STEPS}_B={bs}_lr={lr}.model')
      print(f'>> [Load] from {fp}')
      model.load_state_dict(torch.load(fp))
      
      acc_m [i, j] = get_clean_acc (model, dataloader)
      racc_m[i, j] = get_clean_racc(model, dataloader, args.steps, args.eps, args.alpha)

      print(f'acc={acc_m[i, j]:.3%} racc={racc_m[i, j]:.3%}')

  kwargs = dict(vmin=0.0, vmax=1.0, square=True, annot=True, xticklabels=LR, yticklabels=BATCH_SIZE, cbar=False)
  plt.clf() ; sns.heatmap(acc_m,  **kwargs)         ; plt.title('acc')  ; plt.savefig(os.path.join(LOG_PATH, 'acc.png'),  dpi=600)
  plt.clf() ; sns.heatmap(racc_m, **kwargs)         ; plt.title('racc') ; plt.savefig(os.path.join(LOG_PATH, 'racc.png'), dpi=600)
  plt.clf() ; sns.heatmap(acc_m - racc_m, **kwargs) ; plt.title('diff') ; plt.savefig(os.path.join(LOG_PATH, 'diff.png'), dpi=600)

  np.save(os.path.join(LOG_PATH, 'acc.npy'),  acc_m)
  np.save(os.path.join(LOG_PATH, 'racc.npy'), racc_m)
  np.save(os.path.join(LOG_PATH, 'diff.npy'), acc_m - racc_m)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-B', '--batch_size', default=32, type=int)
  parser.add_argument('--steps', default=40,   type=int)
  parser.add_argument('--eps',   default=0.03, type=float)
  parser.add_argument('--alpha', default=0.01, type=float)
  args = parser.parse_args()

  tester(args)
