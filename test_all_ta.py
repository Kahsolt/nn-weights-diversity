#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torchattacks import AutoAttack, PGD, PGDL2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import get_mnist, NUM_CLASSES
from model import CNN
from test import device
from test_all import TRAIN_STEPS, BATCH_SIZE, LR
from train import LOG_PATH


def get_clean_racc_torchattack(model, dataloader, atk, limit=10, ) -> float:
  total, ok = 0, 0
  for i, (X, Y) in tqdm(enumerate(dataloader)):
    if i > limit: break
    X, Y = X.to(device), Y.to(device)
    AX = atk(X, Y)
    pred_AX = model(AX).argmax(axis=-1)
    total += len(X)
    ok  += (pred_AX == Y).sum().item()
  return ok / total


def tester(args):
  ''' Data '''
  dataloader = get_mnist(is_train=False, batch_size=args.batch_size, shuffle=True)

  ''' Model '''
  model = CNN().to(device)
  model.eval()


  if args.method == 'aa':
    atk = AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=NUM_CLASSES)
  elif args.method == 'pgd':
    atk = PGD(model, eps=8/255, alpha=2/255, steps=40)
  elif args.method == 'pgdl2':
    atk = PGDL2(model, eps=1, alpha=2/255, steps=40)
  else: raise ValueError()
  
  ''' Grid Test '''
  shape = (len(BATCH_SIZE), len(LR))
  racc_m = np.zeros(shape)
  for i, bs in enumerate(BATCH_SIZE):
    for j, lr in enumerate(LR):
      fp = os.path.join(LOG_PATH, f'mnist_S={TRAIN_STEPS}_B={bs}_lr={lr}.model')
      print(f'>> [Load] from {fp}')
      model.load_state_dict(torch.load(fp))
      
      racc_m[i, j] = get_clean_racc_torchattack(model, dataloader, atk, limit=args.limit)
      print(f'racc={racc_m[i, j]:.3%}')

  kwargs = dict(vmin=0.0, vmax=1.0, square=True, annot=True, xticklabels=LR, yticklabels=BATCH_SIZE, cbar=False)
  plt.clf() ; sns.heatmap(racc_m, **kwargs) ; plt.title('racc') ; plt.savefig(os.path.join(LOG_PATH, f'racc_{args.method}.png'), dpi=600)
  np.save(os.path.join(LOG_PATH, f'racc_{args.method}.npy'), racc_m)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-B', '--batch_size', default=32, type=int)
  parser.add_argument('-M', '--method', default='pgd', choices=['pgd', 'pgdl2', 'aa'])
  parser.add_argument('-L', '--limit', default=10, type=int)
  args = parser.parse_args()

  tester(args)
