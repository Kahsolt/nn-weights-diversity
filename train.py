#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD

from data import get_mnist
from model import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = 'log'
IMG_PATH = 'img'


def train(args):
  print('>> [Train]')

  ''' Model '''
  model = CNN().to(device)
  optimizer = SGD(model.parameters(), lr=args.lr)

  ''' Data '''
  dataloader = get_mnist(is_train=True, batch_size=args.batch_size)
  
  ''' Train '''
  epoch = 0
  step = 0
  loss_list, acc_list = [], []
  model.train()
  while True:
    if step > args.steps: break
    epoch += 1

    total, ok = 0, 0
    for X, Y in dataloader:
      if step > args.steps: break
      step  += 1

      X = X.to(device)
      Y = Y.to(device)
      
      output = model(X)
      loss = F.cross_entropy(output, Y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total += len(X)
      ok    += (Y == output.argmax(axis=-1)).sum().item()

      if step % 100 == 0:
        loss_list.append(loss.item())
        acc_list.append(ok / total)
        print(f"[Epoch {epoch+1} / Step {step}], loss: {loss_list[-1]:.7f}, accuracy: {acc_list[-1]:.3%}")

  ''' Save '''
  model_fp = os.path.join(LOG_PATH, f'{args.exp_name}.model')
  os.makedirs(os.path.dirname(model_fp), exist_ok=True)
  torch.save(model.state_dict(), model_fp)

  stat_fp  = os.path.join(IMG_PATH, f'{args.exp_name}.png')
  os.makedirs(os.path.dirname(stat_fp), exist_ok=True)
  fig, ax1 = plt.subplots()
  ax1.plot(acc_list, c='r', alpha=0.75, label='accuracy')
  ax2 = ax1.twinx()
  ax2.plot(loss_list, c='b', alpha=0.75, label='loss')
  fig.legend()
  fig.suptitle(args.exp_name)
  fig.savefig(stat_fp, dpi=400)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-S', '--steps',      default=10000, type=int)
  parser.add_argument('-B', '--batch_size', default=32,    type=int)
  parser.add_argument('--lr',               default=0.05,  type=float)
  args = parser.parse_args()

  args.exp_name = f'mnist_S={args.steps}_B={args.batch_size}_lr={args.lr}'

  model_fp = os.path.join(LOG_PATH, f'{args.exp_name}.model')
  if os.path.exists(model_fp):
    print(f'>> ignore {args.exp_name} due to file exists')
    exit(0)

  train(args)
