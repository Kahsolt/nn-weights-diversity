#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import get_mnist
from model import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.enable_grad()
def pgd(X:Tensor, Y:Tensor, model:nn.Module, steps=40, eps=8/255, alpha=0.01) -> Tensor:
  X_orig = X.detach().clone()
  X_adv = X_orig + torch.empty_like(X).uniform_(-eps, eps)

  for _ in range(steps):
    X_adv.requires_grad = True
    
    output = model(X_adv)
    loss = F.cross_entropy(output, Y, reduction='none')
    grad = torch.autograd.grad(loss, X_adv, grad_outputs=loss)[0]

    X_adv = X_adv + grad.sign() * alpha
    delta = torch.clamp(X_adv - X_orig, -eps, eps)
    X_adv = (X_orig + delta).detach()

  return X_adv


def get_clean_acc(model, dataloader) -> float:
  total, ok = 0, 0
  for X, Y in tqdm(dataloader):
    X, Y = X.to(device), Y.to(device)
    pred_X = model(X).argmax(axis=-1)
    total += len(X)
    ok  += (pred_X == Y).sum().item()
  return ok / total

def get_clean_racc(model, dataloader, steps=40, eps=8/255, alpha=0.01) -> float:
  total, ok = 0, 0
  for X, Y in tqdm(dataloader):
    X, Y = X.to(device), Y.to(device)
    AX = pgd(X, Y, model, steps, eps, alpha)
    pred_AX = model(AX).argmax(axis=-1)
    total += len(X)
    ok  += (pred_AX == Y).sum().item()
  return ok / total


@torch.no_grad()
def test(args):
  print('>> [Test]')

  ''' Model '''
  model = CNN().to(device)
  model.eval()
  
  ''' Ckpt '''
  print(f'>> [Load] from {args.ckpt}')
  state_dict = torch.load(args.ckpt)
  model.load_state_dict(state_dict)

  ''' Data '''
  dataloader = get_mnist(is_train=False, batch_size=args.batch_size)

  ''' Test '''
  acc = get_clean_acc(model, dataloader)
  print(f"clean accuracy: {acc:.3%}")

  ''' Attack '''
  racc = get_clean_racc(model, dataloader, args.steps, args.eps, args.alpha)
  print(f"remnent accuracy: {racc:.3%}")


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('ckpt', help='path to *.pth file')
  parser.add_argument('-B', '--batch_size', default=32, type=int)
  parser.add_argument('--steps', default=40,   type=int)
  parser.add_argument('--eps',   default=0.03, type=float)
  parser.add_argument('--alpha', default=0.01, type=float)
  args = parser.parse_args()

  test(args)
