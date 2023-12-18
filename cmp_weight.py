#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/01/19 

import os
from argparse import ArgumentParser
from typing import Dict, List

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from test_all import TRAIN_STEPS, BATCH_SIZE, LR
from train import LOG_PATH

KNOWN_KEYS = [
  'conv1.weight', 
  'conv1.bias', 
  'conv2.weight', 
  'conv2.bias', 
  'fc.weight', 
  'fc.bias',
]

def get_dist(A: Dict[str, Tensor], B: Dict[str, Tensor]) -> List[float]:
  diffs: List[float] = []
  for k in KNOWN_KEYS:
    d = (A[k] - B[k]).abs().mean().item()   # L1 dist
    diffs.append(d)
  return diffs


def cmp_weights(args):
  fps = [
    os.path.join(LOG_PATH, f'mnist_S={TRAIN_STEPS}_B={bs}_lr={lr}.model')
      for bs in BATCH_SIZE
        for lr in LR
  ]
  ticklabels = [
    f'{bs}_{lr}'
      for bs in BATCH_SIZE
        for lr in LR
  ]
  n_exp = len(fps)
  print(f'>> {n_exp}')
  distmat: List[List[None]] = [[None for _ in range(n_exp)] for _ in range(n_exp)]

  for i in range(len(fps)-1):
    ckpt1 = torch.load(fps[i])
    for j in range(i+1, len(fps)):
      ckpt2 = torch.load(fps[j])

      d: List[float] = get_dist(ckpt1, ckpt2)
      distmat[i][j] = d
      distmat[j][i] = d

  for i in range(len(fps)):
    distmat[i][i] = [0.0 for _ in range(len(KNOWN_KEYS))]
  distmat: np.ndarray = np.asarray(distmat, dtype=np.float32)     # [N=81, N=81, D=6]

  kwargs = dict(vmin=distmat.min(), vmax=distmat.max(), square=True, cbar=True)
  for i in range(len(KNOWN_KEYS)):
    plt.clf()
    sns.heatmap(distmat[:, :, i], **kwargs)
    name = KNOWN_KEYS[i]
    plt.title(name)
    plt.savefig(os.path.join(LOG_PATH, f'weights_distmat-{name}.png'), dpi=600)

  np.save(os.path.join(LOG_PATH, 'weights_distmat.npy'), distmat)


if __name__ == '__main__':
  parser = ArgumentParser()
  args = parser.parse_args()

  cmp_weights(args)
