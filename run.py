#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/01/17 

import os

from test_all import BATCH_SIZE, LR, TRAIN_STEPS

for bs in BATCH_SIZE:
  for lr in LR:
    exp_name = f'mnist_S={TRAIN_STEPS}_B={bs}_lr={lr}'
    model_fp = os.path.join('log', f'{exp_name}.model')
    if os.path.exists(model_fp): continue

    cmd = f'python train.py -B {bs} --lr {lr}'
    print(cmd)
    os.system(cmd)
