#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

DATA_PATH = 'data'
NUM_CLASSES = 10


def get_mnist(is_train=True, batch_size=32, shuffle=False):
  dataset    = MNIST(root=DATA_PATH, train=is_train, transform=T.ToTensor(), download=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle or is_train, drop_last=True, pin_memory=True, num_workers=0)
  print(f'n_batches: {len(dataloader)}, n_samples: {len(dataloader.dataset)}')
  return dataloader


def imshow(X: Tensor, title='', fp=None):
  grid_X = make_grid(X).permute([1, 2, 0]).detach().cpu().numpy()
  plt.clf()
  plt.axis('off')
  plt.imshow(grid_X)
  plt.suptitle(title)
  plt.tight_layout()
  if fp: plt.savefig(fp, dpi=400)
  else:  plt.show()


if __name__ == '__main__':
  dataloader = get_mnist(is_train=False)
  g = iter(dataloader)

  X, Y = next(g)
  print(X.shape)    # [B=32, C=1, H=28, W=28]
  print(Y.shape)    # [B=32]
  print(Y)

  imshow(X)
