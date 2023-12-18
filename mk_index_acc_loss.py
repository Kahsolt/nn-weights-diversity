#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/01/17 

import os

from test_all import BATCH_SIZE, LR, TRAIN_STEPS
from train import IMG_PATH

HTML_FILE = 'index_acc_loss.html'
IMG_WIDTH  = 480
IMG_HEIGHT = 400

html = '''
<html>
<head>
  <title>nn-weights-diversity</title>
</head>
<body>
<table>
%s
</table>
</body>
</html>
'''

if __name__ == '__main__':
  mk_tr = lambda txt: f'<tr>{txt}</tr>'
  mk_td = lambda txt: f'<td>{txt}</td>'
  mk_img = lambda fp: f'<img src="{fp}" width="{IMG_WIDTH}" height="{IMG_HEIGHT}">'
  mk_card = lambda fp: mk_td(mk_img(fp))

  trs = []
  for bs in BATCH_SIZE:
    tr = []
    for lr in LR:
      fp = os.path.join(IMG_PATH, f'mnist_S={TRAIN_STEPS}_B={bs}_lr={lr}.png')
      tr.append(mk_card(fp))
    trs.append(mk_tr('\n'.join(tr)))
  trs = '\n'.join(trs)

  html = html % trs
  with open(HTML_FILE, 'w', encoding='utf-8') as fh:
    fh.write(html)
