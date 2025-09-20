import pandas as pd
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

def preprocess(data):
# for each date since the start_date, 
  start_date = data['start_date']
  for sdi,d in enumerate(data['dates']):
    if d == start_date:
      break

  ret = {}
  for i in range(sdi, len(data['dates'])):
    dt = data['dates'][i]
    ret[dt] = {}
    ret[dt]['average'] = data['average'][dt]
    ret[dt]['macd_vals'] = data['macd'][0][dt]
    ret[dt]['macd_signal'] = data['macd'][1][dt]
    ret[dt]['tsi'] = data['tsi'][dt]
    ret[dt]['rsi'] = data['rsi'][dt]
    ret[dt]['obv'] = data['obv'][dt]
    ret[dt]['close'] = float(data['labels'][dt])

  return ret

class LSTM2Head(nn.Module):
  def __init__(self, F, H=128):
    super().__init__()
    self.lstm = nn.LSTM(F, H, batch_first=True)
    self.head_next = nn.Linear(H, 1)   # logits
    self.head_hist = nn.Linear(H, 1)   # logits

  def forward(self, x):                  # x: [B,W,F]
    out, _ = self.lstm(x)
    h = out[:, -1, :]                  # [B,H]
    return self.head_next(h).squeeze(1), self.head_hist(h).squeeze(1)

def main(window):
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', nargs=1, default='', help='the input file to process')
  parser.add_argument('--checkpoint', nargs=1, default='', help='the checkpoint file to load')
  parser.add_argument('--save', nargs=1, default='', help='the path to save the new checkpoint file')
  args = parser.parse_args()

  if not args.input:
    print('please provide an input file')
    exit(2)

  with open(args.input[0], 'r') as f:
    data = json.load(f)

  pp = preprocess(data)
  df = pd.DataFrame.from_dict(pp, orient='index').sort_index()
  df['next'] = (df['close'].shift(-1) > df['close']).astype('int8')
  df['hist'] = (df['close'].shift(10) > df['close']).astype('int8')
  df["target"] = df["close"].shift(-1)
  df = df.dropna()
  print (df)

  mu = df.mean().to_dict()
  sd = df.std().to_dict()

  ckpt = {}
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint[0])
    mu = pd.Series(ckpt["mu"], name="mu").to_dict()
    sd = pd.Series(ckpt["sd"], name="sd").to_dict()

  print(mu)

  df_norm = (df - mu) / sd

  X_all = df_norm.drop(columns=['close', 'target', 'average', 'next', 'hist']).to_numpy(dtype="float32")
  print(X_all)
  Y_next = df['next'].to_numpy(dtype='float32')
  Y_hist = df['hist'].to_numpy(dtype='float32')

  Xs, yns, yhs = [], [], []
  for i in range(window-1, len(X_all)):
    Xs.append(X_all[i-window+1:i+1])
    yns.append(Y_next[i])
    yhs.append(Y_hist[i])

  X = np.stack(Xs).astype("float32")
  yN = np.array(yns, dtype="float32")
  yH = np.array(yhs, dtype="float32")

  train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(yN), torch.from_numpy(yH))
  train_dl = DataLoader(train_ds, batch_size=32, shuffle=False)

  model = LSTM2Head(X.shape[2])
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  bce = nn.BCEWithLogitsLoss()

  if ckpt:
    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["optimizer_state"])

  if args.save:
    for epoch in range(500):
      model.train()
      for xb, yN_b, yH_b in train_dl:
        n_logit, h_logit = model(xb)
        loss = bce(n_logit, yN_b) + bce(h_logit, yH_b)
        opt.zero_grad();
        loss.backward();
        opt.step()
      print("epoch", epoch, "loss", loss.item())

    ckpt = {
      "model_state": model.state_dict(),
      "optimizer_state": opt.state_dict(),
      'mu': mu,
      'sd': sd,
    }
    torch.save(ckpt, args.save[0])
  else:
    model.eval()
    for xb, yN_b, yH_b in train_dl:
      n_logit, h_logit = model(xb)
      loss = bce(n_logit, yN_b) + bce(h_logit, yH_b)
      print("loss", loss.item())

if __name__ == '__main__':
  main(32)
