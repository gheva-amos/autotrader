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

def plot(df):
  plt.figure(figsize=(12,6))
  for col in ['average', 'macd_vals', 'macd_signal', 'tsi', 'rsi', 'obv', 'target']:
    plt.plot(df.index, df[col], label=col)

  plt.legend()
  plt.show()

def load_for_torch(X, Y, batch):
  ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
  return DataLoader(ds, batch_size=batch, shuffle=False)

class LSTMHead(nn.Module):
  def __init__(self, F, H=64, L=2, bidir=False, drop=0.2):
    super().__init__()
    self.lstm = nn.LSTM(F, H, num_layers=L, batch_first=True,
        dropout=(drop if L > 1 else 0.0), bidirectional=bidir)

    D = 2 if bidir else 1
    self.head = nn.Sequential(
      nn.LayerNorm(D*H),
      nn.Linear(D*H, 128), nn.ReLU(),
      nn.Dropout(drop),
      nn.Linear(128, 64), nn.ReLU(),
      nn.Dropout(drop),
      nn.Linear(64, 1)
    )

  def forward(self, x):
    out, _ = self.lstm(x)
    h_last = out[:, -1, :]
    return self.head(h_last).squeeze(1)

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
  df["target"] = df["close"].shift(-1)
  df = df.dropna()

  df_norm = (df - df.mean()) / df.std()
  X_all = df_norm.drop(columns=['close', 'target', 'average']).to_numpy(dtype="float32")
  Y_all = df_norm['target'].to_numpy(dtype="float32")

  Xs, Ys = [], []
  for i in range(window - 1, len(X_all)):
    Xs.append(X_all[i - window + 1:i + 1]) 
    Ys.append(Y_all[i])

  X = np.stack(Xs, axis=0).astype("float32")
  Y = np.array(Ys)

  k = int(len(X)*0.8)
  X_tr, Y_tr = X[:k], Y[:k]
  X_val, Y_val = X[k:], Y[k:]
  train_dl = load_for_torch(X_tr, Y_tr, 32)
  val_dl = load_for_torch(X_val, Y_val, 64)

  ckpt = {}
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint[0])

  F = X.shape[2]
  model = LSTMHead(F, 64, 3, True)

  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.MSELoss()

  if ckpt:
    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["optimizer_state"])

  if args.save:
    for epoch in range(500):
      model.train()
      for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad();
        loss.backward();
        opt.step()
      print("epoch", epoch, "loss", loss.item())

  for xb, yb in val_dl:
    model.eval()
    pred = model(xb)
    loss = loss_fn(pred, yb)
    print("loss", loss.item())

  ckpt = {
    "model_state": model.state_dict(),
    "optimizer_state": opt.state_dict(),
  }
  if args.save:
    torch.save(ckpt, args.save[0])

  print(data.keys())

if __name__ == '__main__':
  main(32)
