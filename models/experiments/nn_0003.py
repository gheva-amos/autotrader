import pandas as pd
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import IterableDataset, TensorDataset, DataLoader
import torch.nn as nn
from itertools import zip_longest
import glob
import os
import traceback

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
    ret[dt]['ma26'] = data['ma26'][dt]
    ret[dt]['ma10'] = data['ma10'][dt]
    ret[dt]['macd_vals'] = data['macd'][0][dt]
    ret[dt]['macd_signal'] = data['macd'][1][dt]
    ret[dt]['tsi'] = data['tsi'][dt]
    ret[dt]['rsi'] = data['rsi'][dt]
    ret[dt]['obv'] = data['obv'][dt]
    ret[dt]['close'] = float(data['labels'][dt])

  return ret

def load_json(path, window):
  with open(path, 'r') as f:
    data = json.load(f)

  pp = preprocess(data)

  df = pd.DataFrame.from_dict(pp, orient='index').sort_index()

  df['next'] = (df['close'].shift(-1) > df['close']).astype('int8')
  df['hist'] = (df['close'].shift(10) > df['close']).astype('int8')
  df["target"] = df["close"].shift(-1)
  df = df.dropna()

  df_norm = (df - df.mean()) / df.std()

  X_all = df_norm.drop(columns=['close', 'target', 'next', 'hist']).to_numpy(dtype="float32")
  Y_next = df['next'].to_numpy(dtype='float32')
  Y_hist = df['hist'].to_numpy(dtype='float32')
  Y_target = df_norm['target'].to_numpy(dtype='float32')

  return (apply_window(X_all, Y_next, Y_hist, Y_target, window), df)

def apply_window(X_all, Y_next, Y_hist, Y_target, window):
  Xs, yns, yhs, ytarg = [], [], [], []
  for i in range(window-1, len(X_all)):
    Xs.append(X_all[i-window+1:i+1])
    yns.append(Y_next[i])
    yhs.append(Y_hist[i])
    ytarg.append(Y_target[i])

  X = np.stack(Xs).astype("float32")
  yN = np.array(yns, dtype="float32")
  yH = np.array(yhs, dtype="float32")
  yT = np.array(ytarg, dtype="float32")

  return (X, yN, yH, yT)

def load_files(pt, window):
  json_files = glob.glob(os.path.join(pt, "*.json"))
  ret = []
  symbs = []
  for p in json_files:
    try:
      loaded = load_json(p, window)[0]
      ret.append(loaded)
      symbs.append(os.path.splitext(os.path.basename(p))[0])
    except Exception as e:
      print(f"Error loading {p}:", e)
      traceback.print_exc()
  return symbs, ret

class RoundRobinDataSource(IterableDataset):
  def __init__(self, per_symbol_tuples):
    '''using data returned as the second element of load_files'''
    self.streams = []
    for X, yN, yH, tgt in per_symbol_tuples:
      self.streams.append((
        torch.from_numpy(X),
        torch.from_numpy(yN),
        torch.from_numpy(yH),
        torch.from_numpy(tgt),
      ))

  def __iter__(self):
    iters = [zip(X, yN, yH, tgt) for (X, yN, yH, tgt) in self.streams]
    for tup in zip_longest(*iters, fillvalue=None):
      for item in tup:
        if item is None:
          continue
        xb, yN_b, yH_b, tgt_b = item
        yield xb, yN_b, yH_b, tgt_b

class BinaryHead(nn.Module):
  def __init__(self, in_dim, hidden=(128, 64), drop=0.2):
    super().__init__()

    layers = []
    d = in_dim
    for h in hidden:
      layers += [nn.LayerNorm(d)]
      layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
      d = h
    layers.append(nn.Linear(d, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
     return self.net(x).squeeze(1)

class Cerberos(nn.Module):
  def __init__(self, features, hidden=128, layers=2, drop=0.2):
    super().__init__()
    self.memory = nn.LSTM(features, hidden, num_layers=layers, batch_first=True,
      dropout=(drop if layers > 1 else 0.0))
    self.head_next = BinaryHead(hidden, hidden=(128,128), drop=drop)
    self.head_hist = BinaryHead(hidden, hidden=(128,64), drop=drop)
    self.head_target = nn.Sequential(                  # keep as regression
      nn.LayerNorm(hidden),
      nn.Linear(hidden, 128),
      nn.ReLU(),
      nn.Dropout(drop),
      nn.Linear(128, 1)
    )

  def forward(self, x):
    out, _ = self.memory(x)
    h = out[:, -1, :]
    n_logit = self.head_next(h)
    h_logit = self.head_hist(h)
    t_logit = self.head_target(h).squeeze(1)
    return n_logit, h_logit, t_logit

def calc_nl_loss(n, l):
  return 0.7 * n + 0.3 * l

def calc_loss(nl, tgt):
  return 0.9 * nl + 0.1 * tgt

def train(model, train_loader, val_loader, epochs, cp_path):
  bce = nn.BCEWithLogitsLoss()
  mse = nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  lowest_loss = validate_epoch(val_loader, model, mse, bce)
  print(f'validation: {lowest_loss}')
  for epoch in range(epochs):
    model.train()
    for xb, yN, yH, tgt in train_loader:
      n_logit, h_logit, tgt_pred = model(xb)
      nl_loss = calc_nl_loss(bce(n_logit, yN.float()), bce(h_logit, yH.float()))
      tgt_loss = mse(tgt_pred, tgt.float())
      loss = calc_loss(nl_loss, tgt_loss)
      opt.zero_grad()
      loss.backward()
      opt.step()
    print("epoch", epoch, "loss", loss.item())
    val = validate_epoch(val_loader, model, mse, bce)
    print(f'validation: {val}')
    if val < lowest_loss:
      ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": epoch,
      }
      torch.save(ckpt, os.path.join(cp_path, str(epoch) + '.cp'))
      lowest_loss = val
    
def validate_epoch(val_loader, model, mse, bce):    
  model.eval()
  losses = []
  for xb, yN, yH, tgt in val_loader:
    with torch.no_grad():
      n_logit, h_logit, tgt_pred = model(xb)
      nl_loss = calc_nl_loss(bce(n_logit, yN.float()), bce(h_logit, yH.float()))
      tgt_loss = mse(tgt_pred, tgt.float())
      loss = calc_loss(nl_loss, tgt_loss)
      losses.append(loss)
  return sum(losses) / len(losses)

def train_model(window, data_dir, val_dir, checkpoint_dir, epochs):

  symbols, data = load_files(data_dir, window)

  rr_ds = RoundRobinDataSource(data)
  train_loader = DataLoader(rr_ds, batch_size=32, shuffle=False)

  xb, _, _, _ = next(iter(train_loader))

  model = Cerberos(xb.shape[2], 128, 3)

  symbols, data = load_files(data_dir, window)

  rr_ds = RoundRobinDataSource(data)
  val_loader = DataLoader(rr_ds, batch_size=32, shuffle=False)

  train(model, train_loader, val_loader, epochs, checkpoint_dir)

def validate(window, data_dir, checkpoint):
  symbols, data = load_files(data_dir, window)
  rr_ds = RoundRobinDataSource(data)
  loader = DataLoader(rr_ds, batch_size=32, shuffle=False)
  xb, _, _, _ = next(iter(loader))
  model = Cerberos(xb.shape[2], 128, 3)
  ckpt = torch.load(checkpoint)
  model.load_state_dict(ckpt["model_state"])
  bce = nn.BCEWithLogitsLoss()
  mse = nn.MSELoss()
  model.eval()
  for xb, yN, yH, tgt in loader:
    with torch.no_grad():
      n_logit, h_logit, tgt_pred = model(xb)
      nl_loss = calc_nl_loss(bce(n_logit, yN.float()), bce(h_logit, yH.float()))
      tgt_loss = mse(tgt_pred, tgt.float())
      loss = calc_loss(nl_loss, tgt_loss)
    print(f'loss was: {loss.item()}')

def estimate(window, file, checkpoint):

  X, yN, yH, yT = load_json(file, window)[0]

  x_last = torch.from_numpy(X[-1:])

  model = Cerberos(x_last.shape[2], 128, 3)
  ckpt = torch.load(checkpoint)
  model.load_state_dict(ckpt["model_state"])
  model.eval()
  with torch.no_grad():
    n_logit, h_logit, tgt_pred = model(x_last)
    p_next = torch.sigmoid(n_logit).item()
    p_hist = torch.sigmoid(h_logit).item()
    p_tgt = torch.sigmoid(tgt_pred).item()
  print({"p_next_up": p_next, "p_hist_up": p_hist, "target_pred": p_tgt})

def plot(window, file, checkpoint):
  loaded = load_json(file, window)
  X, yN, yH, yT = loaded[0]
  x_last = torch.from_numpy(X[-1:])
  X = torch.from_numpy(X),
  yN = torch.from_numpy(yN),
  yH = torch.from_numpy(yH),
  tgt = torch.from_numpy(yT),
  df = loaded[1]
  dates  = df.index[window-1:].to_numpy()
  prices = df["close"].iloc[window-1:].to_numpy()
  model = Cerberos(x_last.shape[2], 128, 3)
  ckpt = torch.load(checkpoint)
  model.load_state_dict(ckpt["model_state"])
  model.eval()
  with torch.no_grad():
    n_logit, h_logit, tgt_pred = model(X[0])
    p_next = torch.sigmoid(n_logit).numpy()
    p_hist = torch.sigmoid(h_logit).numpy()
  fig, ax1 = plt.subplots(figsize=(12,6))
  ax1.plot(dates, prices, label="Close", color="black")
  ax1.set_ylabel("Price")
  ax2 = ax1.twinx()
  ax2.plot(dates, p_next, label="p_next_up", alpha=0.7)
  ax2.plot(dates, p_hist, label="p_hist_up", alpha=0.7)
  ax2.set_ylabel("Probability (0–1)")
  ax2.set_ylim(0, 1)

  ax1.legend(loc="upper left")
  ax2.legend(loc="upper right")
  ax1.set_title("Next-day vs 10-day Up Probabilities")
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="mode", required=True)


  train_parser = subparsers.add_parser("train")
  train_parser.add_argument('train_dir', default='', help='the directory with input files to process')
  train_parser.add_argument('val_dir', default='', help='the directory with validation files to process')
  train_parser.add_argument('checkpoints', default='', help='the directory to hols checkpoints')

  eval_parser = subparsers.add_parser("validate")
  eval_parser.add_argument('data_dir', default='', help='the directory with input files to process')
  eval_parser.add_argument('checkpoint', default='', help='the checkpoint file to use')

  estimate_parser = subparsers.add_parser("estimate")
  estimate_parser.add_argument('json', default='', help='the input file to process')
  estimate_parser.add_argument('checkpoint', default='', help='the checkpoint file to use')

  plot_parser = subparsers.add_parser("plot")
  plot_parser.add_argument('json', default='', help='the input file to process')
  plot_parser.add_argument('checkpoint', default='', help='the checkpoint file to use')

  args = parser.parse_args()
  if args.mode == 'train':
    train_model(32, args.train_dir, args.val_dir, args.checkpoints, 50)
  elif args.mode == 'validate':
    validate(32, args.data_dir, args.checkpoint)
  elif args.mode == 'estimate':
    estimate(32, args.json, args.checkpoint)
  elif args.mode == 'plot':
    plot(32, args.json, args.checkpoint)
