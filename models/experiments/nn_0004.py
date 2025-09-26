import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset, TensorDataset, DataLoader
import torch.nn as nn
import argparse
import glob
import os
from itertools import zip_longest
import math

device = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)

ALL_COLUMNS = ['open', 'high', 'low', 'volume', 'rsi14', 'macd',
       'macd_signal', 'macd_diff', 'atr14', 'stoch_k', 'stoch_d', 'sma20',
       'sma50', 'ema20', 'ema50', 'bb_mavg', 'bb_high', 'bb_low', 'next',
       'hist', 'target', 'close']
class BinaryHead(nn.Module):
  def __init__(self, in_dim, hidden=(128, 64), drop=0.2):
    super().__init__()

    layers = []
    d = in_dim
    for h in hidden:
      layers += [nn.Linear(d, h, bias=True), nn.ReLU(), nn.Dropout(drop)]
      layers += [nn.LayerNorm(h)]
      d = h
    layers.append(nn.Linear(d, 1, bias=True))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
     return self.net(x).squeeze(1)

class MemoryNet(nn.Module):
  def __init__(self, in_dim, hid=64, layers=1, drop=0.0):
    super().__init__()
    self.memory = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True,
    dropout=(drop if layers>1 else 0.0))

  def forward(self, x):
    out, _ = self.memory(x)
    return out[:, -1, :]  

class TreMomVol(nn.Module):
  def __init__(self, features, tre_idx, mom_idx, vol_idx, hidden=128, layers=2, drop=0.0):
    super().__init__()
    self.tre_idx = torch.tensor(tre_idx)
    self.mom_idx = torch.tensor(mom_idx)
    self.vol_idx = torch.tensor(vol_idx)

    self.trend = MemoryNet(len(tre_idx), hidden, layers, drop)
    self.momentum = MemoryNet(len(mom_idx), hidden, layers, drop)
    self.volatility = MemoryNet(len(vol_idx), hidden, layers, drop)

    fused = 3 * hidden
    self.fuse = nn.Sequential(                  # keep as regression
      nn.LayerNorm(fused),
      nn.Linear(fused, 256),
      nn.ReLU(),
      nn.Dropout(drop),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(drop),
    )

    self.head_next = BinaryHead(128, hidden=(128,), drop=drop)
    self.head_hist = BinaryHead(128, hidden=(64,), drop=drop)
    self.head_target = nn.Sequential(                  # keep as regression
      nn.LayerNorm(128),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Dropout(drop),
      nn.Linear(128, 1)
    )

  def forward(self, x):
    tre_x = x.index_select(2, self.tre_idx)#to(x.device))
    mom_x = x.index_select(2, self.mom_idx)#to(x.device))
    vol_x = x.index_select(2, self.vol_idx)#to(x.device))

    tre_y = self.trend(tre_x)
    mom_y = self.momentum(mom_x)
    vol_y = self.volatility(vol_x)

    tmp_y = torch.cat([tre_y, mom_y, vol_y], dim=1)
    h = self.fuse(tmp_y)
    n_logit = self.head_next(h).squeeze(-1)
    h_logit = self.head_hist(h).squeeze(-1)
    t_logit = self.head_target(h).squeeze(1)
    return n_logit, h_logit, t_logit

class Cerberos(nn.Module):
  def __init__(self, features, hidden=128, layers=3, drop=0.5):
    super().__init__()
    self.memory = nn.LSTM(features, hidden, num_layers=layers, batch_first=True,
      dropout=(drop if layers > 1 else 0.0))
    self.head_next = BinaryHead(hidden, hidden=(256,128,64), drop=drop)
    self.head_hist = BinaryHead(hidden, hidden=(128,64,32), drop=drop)
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

def load_files(directory):
  parquets = glob.glob(os.path.join(directory, "*.parquet"))

  ret = {}
  for parq in parquets:
    symbol = os.path.splitext(os.path.basename(parq))[0]
    ret[symbol] = pd.read_parquet(parq)
    ret[symbol]["date"] = pd.to_datetime(ret[symbol]["date"])
    ret[symbol] = ret[symbol].set_index("date").sort_index()

  return preprocess(ret)

def align_features(df, cols):
  df = df.copy()
  df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
  for c in cols:
    if c not in df: df[c] = pd.NA
  df = df.loc[:, cols]
  return df.dropna(subset=cols)

def preprocess(data):
  ret = {}
  for sym, df in data.items():
    num_cols = df.select_dtypes(include="number").columns
    df_norm = df.copy()
    df_norm[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std().replace(0, 1e-8)
    df_norm.index = df.index
    df_norm['next'] = (df['close'].shift(-1) > df['close']).astype('int8')
    df_norm['hist'] = (df['close'].shift(10) > df['close']).astype('int8')
    df_norm["target"] = df["close"].shift(-1)
    df_norm = df_norm.dropna()
    ret[sym] = align_features(df_norm, ALL_COLUMNS)

  return ret

def apply_window(df, window):
  X_all = df.drop(columns=['close', 'target', 'next', 'hist']).to_numpy(dtype="float32")
  Y_next = df['next'].to_numpy(dtype='float32')
  Y_hist = df['hist'].to_numpy(dtype='float32')
  Y_target = df['target'].to_numpy(dtype='float32')
  Xs, yns, yhs, ytarg = [], [], [], []
  for i in range(window-1, len(X_all)):
    Xs.append(X_all[i-window+1:i+1])
    yns.append(Y_next[i])
    yhs.append(Y_hist[i])
    ytarg.append(Y_target[i])

  X, yN, yH, yT = [], [], [], []
  if Xs:
    X = np.stack(Xs).astype("float32")
    yN = np.array(yns, dtype="float32")
    yH = np.array(yhs, dtype="float32")
    yT = np.array(ytarg, dtype="float32")

  return (X, yN, yH, yT)

def to_data_loader(directory, window):
  data = load_files(directory)
  sym_data = []
  for sym, df in data.items():
    tmp = apply_window(df, window)
    if len(tmp[0]):
      sym_data.append(tmp)

  rr_ds = RoundRobinDataSource(sym_data)
  return DataLoader(rr_ds, batch_size=32, shuffle=False)

def calc_nl_loss(n, l):
  return 0.7 * n + 0.3 * l

def calc_loss(nl, tgt):
  return 0.9 * nl + 0.1 * tgt

def validate_epoch(val_loader, model, mse, bce_next, bce_hist):    
  model.eval()
  losses = []
  for xb, yN, yH, tgt in val_loader:
    with torch.no_grad():
      xb = xb.float()
      yN = yN.float()
      yH = yH.float()
      tgt = tgt.float()
      n_logit, h_logit, tgt_pred = model(xb)
      nl_loss = calc_nl_loss(bce_next(n_logit, yN.float()), bce_hist(h_logit, yH.float()))
      tgt_loss = mse(torch.sigmoid(tgt_pred), tgt.float())
      loss = calc_loss(nl_loss, tgt_loss)
      losses.append(loss)
  return sum(losses) / len(losses)

def scale_ys(loader):
  pos_next = 0.0
  total_next = 0.0
  pos_hist = 0.0
  total_hist = 0.0

  with torch.no_grad():
    for xb, yN, yH, tgt in loader:
      yN = yN.float()
      yH = yH.float()
      pos_next += yN.sum().item()
      total_next += yN.numel()
      pos_hist += yH.sum().item()
      total_hist += yH.numel()
  P_next = max(pos_next, 1e-8) 
  N_next = total_next - pos_next
  P_hist = max(pos_hist, 1e-8)
  N_hist = total_hist - pos_hist

  p_next = max(pos_next / max(total_next, 1e-8), 1e-6)
  p_hist = max(pos_hist / max(total_hist, 1e-8), 1e-6)
  pw_next = torch.tensor(N_next / P_next, dtype=torch.float32)
  pw_hist = torch.tensor(N_hist / P_hist, dtype=torch.float32)

  return pw_next, pw_hist, p_next, p_hist

def set_last_linear_bias(mod, value):
  last_lin = None
  for m in mod.modules():
    if isinstance(m, nn.Linear):
      last_lin = m
  if last_lin is None or last_lin.bias is None:
    raise RuntimeError("No Linear with bias in this head")
  with torch.no_grad():
    last_lin.bias.fill_(value)

def train(model, train_loader, val_loader, epochs, cp_path):
  pw_next, pw_hist, p_next, p_hist = scale_ys(train_loader)
  bc = nn.BCEWithLogitsLoss()
  bce_next = torch.nn.BCEWithLogitsLoss(pos_weight=pw_next)
  bce_hist = torch.nn.BCEWithLogitsLoss(pos_weight=pw_hist)
  bias = math.log(p_next / (1 - p_next)) 
  set_last_linear_bias(model.head_next, bias)
#model.head_next.bias.data.fill_(bias)
  mse = nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  lowest_loss = validate_epoch(val_loader, model, mse, bce_next, bce_hist)
  print(f'validation: {lowest_loss}')
  for epoch in range(epochs):
    model.train()
    for xb, yN, yH, tgt in train_loader:
      xb = xb.float()
      yN = yN.float()
      yH = yH.float()
      tgt = tgt.float()
      opt.zero_grad()
      n_logit, h_logit, tgt_pred = model(xb)
      nl_loss = calc_nl_loss(bce_next(n_logit, yN.float()), bce_hist(h_logit, yH.float()))
      tgt_loss = mse(torch.sigmoid(tgt_pred), tgt.float())
      loss = calc_loss(nl_loss, tgt_loss)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
    print("epoch", epoch, "loss", loss.item())
    val = validate_epoch(val_loader, model, mse, bce_next, bce_hist)
    print(f'validation: {val}')
    if val < lowest_loss:
      ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "epoch": epoch,
      }
      torch.save(ckpt, os.path.join(cp_path, str(epoch) + '.cp'))
      lowest_loss = val

def train_model(window, data_dir, val_dir, checkpoint_dir, epochs):

  train_loader = to_data_loader(data_dir, window)

  val_loader = to_data_loader(val_dir, window)

# Removed close as it is being removed in the data loader
  all_columns = ALL_COLUMNS
  tmp = ['open', 'high', 'low', 'volume', 'rsi14', 'macd',
         'macd_signal', 'macd_diff', 'atr14', 'stoch_k', 'stoch_d', 'sma20',
         'sma50', 'ema20', 'ema50', 'bb_mavg', 'bb_high', 'bb_low', 'next',
         'hist', 'target']
  trend_columns = ['macd', 'macd_signal', 'macd_diff', 'sma20', 'sma50', 'ema20', 'ema50']
  momentum_columns = ['rsi14', 'stoch_k', 'stoch_d']
  volatility_columns = ['atr14', 'bb_mavg', 'bb_high', 'bb_low']

  trend_indexes = [all_columns.index(c) for c in trend_columns]
  momentum_indexes = [all_columns.index(c) for c in momentum_columns]
  volatility_indexes = [all_columns.index(c) for c in volatility_columns]

  print(trend_indexes)
  print(momentum_indexes)
  print(volatility_indexes)
  xb, _, _, _ = next(iter(train_loader))

  model = TreMomVol(xb.shape[2], trend_indexes, momentum_indexes, volatility_indexes, 128, 3, 0.5)

  train(model, train_loader, val_loader, epochs, checkpoint_dir)


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

  args = parser.parse_args()

  if args.mode == 'train':
    train_model(15, args.train_dir, args.val_dir, args.checkpoints, 50)
