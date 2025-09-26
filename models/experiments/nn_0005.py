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

DEVICE1 = torch.device("cpu")
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)

ALL_COLUMNS = ['open', 'high', 'low', 'volume', 'rsi14', 'macd',
       'macd_signal', 'macd_diff', 'atr14', 'stoch_k', 'stoch_d', 'sma20',
       'sma50', 'ema20', 'ema50', 'bb_mavg', 'bb_high', 'bb_low', 'next',
       'hist', 'target', 'close']

class MemoryNet(nn.Module):
  def __init__(self, in_dim, hid=64, layers=1, drop=0.0):
    super().__init__()
    self.memory = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True,
    dropout=(drop if layers>1 else 0.0))

  def forward(self, x):
    out, _ = self.memory(x)
    return out[:, -1, :]  

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

class TCNMem(nn.Module):
  def __init__(self, features, hidden=64, drop=0.2):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv1d(features, hidden, kernel_size=3, padding=1),
      nn.GELU(),
      nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
      nn.GELU(),
      nn.Dropout(drop),
    )

  def forward(self, x):
    z = x.transpose(1, 2)
    z = self.net(z)
    return z.transpose(1, 2)

class Model0003(nn.Module):
  def __init__(self, features, tre_idx, mom_idx, vol_idx, hidden=128, layers=2, drop=0.0):
    super().__init__()
    self.register_buffer('tre_idx', torch.tensor(tre_idx))
    self.register_buffer('mom_idx', torch.tensor(mom_idx))
    self.register_buffer('vol_idx', torch.tensor(vol_idx))

    self.trend = nn.Linear(len(tre_idx), hidden)
    self.momentum = nn.Linear(len(mom_idx), hidden)
    self.volatility = nn.Linear(len(vol_idx), hidden)

#self.memory = nn.LSTM(3*hidden, hidden, batch_first=True)
    self.memory = TCNMem(3*hidden, hidden, drop)

    self.shared = nn.Sequential(nn.LayerNorm(hidden),
      nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(drop))

    self.head_next = nn.Linear(128, 1)
    self.head_hist = nn.Linear(128, 1)
    self.head_tgt  = nn.Linear(128, 1)

    self.dropper = nn.Dropout(drop)
    self.alpha = nn.Parameter(torch.zeros(3))

  def forward(self, x):
    x.to(DEVICE)
    tre_x = x.index_select(2, self.tre_idx.to(DEVICE))
    mom_x = x.index_select(2, self.mom_idx.to(DEVICE))
    vol_x = x.index_select(2, self.vol_idx.to(DEVICE))
    B, W, _ = tre_x.shape
    tre_x = tre_x.reshape(B*W, -1)
    mom_x = mom_x.reshape(B*W, -1)
    vol_x = vol_x.reshape(B*W, -1)

    tre_y = self.dropper(self.trend(tre_x).reshape(B, W, -1))
    mom_y = self.dropper(self.momentum(mom_x).reshape(B, W, -1))
    vol_y = self.dropper(self.volatility(vol_x).reshape(B, W, -1))

    w = torch.softmax(self.alpha, dim=0)

    tmp_y = torch.cat([w[0] * tre_y, w[1] * mom_y, w[2] * vol_y], dim=2)

#h, _ = self.memory(tmp_y)
    h = self.memory(tmp_y)
    tmp_y = self.shared(h[:, -1, :])
    n_logit = self.head_next(tmp_y).squeeze(1)
    h_logit = self.head_hist(tmp_y).squeeze(1)
    t_logit = self.head_tgt(tmp_y).squeeze(1)
    return n_logit, h_logit, t_logit


class Model0002(nn.Module):
  def __init__(self, features, tre_idx, mom_idx, vol_idx, hidden=128, layers=2, drop=0.0):
    super().__init__()
    self.register_buffer('tre_idx', torch.tensor(tre_idx))
    self.register_buffer('mom_idx', torch.tensor(mom_idx))
    self.register_buffer('vol_idx', torch.tensor(vol_idx))
    self.hidden_count = hidden

    self.trend = MemoryNet(len(tre_idx), hidden, layers, drop)
    self.momentum = MemoryNet(len(mom_idx), hidden, layers, drop)
    self.volatility = MemoryNet(len(vol_idx), hidden, layers, drop)

    self.alpha = nn.Parameter(torch.zeros(3))

    self.tre_norm = nn.LayerNorm(hidden)
    self.mom_norm = nn.LayerNorm(hidden)
    self.vol_norm = nn.LayerNorm(hidden)

    self.norm = nn.LayerNorm(hidden)

    fused = 3 * hidden
    self.fuse = nn.Sequential(
      nn.LayerNorm(fused),
      nn.Linear(fused, 256),
      nn.ReLU(),
      nn.Dropout(drop),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(drop),
    )
    self.head_next = BinaryHead(128, hidden=(32,), drop=drop)

  def forward(self, x):
    x.to(DEVICE)
    tre_x = x.index_select(2, self.tre_idx.to(DEVICE))
    mom_x = x.index_select(2, self.mom_idx.to(DEVICE))
    vol_x = x.index_select(2, self.vol_idx.to(DEVICE))

    tre_y = self.tre_norm(self.trend(tre_x))
    mom_y = self.vol_norm(self.momentum(mom_x))
    vol_y = self.mom_norm(self.volatility(vol_x))

    w = torch.softmax(self.alpha, dim=0)

#   with torch.no_grad():
#     print("||mom||", mom_y.norm(dim=1).mean().item(),
#     "||tre||", tre_y.norm(dim=1).mean().item(),
#     "||vol||", vol_y.norm(dim=1).mean().item())
#     print(w)
#tmp_y = w[0] * tre_y + w[1] * mom_y + w[2] * vol_y

    tmp_y = torch.cat([w[0] * tre_y, w[1] * mom_y, w[2] * vol_y], dim=1)
#z = self.norm(tmp_y)
    h = self.fuse(tmp_y)
    n_logit = self.head_next(h)

    return n_logit

class Model0001(nn.Module):
  def __init__(self, features, hidden=64, layers=2, drop=0.5):
    super().__init__()
    self.memory = nn.LSTM(features, hidden, num_layers=layers, batch_first=True,
      dropout=(drop if layers > 1 else 0.0))
    self.fc = nn.Linear(hidden, 1, bias=True)

  def forward(self, x):
    out, _ = self.memory(x)
    z = out[:, -1, :]
    return self.fc(z).squeeze(1) 

class RoundRobinDataSource(IterableDataset):
  def __init__(self, per_symbol_tuples):
    '''using data returned as the second element of load_files'''
    self.streams = []
    for X, yN, yH, tgt in per_symbol_tuples:
      Nxs = len(X)
      NyN = len(yN)
      NyH = len(yH)
      Ntg = len(tgt) if hasattr(tgt, "__len__") else Nxs
      N   = min(Nxs, NyN, NyH, Ntg)
      if N == 0:
        continue
      X = X[:N]
      yN = yN[:N]
      yH = yH[:N]
      if hasattr(tgt, "__len__"):
        tgt = tgt[:N]
      else:
        tgt = np.zeros((N,), dtype="float32")

      Xt = torch.from_numpy(X).to(torch.float32).contiguous()        # [N,W,F]
      yNt = torch.from_numpy(yN).to(torch.float32).contiguous()       # [N]
      yHt = torch.from_numpy(yH).to(torch.float32).contiguous()       # [N]
      tt  = torch.from_numpy(tgt).to(torch.float32).contiguous()      # [N]

      self.streams.append((
        Xt, yNt, yHt, tt,
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
    df_norm.sort_index()
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

def split_sequence(X, yN, yH, tgt, frac=0.8):
  N = int(len(X) * frac)
  train = (X[:N], yN[:N], yH[:N], tgt[:N])
  val   = (X[N:], yN[N:], yH[N:], tgt[N:])
  return train, val

def to_data_loader_2(directory, window):
  data = load_files(directory)
  train_tuples = []
  val_tuples = []
  for sym, df in data.items():
    tmp = apply_window(df, window)
    if len(tmp[0]) == 0:
      continue
    t, v = split_sequence(tmp[0], tmp[1], tmp[2], tmp[3])
    train_tuples.append(t)
    val_tuples.append(v)

  rr_t = RoundRobinDataSource(train_tuples)
  rr_v = RoundRobinDataSource(val_tuples)
  return DataLoader(rr_t, batch_size=32, shuffle=False), DataLoader(rr_v, batch_size=32, shuffle=False)

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

def calc_loss(n, h, t):
  return 0.7 * n + 0.3 * h + 0.01 * t

@torch.no_grad()
def validate0003(val_loader, model, bce_next, bce_hist, mse):    
  model.eval()
  losses = []
  total_loss = 0.0
  total_N    = 0
  all_p = []
  all_y = []
  for xb, yN, yH, tgt in val_loader:
    xb = xb.float().to(DEVICE)
    yN = yN.float().to(DEVICE)
    yH = yH.float().to(DEVICE)
    tgt = tgt.float().to(DEVICE)
    n_logit, h_logit, t_logit = model(xb)
    loss = calc_loss(bce_next(n_logit, yN), bce_hist(h_logit, yH), mse(t_logit, tgt))
    losses.append(loss.item())

    p = torch.sigmoid(n_logit).detach().cpu().numpy()
    y = yN.detach().cpu().numpy()
    all_p.append(p)
    all_y.append(y)
  all_p = np.concatenate(all_p)
  all_y = np.concatenate(all_y)

  acc = float(((all_p > 0.3) == (all_y > 0.5)).mean())

  return sum(losses) / len(losses) , acc

def validate_epoch(val_loader, model, bce_next):    
  model.eval()
  losses = []
  with torch.no_grad():
    for xb, yN, yH, tgt in val_loader:
      xb = xb.float().to(DEVICE)
      yN = yN.float().to(DEVICE)
      yH = yH.float().to(DEVICE)
      tgt = tgt.float().to(DEVICE)
      n_logit = model(xb)
      loss = bce_next(n_logit, yN.float())
      losses.append(loss.item())
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

def train_0003(model, train_loader, val_loader, epochs, cp_path):
  pw_next, pw_hist, p_next, p_hist = scale_ys(train_loader)
  bce_next = torch.nn.BCEWithLogitsLoss(pos_weight=pw_next)
  bce_hist = torch.nn.BCEWithLogitsLoss(pos_weight=pw_hist)
  mse = nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

  for epoch in range(epochs):
    model.train()
    for xb, yN, yH, tgt in train_loader:
      xb = xb.float().to(DEVICE)
      yN = yN.float().to(DEVICE)
      yH = yH.float().to(DEVICE)
      tgt = tgt.float().to(DEVICE)
      opt.zero_grad()
      n_logit, h_logit, t_logit = model(xb)
      loss = calc_loss(bce_next(n_logit, yN), bce_hist(h_logit, yH), mse(t_logit, tgt))
      loss.backward()
      opt.step()
    print("epoch", epoch, "loss", loss.item())
    val, acc = validate0003(val_loader, model, bce_next, bce_hist, mse)
    print(f"validation: {val}, accuracy: {acc}")

def train_0001(model, train_loader, val_loader, epochs, cp_path):
  pw_next, pw_hist, p_next, p_hist = scale_ys(train_loader)
  bce_next = torch.nn.BCEWithLogitsLoss(pos_weight=pw_next)
  bce_hist = torch.nn.BCEWithLogitsLoss(pos_weight=pw_hist)
  mse = nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
  xb, yN, _, _ = next(iter(train_loader))
  for epoch in range(epochs):
    model.train()
    for xb, yN, yH, tgt in train_loader:
      xb = xb.float().to(DEVICE)
      yN = yN.float().to(DEVICE)
      opt.zero_grad()
      n_logit = model(xb)
      loss = bce_next(n_logit, yN.float())
      loss.backward()
      opt.step()
    val = validate_epoch(val_loader, model, bce_next)
    print("epoch", epoch, "loss", loss.item())
    print(f"validation: {val}")

def train_model(window, data_dir, val_dir, checkpoint_dir, epochs):

#train_loader = to_data_loader(data_dir, window)

#val_loader = to_data_loader(val_dir, window)

  train_loader, val_loader = to_data_loader_2(data_dir, window)
# Removed close as it is being removed in the data loader
  all_columns = ALL_COLUMNS
  trend_columns = ['macd', 'macd_signal', 'macd_diff', 'sma20', 'sma50', 'ema20', 'ema50']
  momentum_columns = ['rsi14', 'stoch_k', 'stoch_d']
  volatility_columns = ['atr14', 'bb_mavg', 'bb_high', 'bb_low']
  bar_columns = ['open', 'close', 'high', 'low', 'volume']

  trend_indices = [all_columns.index(c) for c in trend_columns]
  momentum_indices = [all_columns.index(c) for c in momentum_columns]
  volatility_indices = [all_columns.index(c) for c in volatility_columns]
  bar_indices = [all_columns.index(c) for c in bar_columns]

  xb, _, _, _ = next(iter(train_loader))

#model = Model0001(xb.shape[2]).to(DEVICE)
#model = Model0002(xb.shape[2], trend_indices, momentum_indices, volatility_indices, 128, 3, 0.5).to(DEVICE)
  model = Model0003(xb.shape[2], trend_indices, momentum_indices, volatility_indices, 128, 3, 0.35).to(DEVICE)
  train_0003(model, train_loader, val_loader, epochs, checkpoint_dir)


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
    train_model(15, args.train_dir, args.val_dir, args.checkpoints, 2500)
