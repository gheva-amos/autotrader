import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset, TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import glob
import os
import math
from models.networks.nn0001 import NN0001
from models.networks.rrds import RoundRobinDataSource


DEVICE1 = torch.device("cpu")
DEVICE = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)

ALL_COLUMNS = ['open', 'high', 'low', 'volume', 'rsi14', 'macd',
       'macd_signal', 'macd_diff', 'atr14', 'stoch_k', 'stoch_d', 'sma20',
       'sma50', 'ema20', 'ema50', 'bb_mavg', 'bb_high', 'bb_low', 'next',
       'hist', 'target', 'close',
       'open_ema3', 'close_ema3', 'high_ema3',
       'low_ema3', 'macd_diff_ema3', 'stoch_k_ema3', 'stoch_d_ema3',
       'open_m3', 'close_m3', 'high_m3',
       'low_m3', 'macd_diff_m3', 'stoch_k_m3', 'stoch_d_m3',
       'open_med3', 'close_med3', 'high_med3',
       'low_med3', 'macd_diff_med3', 'stoch_k_med3', 'stoch_d_med3']

meta_lr = 1e-3
meta_drop = 0.2
meta_decay = 1e-4
meta_thr = 0.7
meta_window = 15
meta_epochs = 200
meta_batch_size = 32

#{{{ Loading data

def preprocess(data):
  ret = {}
  for sym, df in data.items():
    ret[sym] = df

  return ret

def load_file(file):
  ret = {}
  symbol = os.path.splitext(os.path.basename(file))[0]
  ret[symbol] = pd.read_parquet(file)
# ret[symbol]["date"] = pd.to_datetime(ret[symbol]["date"])
# ret[symbol] = ret[symbol].set_index("date").sort_index()

  return preprocess(ret)

def load_files(directory):
  parquets = glob.glob(os.path.join(directory, "*.parquet"))

  ret = {}
  for parq in parquets:
    symbol = os.path.splitext(os.path.basename(parq))[0]
    ret[symbol] = pd.read_parquet(parq)
#ret[symbol]["date"] = pd.to_datetime(ret[symbol]["date"])
#    ret[symbol] = ret[symbol].set_index("date").sort_index()

  return preprocess(ret)

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

def load_for_plotting(file, window):
  data = load_file(file)
  tuples = []
  for sym, df in data.items():
    tmp = apply_window(df, window)
    tuples.append(tmp)
  rr = RoundRobinDataSource(tuples)
  return DataLoader(rr, batch_size=len(tuples), shuffle=False)

def to_data_loaders(directory, window):
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
  return DataLoader(rr_t, batch_size=meta_batch_size, shuffle=False), DataLoader(rr_v, batch_size=len(val_tuples), shuffle=False)

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
  N_next = max(total_next - pos_next, 0)
  P_hist = max(pos_hist, 1e-8)
  N_hist = max(total_hist - pos_hist, 0)

  p_next = max(pos_next / max(total_next, 1e-8), 1e-6)
  p_hist = max(pos_hist / max(total_hist, 1e-8), 1e-6)
  pw_next = torch.tensor(N_next / P_next, dtype=torch.float32, device=DEVICE)
  pw_hist = torch.tensor(N_hist / P_hist, dtype=torch.float32, device=DEVICE)

  return pw_next, pw_hist, p_next, p_hist

#}}}

def calc_loss(n, h, t):
  return 0.4 * n + 0.6 * h + 0.01 * t

def column_indices():
  all_columns = ALL_COLUMNS
  trend_columns = ['macd', 'macd_signal', 'macd_diff', 'sma20', 'sma50', 'ema20', 'ema50']
  momentum_columns = ['rsi14', 'stoch_k', 'stoch_d']
  volatility_columns = ['atr14', 'bb_mavg', 'bb_high', 'bb_low']
  bar_columns = ['open', 'close', 'high', 'low', 'volume']
  smothed_columns = ['open_ema3', 'close_ema3', 'high_ema3', 'low_ema3', 'macd_diff_ema3', 'stoch_k_ema3', 'stoch_d_ema3']
  smothed_columns += ['open_med3', 'close_med3', 'high_med3', 'low_med3', 'macd_diff_med3', 'stoch_k_med3', 'stoch_d_med3']
  smothed_columns += ['open_m3', 'close_m3', 'high_m3', 'low_m3', 'macd_diff_m3', 'stoch_k_m3', 'stoch_d_m3']

  trend_indices = [all_columns.index(c) for c in trend_columns]
  momentum_indices = [all_columns.index(c) for c in momentum_columns]
  volatility_indices = [all_columns.index(c) for c in volatility_columns]
  bar_indices = [all_columns.index(c) for c in bar_columns]
  smothed_indices = [all_columns.index(c) for c in smothed_columns]

  return trend_indices, momentum_indices, volatility_indices, bar_indices, smothed_indices
#{{{ Common network classes

#}}}

#{{{ Models
#}}}

#{{{ 001

@torch.no_grad()
def validate001(model, val_loader, bce_next, bce_hist, mse, thr=0.5):
  model.eval()
  tot_loss, tot_N = 0.0, 0
  Pn, Yn = [], []
  Ph, Yh = [], []
  all_acc_next, all_acc_hist = [], []
  for xb, yN, yH, tgt in val_loader:
    xb = xb.float().to(DEVICE)
    yN = yN.float().to(DEVICE)
    yH = yH.float().to(DEVICE)
    tgt = tgt.float().to(DEVICE)
    n_logit, h_logit, t_logit = model(xb)

    loss = calc_loss(bce_next(n_logit, yN), bce_hist(h_logit, yH), mse(t_logit, tgt))

    p_next = torch.sigmoid(n_logit)
    p_hist = torch.sigmoid(h_logit)

    pred_next = (p_next > 0.7).int().cpu()
    pred_hist = (p_hist > 0.7).int().cpu()

    all_acc_next.append((pred_next == yN.cpu().int()).float())
    all_acc_hist.append((pred_hist == yH.cpu().int()).float())

    bs = yN.size(0)
    tot_loss += loss.item() * bs
    tot_N += bs

  accn = torch.cat(all_acc_next).mean().item()
  acch = torch.cat(all_acc_hist).mean().item()
  acc = (accn + acch) / 2

  return float(tot_loss / max(tot_N,1)), acc

def do_train001(model, train_loader, val_loader, checkpoint_dir, epochs):
  pw_next, pw_hist, p_next, p_hist = scale_ys(train_loader)
  bce_next = torch.nn.BCEWithLogitsLoss(pos_weight=pw_next)
  bce_hist = torch.nn.BCEWithLogitsLoss(pos_weight=pw_hist)
  mse = nn.MSELoss()
  opt = torch.optim.Adam(model.parameters(), lr=meta_lr, weight_decay=meta_decay)

  best_acc = 0.0
  wait = 10
  best_loss = 100.0
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
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()

    print("epoch", epoch, "loss", loss.item())
    tmp = loss.item()
    loss, acc = validate001(model, val_loader, bce_next, bce_hist, mse, meta_thr)
    if acc >= best_acc and tmp < best_loss:
      best_loss = tmp
      best_acc = acc
      ckpt = {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict()
      }
      torch.save(ckpt, os.path.join(checkpoint_dir, f"{epoch}.cp"))
      wait = 30
    elif wait == 0:
      wait = 30
      best_acc = 0.0
      best_loss = tmp
    else:
      wait -= 1

    print(f"loss: {loss}, acc: {acc}")

def train001(window, data_dir, checkpoint_dir, epochs):
  train_loader, val_loader = to_data_loaders(data_dir, window)
  trend_indices, momentum_indices, volatility_indices, bar_indices, smothed_indices = column_indices()

  xb, _, _, _ = next(iter(train_loader))
  model = NN0001(xb.shape[2], trend_indices, momentum_indices, volatility_indices, smothed_indices, 32, meta_drop).to(DEVICE)

  do_train001(model, train_loader, val_loader, checkpoint_dir, epochs)

@torch.no_grad()
def plot001(window, file, checkpoint):
  loader = load_for_plotting(file, window)
  trend_indices, momentum_indices, volatility_indices, bar_indices, smothed_indices = column_indices()

  xb, _, _, _ = next(iter(loader))
  model = NN0001(xb.shape[2], trend_indices, momentum_indices, volatility_indices, smothed_indices, 32, meta_drop).to(DEVICE)
  ckpt = torch.load(checkpoint)
  model.load_state_dict(ckpt["model_state"])

  model.eval()
  p_next, p_hist, targets, idxs = [], [], [], []
  idx = 0
  for xb, yN, yH, tgt in loader:
    xb = xb.float().to(DEVICE)
    n_logit, h_logit, t_logit = model(xb)
    for tmp in torch.sigmoid(n_logit):
      p_next.append(tmp.item())
    for tmp in torch.sigmoid(h_logit):
      p_hist.append(tmp.item())
    for tmp in tgt:
      targets.append(tmp.item())
      idxs.append(idx)
      idx += 1

  pred_next = pd.Series(p_next, index=idxs, name="p_next_up")
  pred_hist = pd.Series(p_hist, index=idxs, name="p_hist_up")
  target = pd.Series(targets, index=idxs, name="target")

  ax = target.plot(figsize=(11,4), lw=1.2, label="Close")
  ax2 = ax.twinx()
  pred_next.plot(ax=ax2, color="orange", lw=1.0, alpha=0.7, label="P(next)")
  pred_hist.plot(ax=ax2, color="green", lw=1.0, alpha=0.7, label="P(hist)")

  ax.set_ylabel("Close")
  ax2.set_ylabel("Probabilities")
  ax.legend(loc="upper left")
  ax2.legend(loc="upper right")
  plt.tight_layout()
  plt.show()

@torch.no_grad()
def eval001(window, file, checkpoint):
  loader = load_for_plotting(file, window)
  trend_indices, momentum_indices, volatility_indices, bar_indices, smothed_indices = column_indices()

  pw_next, pw_hist, p_next, p_hist = scale_ys(loader)
  bce_next = torch.nn.BCEWithLogitsLoss(pos_weight=pw_next)
  bce_hist = torch.nn.BCEWithLogitsLoss(pos_weight=pw_hist)
  mse = nn.MSELoss()

  xb, _, _, _ = next(iter(loader))
  model = NN0001(xb.shape[2], trend_indices, momentum_indices, volatility_indices, smothed_indices, 32, meta_drop).to(DEVICE)
  ckpt = torch.load(checkpoint)
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  for xb, yN, yH, tgt in loader:
    xb = xb.float().to(DEVICE)
    yN = yN.float().to(DEVICE)
    yH = yH.float().to(DEVICE)
    tgt = tgt.float().to(DEVICE)
    n_logit, h_logit, t_logit = model(xb)
    loss = calc_loss(bce_next(n_logit, yN), bce_hist(h_logit, yH), mse(t_logit, tgt))
    print(loss.item())
  p_next = torch.sigmoid(n_logit)
  p_hist = torch.sigmoid(h_logit)
  print(p_next.item(), p_hist.item())

def main001(args):
  if args.mode == 'train':
    train001(meta_window, args.train_dir, args.checkpoints, meta_epochs)
  elif args.mode == 'plot':
    plot001(meta_window, args.parquet, args.checkpoint)
  elif args.mode == 'eval':
    eval001(meta_window, args.parquet, args.checkpoint)


#}}}

if __name__ == '__main__':
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument('--window', default=15)
  common.add_argument('--lr', default=0.1)
  common.add_argument('--decay', default=1e-4)
  common.add_argument('--drop', default=0.2)
  common.add_argument('--threshold', default=0.5)
  common.add_argument('--epochs', default=200)


  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="mode", required=True)

  train_parser = subparsers.add_parser("train", parents=[common])
  train_parser.add_argument('train_dir', default='', help='the directory with input files to process')
  train_parser.add_argument('checkpoints', default='', help='the directory to hols checkpoints')

  plot_parser = subparsers.add_parser("plot", parents=[common])
  plot_parser.add_argument('parquet', default='', help='the file with the data to load and evaluate')
  plot_parser.add_argument('checkpoint', default='', help='the checkpoint to use for plotting')

  eval_parser = subparsers.add_parser("eval", parents=[common])
  eval_parser.add_argument('parquet', default='', help='the file with the data to load and evaluate')
  eval_parser.add_argument('checkpoint', default='', help='the checkpoint to use for plotting')

  args = parser.parse_args()

  meta_lr = float(args.lr)
  meta_drop = float(args.drop)
  meta_decay = float(args.decay)
  meta_thr = float(args.threshold)
  meta_window = int(args.window)
  meta_epochs = int(args.epochs)

  main001(args)

# vim: set foldmethod=marker foldmarker={{{,}}} :
