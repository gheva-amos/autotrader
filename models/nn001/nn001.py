import pandas as pd
import torch
from models.networks.nn0001 import NN0001
from models.networks.nn01_apply_window import apply_window
from models.networks.rrds import RoundRobinDataSource
from torch.utils.data import DataLoader
from models.networks.nn01_column_indices import column_indices

DEVICE = torch.device(
  "cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu"
)

FEATURES = 39
HIDDEN = 32
class NN01Model:
  def __init__(self, checkpoint, window):
    self.window = window

    trend_indices, momentum_indices, volatility_indices, bar_indices, smothed_indices = column_indices()
    self.model = NN0001(FEATURES, trend_indices, momentum_indices, volatility_indices, smothed_indices, HIDDEN).to(DEVICE)
    ckpt = torch.load(checkpoint)
    self.model.load_state_dict(ckpt["model_state"])
    self.model.eval()

  def load_data(self, df):
    tuples = []
    tmp = apply_window(df, self.window)
    tuples.append(tmp)
    rr = RoundRobinDataSource(tuples)
    return DataLoader(rr, batch_size=len(tuples), shuffle=False)

  @torch.no_grad()
  def eval(self, df):
    loader = self.load_data(df)
    p_next, p_hist, targets, idxs = [], [], [], []
    idx = 0
    for xb, yN, yH, tgt in loader:
      xb = xb.float().to(DEVICE)
      yN = yN.float().to(DEVICE)
      yH = yH.float().to(DEVICE)
      tgt = tgt.float().to(DEVICE)
      n_logit, h_logit, t_logit = self.model(xb)
      for tmp in torch.sigmoid(n_logit):
        p_next.append(tmp.item())
      for tmp in torch.sigmoid(h_logit):
        p_hist.append(tmp.item())
      for tmp in tgt:
        targets.append(tmp.item())
        idxs.append(idx)
        idx += 1
    ret = pd.DataFrame(
      {
        'id': idxs,
        'next': p_next,
        'hist': p_hist,
        'targets': targets
      }
    )
    ret = ret.set_index('id')
    ret = ret.sort_index()
    return ret

