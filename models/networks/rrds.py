from itertools import zip_longest
from torch.utils.data import IterableDataset
import torch

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


