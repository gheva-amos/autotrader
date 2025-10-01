import torch
import torch.nn as nn
from models.networks.binary_head import BinaryHead
from models.networks.reg_head import RegHead
from models.networks.montreal import Montreal
from models.networks.simple_seq import SimpleSequential
from models.networks.memory_lstm import MemoryLSTM
from models.networks.memory_tcn import MemoryTCN
from models.networks.binary_head_memory import BinaryHeadMemory

class NN0001(Montreal):
  def __init__(self, features, mom_idx, tre_idx, vol_idx, smothed_idx, hidden=16, drop=0.2):
    super().__init__(mom_idx, tre_idx, vol_idx, smothed_idx)
    self.body = SimpleSequential(4 * hidden, hidden)

    self.memory = MemoryTCN(features, hidden)
    self.ptsd = MemoryLSTM(hidden, hidden)

    self.encode_mom = SimpleSequential(len(mom_idx), hidden, drop)
    self.encode_tre = SimpleSequential(len(tre_idx), hidden, drop)
    self.encode_vol = SimpleSequential(len(vol_idx), hidden, drop)
    self.encode_smothed = SimpleSequential(len(smothed_idx), hidden, drop)

    self.head_next = BinaryHeadMemory(hidden, (64, 32, 16), drop)
    self.head_hist = BinaryHeadMemory(hidden, (64, 32, 16), drop)
    self.head_tgt  = BinaryHead(hidden, (64, 32, 16), drop)

  def forward(self, x):
    x = self.memory(x)

    m, t, v, s = self.split_data(x)

    m1 = m[:, -1, :]
    t1 = t[:, -1, :]
    v1 = v[:, -1, :]
    s1 = s[:, -1, :]

    m2 = m.mean(dim=1)
    t2 = t.mean(dim=1)
    v2 = v.mean(dim=1)
    s2 = s.mean(dim=1)

    m = m1 + 0.01 * m2
    t = t1 + 0.01 * t2
    v = v1 + 0.01 * v2
    s = s1 + 0.01 * s2

    m = self.encode_mom(m)
    t = self.encode_tre(t)
    v = self.encode_vol(v)
    s = self.encode_smothed(s)

    z = torch.cat([m, t, v, s], dim=1)
    h = self.body(z)
    h = self.ptsd(h)
    
    n = self.head_next(h).squeeze(1)
    r = self.head_hist(h).squeeze(1)
    t = self.head_tgt(h).squeeze(1)
    return n, r, t

