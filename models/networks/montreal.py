import torch
import torch.nn as nn

class Montreal(nn.Module):
  def __init__(self, mom_idx, tre_idx, vol_idx, smothed_idx):
    super().__init__()
    self.register_buffer("mom_idx", torch.tensor(mom_idx, dtype=torch.long))
    self.register_buffer("tre_idx", torch.tensor(tre_idx, dtype=torch.long))
    self.register_buffer("vol_idx", torch.tensor(vol_idx, dtype=torch.long))
    self.register_buffer("smothed_idx", torch.tensor(smothed_idx, dtype=torch.long))

  def split_data(self, x):
    m = x.index_select(2, self.mom_idx)
    t = x.index_select(2, self.tre_idx)
    v = x.index_select(2, self.vol_idx) 
    s = x.index_select(2, self.smothed_idx) 

    return m, t, v, s

