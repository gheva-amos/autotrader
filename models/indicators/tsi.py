from ema import EMA

class TSI:
  def __init__(self, slow, fast):
    self.last_close = None
    self.num_base = EMA(slow)
    self.denum_base = EMA(slow)
    self.num = EMA(fast)
    self.denum = EMA(fast)
    self.tsis = {}

  def process_bar(self, bar):
    close = float(bar['close'])
    if not self.last_close:
      self.last_close = close
      return False

    ret = False
    m = close - self.last_close
    abs_m = abs(m)
    self.last_close = close
    self.num_base.add_value(m)
    self.denum_base.add_value(abs_m)

    if self.num_base.full():
      self.num.add_value(self.num_base.avg())
      self.denum.add_value(self.denum_base.avg())

    if self.num.full():
      ret = True
      avg = self.denum.avg()
      date = bar['date']
      if avg == 0:
        self.tsis[date] = 100 * self.num.avg()/(self.denum.avg() + 0.00001)
      else:
        self.tsis[date] = 100 * self.num.avg()/self.denum.avg()

    return ret
