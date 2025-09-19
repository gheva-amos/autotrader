import collections

class RSI:
  def __init__(self, period):
    self.period = period
    self.factor = (self.period - 1)/self.period
    self.up = collections.deque(maxlen=self.period)
    self.down = collections.deque(maxlen=self.period)
    self.last_close = None
    self.last_up = None
    self.last_down = None
    self.rsis = {}

  def process_bar(self, bar):
    close = float(bar['close'])
    if not self.last_close:
      self.last_close = close
      return False

    if close > self.last_close:
      self.up.append(close - self.last_close)
      self.down.append(0.0)
    else:
      self.up.append(0.0)
      self.down.append(self.last_close - close)

    self.last_close = close

    ret = False
    if len(self.up) == self.period:
      up_avg = sum(self.up)/self.period
      down_avg = sum(self.down)/self.period
      if self.last_up:
        up_avg += self.factor * self.last_up
        down_avg += self.factor * self.last_down

      self.last_up = up_avg
      self.last_down = down_avg

      add = 0.0
      if down_avg == 0.0:
        add = 0.00001
      rs = up_avg / (down_avg + add)
      self.rsis[bar['date']] = 100 - 100 / (1 - rs)
      ret = True

    return ret
