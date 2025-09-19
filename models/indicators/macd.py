from ema import EMA

class MACD:
  def __init__(self, slow, fast, macd):
    self.slow = EMA(slow)
    self.fast = EMA(fast)
    self.macd = EMA(macd)
    self.values = {}
    self.signals = {}

  def process_bar(self, bar):
    ret = False
    close = float(bar['close'])
    date = bar['date']
    self.slow.add_value(close)
    self.fast.add_value(close)

    if self.slow.full():
      self.macd.add_value(self.fast.avg() - self.slow.avg())

    if self.macd.full():
      ret = True
      self.values[date] = self.macd.last()
      self.signals[date] = self.macd.avg()

    return ret
      
