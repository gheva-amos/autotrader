from autotrader.lib.base_model import BaseModel
from moving_average import MovingAverage
from macd import MACD
from tsi import TSI
from rsi import RSI
from obv import OBV
import json

class Indicators(BaseModel):
  def __init__(self, listen, send):
    super().__init__("indicators", listen, send)
    self.start_date = {}
    self.dates = {}
    self.labels = {}
    self.moving_averages = {}
    self.macds = {}
    self.tsis = {}
    self.rsis = {}
    self.obvs = {}

  def process_moving_averages(self, symbol, bar):
    if symbol not in self.moving_averages:
      self.moving_averages[symbol] = MovingAverage(100)
    return self.moving_averages[symbol].process_bar(bar)

  def process_macd(self, symbol, bar):
    if symbol not in self.macds:
      self.macds[symbol] = MACD(26, 12, 9)
    return self.macds[symbol].process_bar(bar)

  def process_tsis(self, symbol, bar):
    if symbol not in self.tsis:
      self.tsis[symbol] = TSI(25, 13)
    return self.tsis[symbol].process_bar(bar)

  def process_rsis(self, symbol, bar):
    if symbol not in self.rsis:
      self.rsis[symbol] = RSI(14)
    return self.rsis[symbol].process_bar(bar)

  def process_obvs(self, symbol, bar):
    if symbol not in self.obvs:
      self.obvs[symbol] = OBV()
    return self.obvs[symbol].process_bar(bar)

  def send_results_for(self, symbol):
    msg = {
      'symbol': symbol,
      'start_date': self.start_date[symbol] if symbol in self.start_date else '',
      'dates': self.dates[symbol],
      'labels': self.labels[symbol],
      'average': self.moving_averages[symbol].averages,
      'macd': (self.macds[symbol].values, self.macds[symbol].signals),
      'tsi': self.tsis[symbol].tsis,
      'rsi': self.rsis[symbol].rsis,
      'obv': self.obvs[symbol].obv,
    }
    self.send_frames(['indicators'.encode(), symbol.encode(), json.dumps(msg).encode()])


  def handle_frames(self, frames):
    if frames[1].decode() == 'historical_bar_done':
      self.send_results_for(frames[2].decode())
      return
    if not frames[1].decode() == 'historical_bar':
      return
    bar = json.loads(frames[2].decode())
    symbol = bar['symbol']
    if not symbol in self.dates:
      self.dates[symbol] = []
    self.dates[symbol].append(bar['date'])
    if not symbol in self.labels:
      self.labels[symbol] = {}
    self.labels[symbol][bar['date']] = bar['close']
    r1 = self.process_moving_averages(symbol, bar)
    r2 = self.process_macd(symbol, bar)
    r3 = self.process_tsis(symbol, bar)
    r4 = self.process_rsis(symbol, bar)
    r5 = self.process_obvs(symbol, bar)
    if r1 and r2 and r3 and r4 and r5 and symbol not in self.start_date:
      self.start_date[symbol] = bar['date']

def main(listen, send):
  ind = Indicators(listen, send)
  ind.start()
  print('loaded')
