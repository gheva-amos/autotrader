from autotrader.lib.base_model import BaseModel
import pandas as pd
import json
import threading
import queue
import ta
import io
from models.nn001.nn001 import NN01Model

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

class Model001(BaseModel):
  def __init__(self, listen, send, send_unprocessed=False, models=[]):
    super().__init__("Model001", listen, send)
    self.data_frames = {}
    self.done_names = queue.Queue()
    self.processor_thread = threading.Thread(target=self.process_thread, daemon=True)
    self.processor_thread.start()
    self.send_unprocessed = send_unprocessed
    self.models = {}
    self.load_models(models)

  def load_models(self, models_args):
    for m in models_args:
      model = NN01Model(m['checkpoint'], m['window'])
      self.models[m['window']] = model

  def send_results_for(self, symbol):
    self.done_names.put(symbol)

  def add_indicators(self, df):
    if "date" in df.columns:
      df["date"] = pd.to_datetime(df["date"])
      df = df.set_index("date").sort_index()
    try:
      cols = ["open","high","low","close","volume"]
      
      df[cols] = df[cols].apply(pd.to_numeric)
    except:
      pass
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    atr = ta.volatility.AverageTrueRange(
      high=df["high"], low=df["low"], close=df["close"], window=14
      )
    df["atr14"] = atr.average_true_range()

    df["stoch_k"] = ta.momentum.StochasticOscillator(
      df["high"], df["low"], df["close"], window=14, smooth_window=3
    ).stoch()

    df["stoch_d"] = ta.momentum.StochasticOscillator(
      df["high"], df["low"], df["close"], window=14, smooth_window=3
    ).stoch_signal()

    df["sma20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["sma50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    boll = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_mavg"] = boll.bollinger_mavg()
    df["bb_high"] = boll.bollinger_hband()
    df["bb_low"]  = boll.bollinger_lband()

    df = self.preprocess(df)
    return df

  def process_thread(self):
    while True:
      symbol = self.done_names.get()
      if symbol is None:
        break
      df = self.data_frames[symbol]
      if not df.empty:
        try:
          df = self.add_indicators(df)
        except:
          print(f"exception {symbol}")
          continue
        self.run_models(symbol, df)
        if self.send_unprocessed:
          buf = io.BytesIO()
          df.to_parquet(buf, index=True)
          self.send_frames(['indicators_pd'.encode(), symbol.encode(), buf.getvalue()])
      
  def run_models(self, symbol, df):
    for win, model in self.models.items():
      buf = io.BytesIO()
      model_name = "model001_window_" + str(win)
      tmp = model.eval(df)
      tmp = self.eval_returns(tmp)
      tmp.to_parquet(buf, index=True)
      self.send_frames(['data_frames'.encode(), symbol.encode(), model_name.encode(), buf.getvalue()])

  def eval_returns(self, df):
    df = df.copy()
    df['strategy'] = (df['hist'] > 0.5).astype(int)
    df['buy'] = (df['strategy'].diff() == 1).astype(int)
    df['sell'] = (df['strategy'].diff() == -1).astype(int)

    initial_capital = 1000
    cash = initial_capital
    shares = 0

    portfolio_values = []
    cash_list = []
    shares_list = []

    for i, row in df.iterrows():
      price = row["close"]

      if row['buy'] == 1:
        shares = cash / price
        cash = 0

      if row['sell'] == 1:
        cash = shares * price
        shares = 0

      portfolio_value = cash + shares * price
      portfolio_values.append(portfolio_value)
      cash_list.append(cash)
      shares_list.append(shares)

    df["portfolio"] = portfolio_values
    df = df.drop(columns=['strategy', 'buy', 'sell', 'close'])

    return df

  def handle_frames(self, frames):
    if frames[1].decode() == 'historical_bar_done':
      self.send_results_for(frames[2].decode())
      return
    if not frames[1].decode() == 'historical_bar':
      return
    bar = json.loads(frames[2].decode())
    symbol = bar['symbol']
    if symbol not in self.data_frames:
      self.data_frames[symbol] = pd.DataFrame(columns=["open","high","low","close","volume"])
      self.data_frames[symbol].index.name = "date"
    row = {}
    row['date'] = pd.to_datetime(bar['date'])
    for entry in ["open","high","low","close","volume"]:
      row[entry] = bar[entry]
    self.data_frames[symbol] = pd.concat([self.data_frames[symbol], pd.DataFrame([row])])

  def align_features(self, df, cols):
    df = df.copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
      if c not in df: df[c] = pd.NA
    df = df.loc[:, cols]
    return df.dropna(subset=cols)

  def preprocess(self, df):
    ret = {}
    num_cols = df.select_dtypes(include="number").columns
    df_norm = df.copy()
    df_norm.sort_index()
    df_norm[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std().replace(0, 1e-8)
    df_norm.index = df.index
    df_norm['next'] = (df['close'].shift(-1) > df['close']).astype('int8')
    df_norm['hist'] = ((df['close'].diff().rolling(10).sum() > 0)).astype('int8')
    df_norm["target"] = df["close"].shift(-1)
    df_norm = self.ema(df_norm, ['open', 'close', 'high', 'low', 'macd_diff', 'stoch_k', 'stoch_d'])
    df_norm = self.roll_mean(df_norm, ['open', 'close', 'high', 'low', 'macd_diff', 'stoch_k', 'stoch_d'])
    df_norm = self.roll_median(df_norm, ['open', 'close', 'high', 'low', 'macd_diff', 'stoch_k', 'stoch_d'])
    df_norm = self.align_features(df_norm, ALL_COLUMNS)
    df_norm = df_norm.dropna()

    return df_norm

  def ema(self, df, cols, span=3, suffix="_ema3"):
    for c in cols:
      df[c+suffix] = df[c].ewm(span=span, adjust=False).mean()
    return df

  def roll_mean(self, df, cols, win=3, suffix="_m3"):
    for c in cols:
      df[c+suffix] = df[c].rolling(win, min_periods=1).mean()  # center=False (causal)
    return df

  def roll_median(self, df, cols, win=3, suffix="_med3"):
    for c in cols:
      df[c+suffix] = df[c].rolling(win, min_periods=1).median()
    return df


def main(listen, send, args=None):
  send_unprocessed = args['send_unprocessed']
  models = args['models']
  ind = Model001(listen, send, send_unprocessed, models)
  ind.start()
  print('loaded')
