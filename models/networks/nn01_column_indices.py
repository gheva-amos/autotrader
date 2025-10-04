
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
