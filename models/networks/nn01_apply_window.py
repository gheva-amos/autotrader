import numpy as np

def apply_window(df, window):
  X_all = df.drop(columns=['close', 'target', 'next', 'hist']).to_numpy(dtype="float32")
  Y_next = df['next'].to_numpy(dtype='float32')
  Y_hist = df['hist'].to_numpy(dtype='float32')
  Y_target = df['target'].to_numpy(dtype='float32')
  Xs, yns, yhs, ytarg = [], [], [], []
  for i in range(window-1, len(X_all)):
    Xs.append(X_all[i-window+1:i+1])
    yns.append(Y_next[i])
    yhs.append(Y_hist[i])
    ytarg.append(Y_target[i])

  X, yN, yH, yT = [], [], [], []
  if Xs:
    X = np.stack(Xs).astype("float32")
    yN = np.array(yns, dtype="float32")
    yH = np.array(yhs, dtype="float32")
    yT = np.array(ytarg, dtype="float32")

  return (X, yN, yH, yT)

