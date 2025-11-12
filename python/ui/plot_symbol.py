import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import matplotlib.pyplot as plt

class PlotSymbol(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent

  def populate(self, df, candlestick):
    for w in self.winfo_children(): w.destroy()
    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
#TODO investigate this, we are normalizing twice but it works this way only
    df_norm = (df - df.min()) / (df.max() - df.min())

    cmap = plt.get_cmap("tab10")
    addplots = [
      mpf.make_addplot(df_norm[col], color=cmap(i % cmap.N), width=1, label=col) for i, col in enumerate(df_norm.columns) if col not in self.candlestick_columns.values()
    ]
    for nn, name in self.candlestick_columns.items():
      df.rename(columns={name: nn}, inplace=True)

    mc_none = mpf.make_marketcolors(
        up='none', down='none',
        edge='none', wick='none',
        volume='none', inherit=True
    )
    style_none = mpf.make_mpf_style(marketcolors=mc_none)
    self.fig, ax = mpf.plot(
      df,
      type='candle',
      style='yahoo' if candlestick else style_none,
      addplot=addplots,
      title="Indicators + Candlesticks",
      returnfig=True
    )
    canvas = FigureCanvasTkAgg(self.fig, master=self)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

  def back(self):
    self.parent.show_analyze_symbol()

  def set_candlestick(self, candlestick_columns):
    self.candlestick_columns = candlestick_columns

  def close(self):
    try:
      plt.close(self.fig)
    except Exception:
        pass
