import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PlotSymbol(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent

  def populate(self, df):
    for w in self.winfo_children(): w.destroy()
    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    df_norm = (df - df.min()) / (df.max() - df.min())
    fig = Figure(figsize=(6,3), dpi=100)
    ax = fig.add_subplot(111)
    df_norm.plot(ax=ax, lw=1.2)
    ax.grid(True, alpha=0.3)
    canvas = FigureCanvasTkAgg(fig, master=self)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

  def back(self):
    self.parent.show_analyze_symbol()
