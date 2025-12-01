import tkinter as tk
from tkinter import ttk
from ui.scrollable import Scrollable

class ListSymbols(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent
    ttk.Label(self, text="Select symbol to analyze").pack(pady=8)

    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)

  def populate(self, symbols):
    for w in self.box.inner.winfo_children(): w.destroy()
    for symbol in symbols.keys():
      btn = ttk.Button(self.box.inner, text=symbol,
          command=lambda sym=symbol: self.on_select(sym)).pack(pady=8)
      
  def on_select(self, symbol):
    self.parent.show_analyze_symbol(symbol)

  def back(self):
    self.parent.select_symbols()
