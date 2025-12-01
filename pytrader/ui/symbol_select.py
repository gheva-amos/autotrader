import tkinter as tk
from tkinter import ttk
from ui.scrollable import Scrollable

class SymbolSelect(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent
    ttk.Label(self, text="Enter symbol manually").pack(pady=8)
    self.manual_symbol = tk.StringVar(value="")
    ttk.Entry(self, textvariable=self.manual_symbol).pack(pady=8)
    ttk.Button(self, text="Process Symbols", command=self.process_symbols).pack(pady=8)
    ttk.Label(self, text="Symbols").pack(pady=8)
    ttk.Button(self, text="Analyze", command=self.analyze).pack(pady=8)
    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)
    self.box.inner.columnconfigure(0, weight=1, uniform="cols")
    self.box.inner.columnconfigure(1, weight=1, uniform="cols")
    self.box.inner.columnconfigure(2, weight=1, uniform="cols")
    self.symbol_vars = {}

  def populate(self, symbols):
    for w in self.box.inner.winfo_children(): w.destroy()
    for i, sym in enumerate(symbols):
      var = tk.BooleanVar() 
      row, col = divmod(i, 3)
      ttk.Checkbutton(self.box.inner, text=sym, variable=var).grid(row=row, column=col, sticky="ew", padx=6, pady=6, ipadx=6, ipady=6)
      self.symbol_vars[sym] = var

  def process_symbols(self):
    symbols = [sym for sym in self.symbol_vars.keys() if self.symbol_vars[sym].get()]
    if self.manual_symbol.get():
      symbols.append(self.manual_symbol.get())
    self.parent.process_symbols(symbols)

  def back(self):
    self.parent.show('ScannerSelect')

  def analyze(self):
    self.parent.show_list_symbols()

