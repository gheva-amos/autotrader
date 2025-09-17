import tkinter as tk
from tkinter import ttk
from ui.scrollable import Scrollable

class InstrumentSelect(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent
    ttk.Button(self, text="Process Instruments", command=self.process_instruments).pack(pady=8)
    ttk.Label(self, text="Instruments").pack(pady=8)
    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)
    self.box.inner.columnconfigure(0, weight=1, uniform="cols")
    self.box.inner.columnconfigure(1, weight=1, uniform="cols")
    self.box.inner.columnconfigure(2, weight=1, uniform="cols")
    self.instrument_vars = {}

  def populate(self, instruments):
    for w in self.box.inner.winfo_children(): w.destroy()
    for i, inst in enumerate(instruments):
      var = tk.BooleanVar() 
      row, col = divmod(i, 3)
      ttk.Checkbutton(self.box.inner, text=inst, variable=var).grid(row=row, column=col, sticky="ew", padx=6, pady=6, ipadx=6, ipady=6)
      self.instrument_vars[inst] = var

  def process_instruments(self):
    instruments = [inst for inst in self.instrument_vars.keys() if self.instrument_vars[inst].get()]
    self.parent.process_instruments(instruments)

  def back(self):
    self.parent.show('ScannerSelect')
