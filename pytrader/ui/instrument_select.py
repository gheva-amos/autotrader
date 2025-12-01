import tkinter as tk
from tkinter import ttk
from ui.scrollable import Scrollable

class InstrumentSelect(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent
    self.instrument_types = ['STK', 'ETF.EQ.US', 'ETF.FI.US']
    self.selected = tk.StringVar(value=self.instrument_types[0])
    self.dropdown = tk.OptionMenu(self, self.selected, *self.instrument_types)
    self.dropdown.pack()
    tk.Button(self, text="OK", command=self.start).pack()

  def start(self):
    self.parent.driver.set_instrument(self.selected.get())
    self.parent.show('StartPage')
    pass

