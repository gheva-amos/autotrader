import tkinter as tk
from tkinter import ttk
from ui.scrollable import Scrollable

class AnalyzeSymbol(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent

    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)
    self.symbol_data = {}

  def populate(self, symbol):
    self.symbol_data = self.parent.driver.known_symbols()[symbol]

    for w in self.box.inner.winfo_children(): w.destroy()
    for label in self.symbol_data:
      ttk.Label(self.box.inner, text=label).pack(pady=8)
      ttk.Button(self.box.inner, text='Graph',
          command=lambda l=label: self.show_graph(l)).pack(pady=8)
      ttk.Button(self.box.inner, text='Table',
          command=lambda l=label: self.show_table(l)).pack(pady=8)

  def back(self):
    self.parent.show_list_symbols()

  def show_graph(self, label):
    self.parent.show_plot_symbol(label)

  def show_table(self, label):
    self.parent.show_table_symbol(label)
