from tkinter import ttk
import tkinter as tk
from ui.scrollable import Scrollable

class ScannerConfig(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    ttk.Label(self, text="Configure Scanner").pack(pady=8)
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)
    self.parent = parent

  def populate(self, index):
    for w in self.box.inner.winfo_children(): w.destroy()
    scanner = self.parent.scanners[index]
    filters = self.parent.filters
    ttk.Label(self.box.inner, text=scanner[3]).pack(pady=8)
    ttk.Button(self.box.inner, text='Submit Scanner', command=lambda e=scanner: self.submit(e)).pack(pady=8)
    ttk.Separator(self.box.inner, orient="horizontal").pack(fill="x", pady=6)
    for i, filt in enumerate(filters):
      ttk.Label(self.box.inner, text=filt).pack(pady=8)
      var1 = tk.StringVar(value="")
      ttk.Label(self.box.inner, text=filters[filt][0]['name']).pack(pady=8)
      w1 = ttk.Entry(self.box.inner, textvariable=var1)
      w1.pack(pady=8)

      if len(filters[filt]) == 2:
        ttk.Label(self.box.inner, text=filters[filt][1]['name']).pack(pady=8)
        var2 = tk.StringVar(value="")
        w2 = ttk.Entry(self.box.inner, textvariable=var2)
        w2.pack(pady=8)

  def submit(self, scanner):
    self.parent.driver.request_scanner(scanner)
    self.parent.show("ScannerSelect")
    pass
