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
    self.filter_fields = {}

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
      self.filter_fields[filt] = {}
      self.filter_fields[filt]['first'] = filters[filt][0]['code']
      self.filter_fields[filt]['first_value'] = var1

      if len(filters[filt]) == 2:
        ttk.Label(self.box.inner, text=filters[filt][1]['name']).pack(pady=8)
        var2 = tk.StringVar(value="")
        w2 = ttk.Entry(self.box.inner, textvariable=var2)
        w2.pack(pady=8)
        self.filter_fields[filt]['second'] = filters[filt][1]['code']
        self.filter_fields[filt]['second_value'] = var2

  def submit(self, scanner):
    apply_filters = {}
    for filt in self.filter_fields:
      if 'first' in self.filter_fields[filt]:
        if self.filter_fields[filt]['first_value'].get():
          apply_filters[self.filter_fields[filt]['first']] = self.filter_fields[filt]['first_value'].get()
      if 'second' in self.filter_fields[filt]:
        if self.filter_fields[filt]['second_value'].get():
          apply_filters[self.filter_fields[filt]['second']] = self.filter_fields[filt]['second_value'].get()
    self.parent.driver.request_scanner(scanner, apply_filters)
    self.parent.show("ScannerSelect")
    pass
