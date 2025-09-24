from tkinter import ttk
from ui.scrollable import Scrollable

class ScannerSelect(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    ttk.Label(self, text="Scanners").pack(pady=8)
    ttk.Button(self, text="Select Symbol", command=self.select_symbols).pack(pady=8)
    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)
    self.box.inner.columnconfigure(0, weight=1, uniform="cols")
    self.box.inner.columnconfigure(1, weight=1, uniform="cols")
    self.box.inner.columnconfigure(2, weight=1, uniform="cols")
    self.parent = parent

  def populate(self, scanners):
    for w in self.box.inner.winfo_children(): w.destroy()
    for i, scanner in enumerate(scanners):
      label = scanner[3]
      row, col = divmod(i, 3)
      btn = ttk.Button(self.box.inner, text=label,
          command=lambda c=i: self.on_select(c))
      btn.grid(row=row, column=col, sticky="ew", padx=6, pady=6, ipadx=6, ipady=6)
    root = self.winfo_toplevel()
    root.geometry("1200x900")

  def on_select(self, index):
    self.parent.configure_scanner(index)

  def back(self):
    self.parent.show('StartPage')

  def select_symbols(self):
    self.parent.select_symbols()


