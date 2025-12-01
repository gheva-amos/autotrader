import tkinter as tk
from tkinter import ttk

class TableSymbol(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent

  def populate(self, df):
    for w in self.winfo_children(): w.destroy()
    ttk.Button(self, text="Back", command=self.back).pack(pady=8)

    frame = ttk.Frame(self)
    frame.pack(fill="both", expand=True)
    vsb = ttk.Scrollbar(frame, orient="vertical")
    hsb = ttk.Scrollbar(frame, orient="horizontal")
    cols = ["index"] + list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show="headings",
      yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.config(command=tree.yview); vsb.pack(side="right", fill="y")
    hsb.config(command=tree.xview); hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    for c in cols:
      tree.heading(c, text=c)
      tree.column(c, width=100, anchor="center")
    for idx, row in df.iterrows():
      vals = [idx] + [row[c] for c in df.columns]
      tree.insert("", "end", values=vals)

  def back(self):
    self.parent.show_analyze_symbol()

