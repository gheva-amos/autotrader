import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from ui.scrollable import Scrollable

class AnalyzeSymbol(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
    self.parent = parent

    ttk.Button(self, text="Back", command=self.back).pack(pady=8)
    ttk.Button(self, text="Graph", command=self.show_graph).pack(pady=8)
    ttk.Button(self, text="Table", command=self.show_table).pack(pady=8)
    ttk.Button(self, text="Save", command=self.save).pack(pady=8)

    self.show_candlestick = tk.BooleanVar(value=False)
    tk.Checkbutton(self, text="Enable candlestick", variable=self.show_candlestick).pack(pady=8)
    
    self.box = Scrollable(self)
    self.box.pack(fill="both", expand=True, padx=12, pady=8)
    self.symbol_data = {}
    self.column_types = {}

  def populate(self, symbol):
    self.symbol_data = self.parent.driver.known_symbols()[symbol]

    for w in self.box.inner.winfo_children(): w.destroy()
    for col in self.symbol_data.columns:
      row_frame = ttk.Frame(self.box.inner)
      row_frame.pack(fill="x", pady=2)
      ttk.Label(row_frame, text=col, width=30).pack(side="left")
      var = tk.StringVar(value="none")
      self.column_types[col] = var
      for opt in ["display", "normalize", "none"]:
        ttk.Radiobutton(row_frame, text=opt.capitalize(), variable=var, value=opt).pack(side="left", padx=5)

  def back(self):
    self.parent.show_list_symbols()

  def show_graph(self):
    display = [col for col, var in self.column_types.items() if var.get() == "display"]
    normalize = [col for col, var in self.column_types.items() if var.get() == "normalize"]
    self.parent.show_plot_symbol(display, normalize, self.show_candlestick.get())

  def show_table(self):
    display = [col for col, var in self.column_types.items() if var.get() == "display"]
    normalize = [col for col, var in self.column_types.items() if var.get() == "normalize"]
    self.parent.show_table_symbol(display, normalize)

  def save(self):
    print(self.symbol_data)
    save_path = filedialog.asksaveasfilename(
      title="Save file as",
      defaultextension=".parquet",
      filetypes=[("Parquet files", "*.parquet")]
    )
    self.symbol_data.to_parquet(save_path)
