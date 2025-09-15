import tkinter as tk
from tkinter import ttk
# This file is mostly done by chatgpt,with some debugging by me

class Scrollable(ttk.Frame):
  def __init__(self, parent, *, height=400):
    super().__init__(parent)
    self.canvas = tk.Canvas(self, highlightthickness=0, height=height)
    self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

    self.inner = ttk.Frame(self.canvas)

    self.inner.bind("<Configure>", self._on_inner_configure)

    self.inner_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
    self.canvas.bind("<Configure>", self._on_canvas_configure)
    self.canvas.configure(yscrollcommand=self.vbar.set)

    self.canvas.pack(side="left", fill="both", expand=True)
    self.vbar.pack(side="right", fill="y")

    # (optional) mousewheel support
    self.canvas.bind("<Enter>", lambda e: self._on_enter())
    self.canvas.bind("<Leave>", lambda e: self._on_leave())
    self._wheel_bound = False

  def _on_inner_configure(self, _):
    self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    # keep inner width equal to visible width
    self.canvas.itemconfigure(1, width=self.canvas.winfo_width())

  def _bind_mouse(self):
    self.canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")        # Windows/macOS
    self.canvas.bind_all("<Button-4>", self._on_wheel, add="+")          # Linux up
    self.canvas.bind_all("<Button-5>", self._on_wheel, add="+")          # Linux down

  def _unbind_mouse(self):
    self.canvas.unbind_all("<MouseWheel>")
    self.canvas.unbind_all("<Button-4>")
    self.canvas.unbind_all("<Button-5>")

  def _on_wheel(self, event):
    print("here")
    delta = -1*(event.delta//120) if event.delta else ( -1 if event.num==5 else 1 )
    self.canvas.yview_scroll(delta, "units")

  def _on_canvas_configure(self, event):
    # Keep inner frame width equal to visible width
    if not self.winfo_exists():
      return
    self.canvas.itemconfigure(self.inner_window, width=event.width)

  def _on_enter(self):
    self.canvas.focus_set()
    self._bind_mouse()
    pass

  def _on_leave(self):
    self._unbind_mouse()
    pass

