from tkinter import ttk

class StartPage(ttk.Frame):
  def __init__(self, parent):
    super().__init__(parent)
#self.pack(fill="both", expand=True)
    self.status = ttk.Label(self, text="Idle")
    self.status.pack()

    ttk.Button(self, text="Fetch scanner parameters",
        command=parent.request_scanner_params).pack(pady=12)
    ttk.Button(self, text="Select symbols",
        command=parent.select_symbols).pack(pady=12)


