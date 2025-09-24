import tkinter as tk
from tkinter import ttk
import queue
from ui.start_page import StartPage
from ui.scanner_select import ScannerSelect
from ui.scanner_config import ScannerConfig
from ui.instrument_select import InstrumentSelect
from ui.symbol_select import SymbolSelect

class ATGui(tk.Tk):
  def __init__(self, driver):
    super().__init__()
    self.title("Auto Trader")
    self.geometry("340x180")
    self.driver = driver
    self.scanners = None
    self.filter = None
    self.current_page = None
    self.inbox = queue.Queue()

    self.start_page = StartPage(self)
    self.scanner_select = ScannerSelect(self)
    self.scanner_config = ScannerConfig(self)
    self.symbol_select = SymbolSelect(self)
    self.instrument_select = InstrumentSelect(self)

    self.pages = {"StartPage": self.start_page,
      "ScannerSelect": self.scanner_select,
      "ScannerConfig": self.scanner_config,
      "InstrumentSelect": self.instrument_select,
      "SymbolSelect": self.symbol_select,
    }
    self.show('InstrumentSelect')

    self.after(10, self.poll_inbox)

  def request_scanner_params(self):
    self.start_page.status.config(text="Requesting...")
    self.driver.request_scanner_params()

  def add_scanner_list(self, scanners):
    self.scanners = scanners
    self.filters = self.driver.get_filter_list()
    self.inbox.put('scanner_list')

  def configure_scanner(self, index):
    self.configure_index = index
    self.inbox.put('config_filter')

  def select_symbols(self):
    self.inbox.put('select_symbols')

  def process_symbols(self, symbols):
    if not symbols:
      return
    self.driver.process_symbols(symbols)

  def show(self, name):#, **kwargs):
    page = self.pages[name]
    if self.current_page:
      self.current_page.pack_forget()
    if name == "StartPage":
      self.driver.scanner_select = False
    self.current_page = page
    self.current_page.pack(fill="both", expand=True)

  def handle_command(self, command):
    if command == 'scanner_list':
      self.show("ScannerSelect")
      self.scanner_select.populate(self.scanners)
    if command == 'config_filter':
      self.show("ScannerConfig")
      self.scanner_config.populate(self.configure_index)
    if command == 'select_symbols':
      self.show('SymbolSelect')
      self.symbol_select.populate(self.driver.instrument_list())

  def poll_inbox(self):
    try:
      command = self.inbox.get_nowait()
    except queue.Empty:
      pass
    else:
      self.handle_command(command)
    finally:
      self.after(1, self.poll_inbox)

