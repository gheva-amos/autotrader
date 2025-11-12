import tkinter as tk
from tkinter import ttk
import queue
from ui.start_page import StartPage
from ui.scanner_select import ScannerSelect
from ui.scanner_config import ScannerConfig
from ui.instrument_select import InstrumentSelect
from ui.symbol_select import SymbolSelect
from ui.list_symbols import ListSymbols
from ui.analyze_symbol import AnalyzeSymbol
from ui.plot_symbol import PlotSymbol
from ui.table_symbol import TableSymbol

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

    self.protocol("WM_DELETE_WINDOW", self.on_close)

    self.start_page = StartPage(self)
    self.scanner_select = ScannerSelect(self)
    self.scanner_config = ScannerConfig(self)
    self.symbol_select = SymbolSelect(self)
    self.instrument_select = InstrumentSelect(self)
    self.list_symbols = ListSymbols(self)
    self.analyze_symbol = AnalyzeSymbol(self)
    self.plot_symbol = PlotSymbol(self)
    self.table_symbol = TableSymbol(self)
    self.symbol = ''

    self.pages = {"StartPage": self.start_page,
      "ScannerSelect": self.scanner_select,
      "ScannerConfig": self.scanner_config,
      "InstrumentSelect": self.instrument_select,
      "SymbolSelect": self.symbol_select,
      "ListSymbols": self.list_symbols,
      "AnalyzeSymbol": self.analyze_symbol,
      "PlotSymbol": self.plot_symbol,
      "TableSymbol": self.table_symbol,
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

  def show_list_symbols(self):
    self.inbox.put('list_symbols')

  def show_analyze_symbol(self, symbol=None):
    if symbol is not None:
      self.symbol = symbol
    self.inbox.put('analyze_symbol')

  def show_plot_symbol(self, display, normalize, candlestick=False):
    self.display_columns = display
    self.normalize_columns = normalize
    self.candlestick = candlestick
    self.inbox.put('plot_symbol')

  def show_table_symbol(self, display, normalize):
    self.display_columns = display
    self.normalize_columns = normalize
    self.candlestick = False
    self.inbox.put('table_symbol')

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
    if command == 'list_symbols':
      self.show('ListSymbols')
      self.list_symbols.populate(self.driver.known_symbols())
    if command == 'analyze_symbol':
      self.show('AnalyzeSymbol')
      self.analyze_symbol.populate(self.symbol)
    if command == 'plot_symbol':
      self.show('PlotSymbol')
      self.plot_symbol.populate(self.select_columns(), self.candlestick)
    if command == 'table_symbol':
      self.show('TableSymbol')
      self.table_symbol.populate(self.select_columns())

  def poll_inbox(self):
    try:
      command = self.inbox.get_nowait()
    except queue.Empty:
      pass
    else:
      self.handle_command(command)
    finally:
      self.after_id = self.after(1, self.poll_inbox)

  def select_columns(self):
    df = self.driver.known_symbols()[self.symbol]
    max_val = df[self.display_columns + self.normalize_columns].to_numpy().max()
    min_val = df[self.display_columns + self.normalize_columns].to_numpy().min()
    dev_by = max_val - min_val
    if dev_by == 0:
      dev_by = 1e-8

    ret = df[self.display_columns].copy()
    for col in self.normalize_columns:
      if col in df:
        ret[col] = df[col]
    for col in self.candlestick_columns:
      if not col in ret:
        ret[col] = df[col]
    return ret

  def set_candlestick(self, candlestick_columns):
    self.candlestick_columns = [col for col in candlestick_columns.values()]
    self.plot_symbol.set_candlestick(candlestick_columns)

  def on_close(self):
    if self.after_id is not None:
      self.after_cancel(self.after_id)
    self.plot_symbol.close()
    self.destroy()
