from autotrader.driver import ATDriver
import time
import argparse
import importlib.util
import os


def load_module(path):
  abs_path = os.path.abspath(path)
  name = os.path.splitext(os.path.basename(abs_path))[0]
  spec = importlib.util.spec_from_file_location(name, abs_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

def main(router_host, preprocessor_host, dist_host, col_host,  instrument):
  driver = ATDriver(router_host, preprocessor_host, dist_host, col_host, instrument)
  
  driver.start()

  driver.stop_thread()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--router', nargs=1, default='tcp://localhost:7001', help='the url for the router')
  parser.add_argument('--preprocessor', nargs=1, default='tcp://localhost:7002', help='the url for the preprocessor')
  parser.add_argument('--distributor', nargs=1, default='tcp://*:7007', help='the url for the distributor')
  parser.add_argument('--collector', nargs=1, default='tcp://*:7006', help='the url for the collector')
  parser.add_argument('--instrument', nargs=1, default='STK', help='the instrument to trade')
  parser.add_argument("--modules", nargs="*", help='list of models to load')

  args = parser.parse_args()

  for module in args.modules:
    m = load_module(module)
    if hasattr(m, "main"):
      m.main()

  main(args.router, args.preprocessor, args.distributor, args.collector, args.instrument)

