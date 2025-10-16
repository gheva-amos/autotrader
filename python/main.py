from autotrader.driver import ATDriver
import time
import argparse
import importlib.util
import os
import sys
import json

def load_module(path):
  abs_path = os.path.abspath(path)
  sys.path.append(os.path.dirname(abs_path))
  name = os.path.splitext(os.path.basename(abs_path))[0]
  spec = importlib.util.spec_from_file_location(name, abs_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

def main(router_host, preprocessor_host, dist_host, col_host,  instrument):
  driver = ATDriver(router_host, preprocessor_host, dist_host, col_host, instrument)
  
  driver.start()

  driver.stop_thread()

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('config', default='python/config.json', help='the config file for this run')

  args = parser.parse_args()

  with open(args.config, 'r') as f:
    cfg = json.load(f)

  router = cfg['router']
  preprocessor = cfg['preprocessor']
  distributor = cfg['distributor']
  collector = cfg['collector']
  instrument = cfg['instrument']

  for model in cfg['models']:
    m = load_module(model['path'])
    if hasattr(m, "main"):
      m.main(distributor, collector, model['args'])

  main(router, preprocessor, distributor, collector, instrument)

