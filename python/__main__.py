import sys
from pathlib import Path

if __name__ == "__main__":
  main()

def main():
  PKG_ROOT = Path(__file__).resolve().parent
  if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
  from .main import run
  run()
