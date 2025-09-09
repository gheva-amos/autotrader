import xml.etree.ElementTree as ET

class ScannerParams:
  def __init__(self, xml):
    self.root = ET.fromstring(xml)
    self.instruments = []
    self.locations = []
    self.scan_types = []
    self.filters = []
    self.parse()

  @staticmethod
  def text(el, name, default=""):
      c = el.find(name)
      return c.text.strip() if (c is not None and c.text) else default

  def parse(self):
    for il in self.root.findall("./InstrumentList"):
      print(1)
      for inst in il.findall("./Instrument"):
        self.instruments.append({
          "type": inst.get("type") or self.text(inst, "type"),
          "secType": inst.get("secType") or self.text(inst, "secType"),
          "filters": inst.get("filters") or self.text(inst, "filters"),
          "name": inst.get("name") or self.text(inst, "name"),
          "group": inst.get("group") or self.text(inst, "group"),
          "assetClass": self.text(inst, "assetClass"),
          })
    print(self.instruments)
    pass
