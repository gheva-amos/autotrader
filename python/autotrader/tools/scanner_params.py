import xml.etree.ElementTree as ET

class ScannerParams:
  def __init__(self, xml):
    self.root = ET.fromstring(xml)
    self.instruments = []
    self.instrument_map = {}
    self.locations = []
    self.scan_types = []
    self.filters = []
    self.filter_fields = {}
    self.instrument_locations = {}
    self.parse()

  @staticmethod
  def text(el, name, default=""):
      c = el.get(name)
      if not c:
        c = el.find(name)
      return c.text.strip() if (c is not None and c.text) else default

  @staticmethod
  def filter_list(string):
    return string.split(',')

  def walk_location_tree(self, location_root):
    locations = location_root.findall('./Location')
    for location in locations:
      name = self.text(location, 'displayName')
      code = self.text(location, 'locationCode')
      instruments = self.text(location, 'instruments').split(',')
      route_exchange = self.text(location, 'routeExchange')
      self.locations.append({'name': name, 'code': code, 'instruments': instruments, 'route_exchange': route_exchange})
      for inst in instruments:
        if inst not in self.instrument_locations:
          self.instrument_locations[inst] = []
        self.instrument_locations[inst].append(code)
      children = location.find('./LocationTree')
      if children is not None:
        self.walk_location_tree(children)

  def parse_instruments(self):
    for il in self.root.findall("./InstrumentList"):
      if il.attrib['varName'] == 'fullInstrumentList':
        continue
      for inst in il.findall("./Instrument"):
        self.instruments.append({
          "type": self.text(inst, "type"),
          "secType": self.text(inst, "secType"),
          "filters": self.filter_list(self.text(inst, "filters")),
          "name": self.text(inst, "name"),
          "group": self.text(inst, "group"),
          })
        self.instrument_map[self.text(inst, "type")] = self.filter_list(self.text(inst, "filters"))

  def parse_combo_fields(self, field):
    ret = []
    values = field.findall('./ComboValue')
    for value in values:
      ret.append({
        'code': self.text(value, 'code')
      })
    return ret

  def parse_field(self, field):
    ret = {}
    ret['code'] = self.text(field, 'code')
    ret['name'] = self.text(field, 'displayName')
    combo_fields = field.find('./ComboValues')
    if combo_fields is None:
      mini = self.text(field, 'minValue')
      maxi = self.text(field, 'maxValue')
      if mini:
        ret['min'] = mini
      if maxi:
        ret['max'] = maxi
    else:
      ret['combo'] = self.parse_combo_fields(combo_fields)
    return ret

  def parse_simple_filter(self, filt):
    elem = {}
    elem['id'] = self.text(filt, 'id')
    elem['category'] = self.text(filt, 'category')
    elem['access'] = self.text(filt, 'access')
    field = filt.find('./AbstractField')
    elem['fields'] = (self.parse_field(field))
    self.filter_fields[elem['id']] = elem['fields']
    self.filters.append(elem)

  def parse_range_filter(self, filt):
    elem = {}
    elem['id'] = self.text(filt, 'id')
    fields = filt.findall('./AbstractField')
    elem['fields'] = (self.parse_field(fields[0]), self.parse_field(fields[1]))
    self.filter_fields[elem['id']] = elem['fields']
    self.filters.append(elem)

  def parse_filters(self):
    filter_list = self.root.find('./FilterList')
    simple_filters = filter_list.findall('./SimpleFilter')

    for sf in simple_filters:
      self.parse_simple_filter(sf)

    range_filters = filter_list.findall('./RangeFilter')
    for rf in range_filters:
      self.parse_range_filter(rf)

  def parse_scan_types(self):
    for st in self.root.findall('./ScanTypeList/ScanType'):
      elem = {}
      elem['name'] = self.text(st, 'displayName')
      elem['code'] = self.text(st, 'scanCode')
      elem['instruments'] = self.text(st, 'instruments').split(',')
      self.scan_types.append(elem)

  def parse(self):
    self.parse_instruments()
    location_root = self.root.find("./LocationTree")
    self.walk_location_tree(location_root)
    self.parse_filters()
    self.parse_scan_types()

