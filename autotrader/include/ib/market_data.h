#ifndef AHG_MRKET_DATA_H__
#define AHG_MRKET_DATA_H__
#include <string>
#include <unordered_map>
#include "EWrapper.h"

namespace autotrader
{

class MarketData
{
public:
  double& price(TickType t);
  double price(TickType t) const;
  int& size(TickType t);
  int size(TickType t) const;
  double& generic(TickType t);
  double generic(TickType t) const;
  std::string& value(TickType t);
  std::string value(TickType t) const;
private:
  std::unordered_map<TickType, double> price_;
  std::unordered_map<TickType, int> size_;
  std::unordered_map<TickType, double> generic_;
  std::unordered_map<TickType, std::string> value_;
};

} // namespace

#endif // AHG_MRKET_DATA_H__
