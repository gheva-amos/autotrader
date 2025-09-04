#include "ib/market_data.h"

namespace autotrader
{

double& MarketData::price(TickType t)
{
  return price_[t];
}

double MarketData::price(TickType t) const
{
  return price_.at(t);
}

int& MarketData::size(TickType t)
{
  return size_[t];
}

int MarketData::size(TickType t) const
{
  return size_.at(t);
}

double& MarketData::generic(TickType t)
{
  return generic_[t];
}

double MarketData::generic(TickType t) const
{
  return generic_.at(t);
}

std::string& MarketData::value(TickType t)
{
  return value_[t];
}

std::string MarketData::value(TickType t) const
{
  return value_.at(t);
}

} // namespace

