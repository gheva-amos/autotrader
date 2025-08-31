#include "debug.h"
#include <iostream>

namespace
{
struct nullbuf : std::streambuf
{
  int overflow(int c) override { return c; }
};

} // namespace

namespace autotrader
{

std::ostream& get_log_stream(bool enabled)
{
  static nullbuf nb;
  static std::ostream null_stream(&nb);
  return enabled ? std::cout : null_stream;
}

} // namespace
