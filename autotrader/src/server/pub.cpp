#include "server/pub.h"
#define DEBUG
#include "debug.h"

namespace autotrader
{

Pub::Pub(zmq::context_t& ctx, std::string address, IBClient& ib) :
  BaseThread(ctx, address, ib, zmq::socket_type::pub)
{
}

void Pub::step()
{
  size_t id;
  if (ib().next_historical_id(id))
  {
    auto bars = ib().historical_bars(id);
    for (auto bar : bars)
    {
      send("history", true);
      send_num(id, true);
      send(bar.time, true);
      send_num(bar.high, true);
      send_num(bar.low, true);
      send_num(bar.open, true);
      send_num(bar.close, true);
      send_num(bar.wap, true);
      send_num(bar.volume, true);
      send_num(bar.count, true);
      send("OK");
    }
  }
  if (ib().next_scanner_id(id))
  {
    auto details = ib().scanner_data(id);
    for (auto detail : details)
    {
    }
  }
  std::string xml;
  if (ib().scanner_params(xml))
  {
    send("scanner_params", true);
    send(xml, true);
    send("OK");
  }
}

} // namespace
