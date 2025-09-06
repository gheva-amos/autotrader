#include "server/pub.h"

namespace autotrader
{

Pub::Pub(zmq::context_t& ctx, std::string address, IBClient& ib) :
  BaseThread(ctx, address, ib, zmq::socket_type::pub)
{
}

void Pub::step()
{
  // TODO push data to listeners
}

} // namespace
