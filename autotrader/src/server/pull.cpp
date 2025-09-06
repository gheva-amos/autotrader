#include "server/pull.h"

namespace autotrader
{

Pull::Pull(zmq::context_t& ctx, std::string address, IBClient& ib) :
  BaseThread(ctx, address, ib, zmq::socket_type::pull)
{
}

void Pull::step()
{
  // TODO pull data from queue
}

} // namespace

