#include "server/router.h"
#include "debug.h"
namespace autotrader
{

Router::Router(zmq::context_t& ctx, std::string address, zmq::socket_type type) :
  BaseThread(ctx, address, type)
{
}

void Router::step()
{
  std::vector<zmq::message_t> frames;
  if (!recv_all(frames))
  {
    return;
  }
  if (frames.size() != 3)
  {
    return;
  }
  auto id{frames[1].to_string()};
  auto verb{frames[2].to_string()};
  if (verb == "status")
  {
    DBG_MSG("sending OK") << std::endl;
    send(frames[0], true);
    send(frames[1], true);
    send("OK");
  }
}

} // namespace

