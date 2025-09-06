#include "server/router.h"
#define DEBUG
#include "debug.h"
namespace autotrader
{

Router::Router(zmq::context_t& ctx, std::string address, IBClient& ib) :
  BaseThread(ctx, address, ib, zmq::socket_type::router)
{
}

void Router::step()
{
  std::vector<zmq::message_t> frames;
  if (!recv_all(frames))
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
  else if (verb == "data")
  {
    process_data_req(frames);
  }
  else if (verb == "stop_data")
  {
    stop_data_req(frames);
  }
}

void Router::process_data_req(std::vector<zmq::message_t>& frames)
{
  auto symbol = frames[3].to_string();
  size_t req_id = start_data_for_symbol(symbol);
  send(frames[0], true);
  send(req_id);
}

void Router::stop_data_req(std::vector<zmq::message_t>& frames)
{
  size_t req_id = std::atol(frames[3].to_string().c_str());
  DBG_MSG(req_id) <<frames[0].to_string()<<std::endl;
  stop_data_for_id(req_id);
  send(frames[0], true);
  send(frames[3], true);
  send("OK");
}

} // namespace

