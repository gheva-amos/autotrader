#ifndef AHG_ROUTER_H__
#define AHG_ROUTER_H__

#include "server/base_thread.h"

namespace autotrader
{

class Router : public BaseThread
{
public:
  Router(zmq::context_t& ctx, std::string address, IBClient& ib);
  virtual void step() override;
private:
  void process_data_req(std::vector<zmq::message_t>& frames);
  void stop_data_req(std::vector<zmq::message_t>& frames);
};

} // namespace

#endif // AHG_ROUTER_H__
