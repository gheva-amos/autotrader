#ifndef AHG_ROUTER_H__
#define AHG_ROUTER_H__

#include "server/base_thread.h"

namespace autotrader
{

class Router : public BaseThread
{
public:
  Router(zmq::context_t& ctx, std::string address, zmq::socket_type type);
  virtual void step() override;
private:
};

} // namespace

#endif // AHG_ROUTER_H__
