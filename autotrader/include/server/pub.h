#ifndef AHG_PUB_H__
#define AHG_PUB_H__

#include "server/base_thread.h"

namespace autotrader
{

class Pub : public BaseThread
{
public:
  Pub(zmq::context_t& ctx, std::string address, IBClient& ib);
  virtual void step() override;
};

} // namespace

#endif // AHG_PUB_H__
