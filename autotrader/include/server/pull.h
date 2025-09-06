#ifndef AHG_PULL_H__
#define AHG_PULL_H__

#include "server/base_thread.h"

namespace autotrader
{

class Pull : public BaseThread
{
public:
  Pull(zmq::context_t& ctx, std::string address, IBClient& ib);
  virtual void step() override;
};

} // namespace

#endif // AHG_PULL_H__
