#ifndef AHG_BASE_THREAD_H__
#define AHG_BASE_THREAD_H__
#include <atomic>
#include <string>
#include <zmq.hpp>

namespace autotrader
{

class BaseThread
{
public:
  BaseThread(zmq::context_t& ctx, std::string address, zmq::socket_type type);
  virtual ~BaseThread() = default;
  virtual void step() = 0;
  zmq::recv_result_t recv(zmq::message_t& msg, bool wait=true);
  bool recv_all(std::vector<zmq::message_t>& frames);
  zmq::send_result_t send(zmq::message_t& msg, bool send_more=false);
  zmq::send_result_t send(const std::string& msg, bool send_more=false);
  void operator()();
  static void start_all_threads();
  static void stop_all_threads();
  static std::atomic<bool> running;
  static void wait(size_t millisecs);
private:
  zmq::socket_t socket_;
};

} // namespace
#endif // AHG_BASE_THREAD_H__
