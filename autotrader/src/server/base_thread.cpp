#include "server/base_thread.h"
#include <thread>
#include <chrono>

namespace autotrader
{

std::atomic<bool> BaseThread::running{false};

void BaseThread::start_all_threads()
{
  running = true;
}

void BaseThread::stop_all_threads()
{
  running = false;
}

void BaseThread::operator()()
{
  while (running)
  {
    step();
  }
}

BaseThread::BaseThread(zmq::context_t& ctx, std::string address, IBClient& ib, zmq::socket_type type) :
  socket_{ctx, type}, ib_{ib}
{
  socket_.bind(address);
}

zmq::recv_result_t BaseThread::recv(zmq::message_t& msg, bool wait)
{
  if (!wait)
  {
    return socket_.recv(msg, zmq::recv_flags::dontwait);
  }
  return socket_.recv(msg);
}

bool BaseThread::recv_all(std::vector<zmq::message_t>& frames)
{
  frames.clear();
  while (true)
  {
    zmq::message_t part;
    if (!socket_.recv(part, zmq::recv_flags::none))
    {
      return false;
    }
    frames.push_back(std::move(part));
    int more = socket_.get(zmq::sockopt::rcvmore);
    if (!more)
    {
      break;
    }
  }
  return true;
}

zmq::send_result_t BaseThread::send(zmq::message_t& msg, bool send_more)
{
  if (send_more)
  {
    return socket_.send(msg, zmq::send_flags::sndmore);
  }
  return socket_.send(msg, zmq::send_flags::none);
}

zmq::send_result_t BaseThread::send(const std::string& msg, bool send_more)
{
  if (send_more)
  {
    return socket_.send(zmq::buffer(msg), zmq::send_flags::sndmore);
  }
  return socket_.send(zmq::buffer(msg), zmq::send_flags::none);
}

void BaseThread::wait(size_t millisecs)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(millisecs));
}

size_t BaseThread::start_data_for_symbol(const std::string& symbol)
{
  return ib_.start_market_data_stream(symbol);
}

void BaseThread::stop_data_for_id(size_t id)
{
  ib_.stop_market_data_stream(id);
}

size_t BaseThread::request_historical_data(const std::string& symbol)
{
  return ib().request_historical_data(symbol);
}

IBClient& BaseThread::ib()
{
  return ib_;
}

} //namespace

