#ifndef AHG_MPSC_QUEUE_H__
#define AHG_MPSC_QUEUE_H__

#include <queue>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <chrono>

namespace autotrader
{

template <typename T>
class MPSCQueue
{
public:
  explicit MPSCQueue(size_t id_from = 0) :
    stop_{false}, counter_{id_from}
  {
  }

  MPSCQueue(const MPSCQueue&) = delete;
  MPSCQueue& operator=(const MPSCQueue&) = delete;
  MPSCQueue(MPSCQueue&&) = delete;
  MPSCQueue& operator=(MPSCQueue&&) = delete;

  size_t push(T value)
  {
    size_t ret{counter_.fetch_add(1, std::memory_order_relaxed)};
    {
      std::lock_guard<std::mutex> lock(lock_);
      data_.push(std::move(value));
    }
    available_.notify_one();
    return ret;
  }

  bool pop(T& ret)
  {
    std::unique_lock<std::mutex> lock(lock_);
    available_.wait_for(lock, std::chrono::milliseconds(100), [&]{ return stop_ || !data_.empty(); });
    //available_.wait(lock, [&]{ return stop_ || !data_.empty(); });
    if (data_.empty())
    {
      return false;
    }
    ret = std::move(data_.front());
    data_.pop();
    return true;
  }

  void stop()
  {
    stop_ = true;
    available_.notify_all();
  }
private:
  std::queue<T> data_;
  std::condition_variable available_;
  std::atomic<bool> stop_;
  std::atomic<size_t> counter_;
  std::mutex lock_;
};

} // namespace

#endif // AHG_MPSC_QUEUE_H__
