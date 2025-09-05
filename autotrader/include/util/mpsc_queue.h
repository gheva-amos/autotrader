#ifndef AHG_MPSC_QUEUE_H__
#define AHG_MPSC_QUEUE_H__

#include <queue>
#include <condition_variable>
#include <atomic>
#include <mutex>

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

  size_t push(T value)
  {
    std::unique_lock<std::mutex> lock(lock_);
    size_t ret{counter_.fetch_add(1, std::memory_order_relaxed)};
    data_.push(std::move(value));
    lock.unlock();
    available_.notify_one();
    return ret;
  }

  bool pop(T& ret)
  {
    std::unique_lock<std::mutex> lock(lock_);
    available_.wait(lock, [&]{ return stop_ || !data_.empty(); });
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
    std::unique_lock<std::mutex> lock(lock_);
    stop_ = true;
    lock.unlock();
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
