#ifndef AHG_TSMAP_H__
#define AHG_TSMAP_H__
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace autotrader
{

template <class K, class V>
class ThreadSafeMap
{
public:
  ThreadSafeMap() :
    counter_{0}
  {
  }
  void insert(std::pair<K, V> e)
  {
    std::unique_lock<std::mutex> lock(lock_);
    map_.insert(e);
  }

  size_t size()
  {
    std::unique_lock<std::mutex> lock(lock_);
    return map_.size();
  }
  V operator[](K key) const
  {
    return map_.at(key);
  }
  V& operator[](K key)
  {
    std::unique_lock<std::mutex> lock(lock_);
    return map_[key];
  }
  size_t next_id()
  {
    return counter_.fetch_add(1, std::memory_order_relaxed);
  }
private:
  std::unordered_map<K, V> map_;
  std::mutex lock_;
  std::atomic<size_t> counter_;
};

} // namespace

#endif // AHG_TSMAP_H__
