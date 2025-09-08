#ifndef AHG_TSMAP_H__
#define AHG_TSMAP_H__
#include <unordered_map>
#include <mutex>

namespace autotrader
{

template <class K, class V>
class ThreadSafeMap
{
public:
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
private:
  std::unordered_map<K, V> map_;
  std::mutex lock_;
};

} // namespace

#endif // AHG_TSMAP_H__
