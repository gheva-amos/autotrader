#ifndef AHG_TSVECTOR_H__
#define AHG_TSVECTOR_H__

#include <vector>
#include <mutex>

namespace autotrader
{

template <class T>
class ThreadSafeVector
{
public:
  size_t push_back(T element)
  {
    std::unique_lock<std::mutex> lock(lock_);
    size_t ret{vector_.size()};
    vector_.push_back(element);
    return ret;
  }

  size_t size()
  {
    std::unique_lock<std::mutex> lock(lock_);
    return vector_.size();
  }

  T operator[](size_t index) const
  {
    return vector_.at(index);
  }

  T& operator[](size_t index)
  {
    std::unique_lock<std::mutex> lock(lock_);
    return vector_[index];
  }

private:
  std::vector<T> vector_;
  std::mutex lock_;
};

} // namespace

#endif // AHG_TSVECTOR_H__
