#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace struct_mapping
{
// Observational metadata only: never retain a sensor message or delay delivery.
// Callers use a monotonic clock and consume on either delivery or filter failure.
class IngressTimingBuffer
{
public:
  explicit IngressTimingBuffer(std::size_t capacity = 256) : capacity_(capacity) {}

  void record(const void* key, double now_sec)
  {
    if (key == nullptr || !std::isfinite(now_sec))
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    eraseLocked(key);
    entries_.emplace_back(key, now_sec);
    while (entries_.size() > capacity_)
      entries_.pop_front();
  }

  double takeMs(const void* key, double now_sec)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
    {
      if (it->first != key)
        continue;
      const double elapsed = (now_sec - it->second) * 1000.0;
      entries_.erase(it);
      return std::isfinite(elapsed) && elapsed >= 0.0 ? elapsed : -1.0;
    }
    return -1.0;  // Evicted/unobserved is missing, not a zero-latency sample.
  }

private:
  void eraseLocked(const void* key)
  {
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
    {
      if (it->first == key)
      {
        entries_.erase(it);
        return;
      }
    }
  }

  const std::size_t capacity_;
  std::mutex mutex_;
  std::deque<std::pair<const void*, double>> entries_;
};
}  // namespace struct_mapping
