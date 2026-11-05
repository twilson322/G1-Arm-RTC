// Real-time plumbing between the 1 kHz control thread and everything else.
//
//  TripleBuffer<T>  one writer, one reader, latest value wins. Neither side ever blocks or waits,
//                   and the payload is never shared while being written (ownership moves through
//                   one atomic index), so it is safe for the control thread on both ends.
//  SpscQueue<T, N>  bounded lock-free ring for log records: the control thread pushes, a logger
//                   thread drains to disk. A full queue drops the record and counts it instead of
//                   blocking the control loop.
//  PeriodicTimer    absolute-deadline sleeping on CLOCK_MONOTONIC, so jitter does not accumulate.
//  LatencyHistogram fixed bins, no allocation after construction.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

namespace g1rtc::rt {

template <class T>
class TripleBuffer {
 public:
  TripleBuffer() = default;
  explicit TripleBuffer(const T& init) { buf_.fill(init); }
  /// Give every slot a value (e.g. to pre-size dynamic members). Only before the threads start.
  void fill(const T& v) { buf_.fill(v); }
  /// Writer: fill this, then publish().
  T& writeBuffer() { return buf_[write_]; }
  void publish() {
    const unsigned prev = middle_.exchange(write_ | kFresh, std::memory_order_acq_rel);
    write_ = prev & kIndex;
  }
  /// Reader: returns true if a newer value was swapped in. read() is valid either way.
  bool update() {
    if ((middle_.load(std::memory_order_relaxed) & kFresh) == 0) return false;
    const unsigned prev = middle_.exchange(read_, std::memory_order_acq_rel);
    read_ = prev & kIndex;
    return true;
  }
  const T& read() const { return buf_[read_]; }

 private:
  static constexpr unsigned kIndex = 0x3, kFresh = 0x4;
  std::array<T, 3> buf_{};
  unsigned write_ = 0, read_ = 1;
  std::atomic<unsigned> middle_{2};
};

template <class T, std::size_t N>
class SpscQueue {
  static_assert((N & (N - 1)) == 0, "capacity must be a power of two");

 public:
  bool push(const T& v) {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    if (h - tail_.load(std::memory_order_acquire) == N) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    buf_[h & (N - 1)] = v;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }
  bool pop(T& out) {
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return false;
    out = buf_[t & (N - 1)];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }
  std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

 private:
  std::array<T, N> buf_{};
  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
  alignas(64) std::atomic<std::uint64_t> dropped_{0};
};

class LatencyHistogram {
 public:
  LatencyHistogram(double bin_us, std::size_t bins) : bin_us_(bin_us), counts_(bins, 0) {}
  void add(double us) {
    const auto b = static_cast<std::size_t>(std::max(0.0, us) / bin_us_);
    ++counts_[std::min(b, counts_.size() - 1)];
    ++n_;
    if (us > max_) max_ = us;
    sum_ += us;
  }
  double percentile(double p) const;
  double max() const { return max_; }
  double mean() const { return n_ ? sum_ / static_cast<double>(n_) : 0.0; }
  std::uint64_t count() const { return n_; }
  const std::vector<std::uint64_t>& counts() const { return counts_; }
  double binWidth() const { return bin_us_; }

 private:
  double bin_us_;
  std::vector<std::uint64_t> counts_;
  std::uint64_t n_ = 0;
  double max_ = 0.0, sum_ = 0.0;
};

class PeriodicTimer {
 public:
  explicit PeriodicTimer(long period_ns);
  /// Sleep until the next deadline; returns wake-up lateness in microseconds.
  double wait();
  /// Periods skipped entirely because a cycle overran by more than a whole period.
  std::uint64_t skippedPeriods() const { return skipped_periods_; }
  static double nowSeconds();

 private:
  long period_ns_;
  timespec next_{};
  std::uint64_t skipped_periods_ = 0;
};

/// Flush subnormal floats to zero (FTZ) and treat subnormal inputs as zero (DAZ) on the calling
/// thread. A robot holding still feeds the controller velocities that decay geometrically through
/// the drive's filter into the subnormal range, where x86 arithmetic is ~100x slower: without
/// this, a Hold tick took about 5x longer than a tick in motion. Returns false where unsupported.
bool enableFlushToZero();
/// Current FTZ/DAZ state, so a caller can restore it.
unsigned floatControlWord();
void setFloatControlWord(unsigned w);

/// Try SCHED_FIFO at `priority` for the calling thread and lock memory. Returns false (and leaves
/// the thread as SCHED_OTHER) when the process lacks CAP_SYS_NICE, as in most containers.
bool trySetRealtime(int priority);
/// Pin the calling thread to one CPU, if there is more than one.
bool pinToCpu(int cpu);

}  // namespace g1rtc::rt
