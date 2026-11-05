#include "g1rtc/rt.hpp"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <thread>

#if defined(__x86_64__) || defined(__SSE2__)
#include <pmmintrin.h>
#include <xmmintrin.h>
#define G1RTC_HAVE_MXCSR 1
#endif

namespace g1rtc::rt {

double LatencyHistogram::percentile(double p) const {
  if (n_ == 0) return 0.0;
  const auto target = static_cast<std::uint64_t>(p / 100.0 * static_cast<double>(n_));
  std::uint64_t acc = 0;
  for (std::size_t i = 0; i < counts_.size(); ++i) {
    acc += counts_[i];
    if (acc > target) return (static_cast<double>(i) + 1.0) * bin_us_;
  }
  return max_;
}

namespace {
void addNs(timespec& t, long ns) {
  t.tv_nsec += ns;
  while (t.tv_nsec >= 1000000000L) {
    t.tv_nsec -= 1000000000L;
    ++t.tv_sec;
  }
}
double diffUs(const timespec& a, const timespec& b) {
  return static_cast<double>(a.tv_sec - b.tv_sec) * 1e6 + static_cast<double>(a.tv_nsec - b.tv_nsec) * 1e-3;
}
}  // namespace

PeriodicTimer::PeriodicTimer(long period_ns) : period_ns_(period_ns) {
  clock_gettime(CLOCK_MONOTONIC, &next_);
}

double PeriodicTimer::wait() {
  addNs(next_, period_ns_);
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  // If we are already more than a full period late, skip the missed deadlines instead of
  // running a burst of back-to-back cycles to "catch up".
  while (diffUs(now, next_) > static_cast<double>(period_ns_) * 1e-3) {
    addNs(next_, period_ns_);
    ++skipped_periods_;
  }
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_, nullptr);
  clock_gettime(CLOCK_MONOTONIC, &now);
  return diffUs(now, next_);
}

double PeriodicTimer::nowSeconds() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_nsec) * 1e-9;
}

bool enableFlushToZero() {
#ifdef G1RTC_HAVE_MXCSR
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  return true;
#else
  return false;  // on aarch64 the equivalent is the FZ bit of FPCR
#endif
}

unsigned floatControlWord() {
#ifdef G1RTC_HAVE_MXCSR
  return _mm_getcsr();
#else
  return 0;
#endif
}

void setFloatControlWord(unsigned w) {
#ifdef G1RTC_HAVE_MXCSR
  _mm_setcsr(w);
#else
  (void)w;
#endif
}

bool trySetRealtime(int priority) {
  sched_param sp{};
  sp.sched_priority = priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) return false;
  mlockall(MCL_CURRENT | MCL_FUTURE);  // best effort; page faults are the next jitter source
  return true;
}

bool pinToCpu(int cpu) {
  if (std::thread::hardware_concurrency() < 2) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

}  // namespace g1rtc::rt
