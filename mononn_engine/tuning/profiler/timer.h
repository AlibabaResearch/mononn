#pragma once
#include <chrono>
#include <string>

namespace mononn_engine {
namespace tuning {
namespace profiler {

template <typename Resolution>
class Timer {
 public:
  using TimePoint = std::chrono::time_point<std::chrono::system_clock,
                                            std::chrono::nanoseconds>;

  void start();
  void stop();

  double duration();

 private:
  TimePoint begin_time;
  TimePoint end_time;
};

template <typename Resolution>
class TimerRAII : public Timer<Resolution> {
 public:
  TimerRAII(const std::string& _message);

  ~TimerRAII();

 private:
  std::string message;
};
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine