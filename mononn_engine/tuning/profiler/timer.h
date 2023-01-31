// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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