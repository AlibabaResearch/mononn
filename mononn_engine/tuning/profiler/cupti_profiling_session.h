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
#include <functional>
#include <unordered_map>
#include <vector>

namespace mononn_engine {
namespace tuning {
namespace profiler {
class CuptiProfilingSession {
 public:
  struct Metrics {
    static const std::string gpu__time_duration_sum;
  };

  class ProfilingResult {
   public:
    double get_time_in_ms(int idx) const {
      return double(this->get_time_in_ns(idx)) / 1e6;
    }

    double get_time_in_us(int idx) const {
      return double(this->get_time_in_ns(idx)) / 1e3;
    }

    double get_time_in_ns(int idx) const {
      return this->metric_range_value.at("gpu__time_duration.sum")
          .at(std::to_string(idx));
    }

    void add_data(const std::string& metricName, const std::string& rangeName,
                  const double& value) {
      if (!this->metric_range_value.count(metricName)) {
        this->metric_range_value[metricName] =
            std::unordered_map<std::string, double>();
      }

      this->metric_range_value[metricName][rangeName] = value;
    }

   private:
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
        metric_range_value;
  };

  CuptiProfilingSession(const std::vector<std::string>& _metricNames,
                        int _numRanges = 1);

  ProfilingResult profiling_context(std::function<void()> kernel_func_wrapper);

  ~CuptiProfilingSession();

 private:
  void cupti_initialize();

  ProfilingResult decode_profiling_result() const;

  std::vector<uint8_t> counterDataImagePrefix;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::vector<uint8_t> counterDataScratchBuffer;
  std::vector<uint8_t> counterAvailabilityImage;
  std::vector<std::string> metricNames;
  int deviceNum = 0;
  int numRanges;
  std::string chipName;
  bool already_profiled = false;
};

void launch_simple_cuda_kernel(int count);
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine