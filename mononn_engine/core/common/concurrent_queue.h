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

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>

namespace mononn_engine {
namespace core {
namespace common {
template <typename T>
class ConcurrentQueue {
 public:
  void push(const T& item) {
    std::unique_lock<std::mutex> lock(this->mtx);
    this->queue.push(item);
    this->cv.notify_one();
  }

  T wait() {
    std::unique_lock<std::mutex> lock(this->mtx);

    this->cv.wait(lock, [this] { return !this->queue.empty(); });

    T ret = std::move(this->queue.front());
    this->queue.pop();

    return std::move(ret);
  }

  size_t size() {
    std::unique_lock<std::mutex> lock(this->mtx);
    return this->queue.size();
  }

 private:
  std::queue<T> queue;
  std::mutex mtx;
  std::condition_variable cv;
};
}  // namespace common
}  // namespace core
}  // namespace mononn_engine