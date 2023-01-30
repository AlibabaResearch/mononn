#include "mononn_engine/tuning/profiler/thread_pool.h"

#include "mononn_engine/helpers/env_variable.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
using EnvVar = mononn_engine::helpers::EnvVar;

void ThreadPool::start_function() {
  // Disable worker thread logging
  bool TF_MONONN_ENABLED = EnvVar::is_true("TF_MONONN_ENABLED");
  bool TF_MONONN_ENABLE_WORKER_THRAED_LOGGING =
      EnvVar::is_true("TF_MONONN_ENABLE_WORKER_THRAED_LOGGING");

  if (TF_MONONN_ENABLED && !TF_MONONN_ENABLE_WORKER_THRAED_LOGGING) {
    ::tensorflow::internal::Flags::__TF_LOG_MULTI_THREAD_GUARD__ = false;
  }

  while (true) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(this->queue_mutex);

      if (!this->stop && this->tasks.empty()) {
        this->condition.wait(
            lock, [this] { return this->stop || !this->tasks.empty(); });
      }

      if (this->stop && this->tasks.empty()) {
        return;
      }

      task = std::move(this->tasks.front());
      this->tasks.pop();
    }

    task();
  }
}

ThreadPool::ThreadPool(size_t threads) : stop(false) {
  for (size_t i = 0; i < threads; ++i) {
    workers.emplace_back(&ThreadPool::start_function, this);
    // workers.emplace_back([this]() -> void {
    //     while (true) {
    //         std::function<void()> task;

    //         {
    //             std::unique_lock<std::mutex> lock(this->queue_mutex);

    //             if (!this->stop && this->tasks.empty()) {
    //                 this->condition.wait(lock, [this]{ return this->stop ||
    //                 !this->tasks.empty(); });
    //             }

    //             if(this->stop && this->tasks.empty()) {
    //                 return;
    //             }

    //             task = std::move(this->tasks.front());
    //             this->tasks.pop();
    //         }

    //         task();
    //     }
    // });
  }
}

size_t ThreadPool::num_remaining_tasks() {
  std::unique_lock<std::mutex> lock(this->queue_mutex);
  return this->tasks.size();
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }

  condition.notify_all();
  for (std::thread& worker : workers) worker.join();
}
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine