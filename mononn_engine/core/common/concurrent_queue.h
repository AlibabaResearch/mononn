#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace mononn_engine {
namespace core {
namespace common {
    template<typename T>
    class ConcurrentQueue {
    public:
        void push(const T &item) {
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
}
}
}