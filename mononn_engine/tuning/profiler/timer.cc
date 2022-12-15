#include "mononn_engine/tuning/profiler/timer.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
    template<typename Resolution>
    struct ResolutionToString;

    template<>
    struct ResolutionToString<std::chrono::seconds> {
        static const std::string str;
    };
    
    template<>
    struct ResolutionToString<std::chrono::milliseconds> {
        static const std::string str;
    };

    template<>
    struct ResolutionToString<std::chrono::microseconds> {
        static const std::string str;
    };

    template<>
    struct ResolutionToString<std::chrono::nanoseconds> {
        static const std::string str;
    };

    const std::string ResolutionToString<std::chrono::seconds>::str = "s";
    const std::string ResolutionToString<std::chrono::milliseconds>::str = "ms";
    const std::string ResolutionToString<std::chrono::microseconds>::str = "us";
    const std::string ResolutionToString<std::chrono::nanoseconds>::str = "ns";

    template<typename Resolution>
    void Timer<Resolution>::start() {
        this->begin_time = std::chrono::high_resolution_clock::now();
    }

    template<typename Resolution>
    void Timer<Resolution>::stop() {
        this->end_time = std::chrono::high_resolution_clock::now();
    }

    template<typename Resolution>
    double Timer<Resolution>::duration() {
        std::chrono::duration<double> d = this->end_time - this->begin_time;
        return std::chrono::duration_cast<Resolution>(d).count();
    }

    template<typename Resolution>
    TimerRAII<Resolution>::TimerRAII(const std::string &_message) : message(_message) {
        this->start();
    }

    template<typename Resolution>
    TimerRAII<Resolution>::~TimerRAII() {
        this->stop();

        LOG(INFO) << message << " " << this->duration() << ResolutionToString<Resolution>::str;
    }

    template class Timer<std::chrono::seconds>;
    template class Timer<std::chrono::milliseconds>;
    template class Timer<std::chrono::microseconds>;
    template class Timer<std::chrono::nanoseconds>;

    template class TimerRAII<std::chrono::seconds>;
    template class TimerRAII<std::chrono::milliseconds>;
    template class TimerRAII<std::chrono::microseconds>;
    template class TimerRAII<std::chrono::nanoseconds>;
}
}
}