#include "mononn_engine/tuning/profiler/cupti_profiling_session.h"
#include <cstdio>
using CuptiProfilingSession = mononn_engine::tuning::profiler::CuptiProfilingSession;

int main() {
    mononn_engine::tuning::profiler::launch_simple_cuda_kernel(10000);

    CuptiProfilingSession session({"gpu__time_duration.sum"}, 5);

    int counts[]{10000, 10000 * 10, 10000 * 100, 10000 * 1000, 10000 * 2000};

    auto result = session.profiling_context([&]() {
        for (int count : counts) {
            mononn_engine::tuning::profiler::launch_simple_cuda_kernel(count);
        }
    });

    for (int idx = 0; idx < 5; idx++) {
        printf("%f\n", result.get_time_in_us(idx));
    }
    
    return 0;
}