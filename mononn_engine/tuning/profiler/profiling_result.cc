#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/tuning/profiler/profiling_result.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
    void ProfilingResult::verify() const {
        if (!this->codegen_success) {
            LOG(FATAL) << "Codegen failed " << "\n" << this->output
                       << "\n" << "Debug directory:" << this->profiling_directory;
        }

        if (!this->build_success) {
            LOG(FATAL) << "Build failed" << "\n" << this->output
                       << "\n" << "Debug directory:" << this->profiling_directory;
        }

        if (!this->profile_success) {
            LOG(FATAL) << "Profile failed" << "\n" << this->output
                       << "\n" << "Debug directory:" << this->profiling_directory;
        }
    }
}
}
}