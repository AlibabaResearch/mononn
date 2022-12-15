#include "mononn_engine/core/gpu/limits.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace gpu {
//    std::string Limits::get_limits() {
//        return R"(
//namespace limits {
//    template<typename T>
//    struct numeric_limits;
//
//    template<>
//    struct numeric_limits<float> {
//        __device__ __forceinline__
//        static float max() {
//            return __FLT_MAX__;
//        }
//    };
//
//    template<>
//    struct numeric_limits<int32_t> {
//        __device__ __forceinline__
//        static int32_t max() {
//            return __INT32_MAX__;
//        }
//    };
//
//    template<>
//    struct numeric_limits<half> {
//        __device__ __forceinline__
//        static half max() {
//            return half(65504.0);
//        }
//    };
//}
//)";
//    }

    std::string Limits::get_max_positive(Dtype type) {
        if (type == Dtype::INT8) {
            return "127";
        } else if (type == Dtype::INT16) {
            return "32767";
        } else if (type == Dtype::INT32) {
            return "2147483647";
        } else if (type == Dtype::FLOAT16) {
            return "half(65504.0)";
        } else if (type == Dtype::FLOAT32){
            return "3.40282e+38";
        } else {
            LOG(FATAL) << "Not support type: " << type.to_string();
        }
    }

    std::string Limits::get_min_negative(Limits::Dtype type) {
        if (type == Dtype::INT8) {
            return "-128";
        } else if (type == Dtype::INT16) {
            return "-32768";
        } else if (type == Dtype::INT32) {
            return "-2147483648";
        } else if (type == Dtype::FLOAT16) {
            return "-half(65504.0)";
        } else if (type == Dtype::FLOAT32){
            return "-3.40282e+38";
        } else {
            LOG(FATAL) << "Not support type: " << type.to_string();
        }
    }
}
}
}