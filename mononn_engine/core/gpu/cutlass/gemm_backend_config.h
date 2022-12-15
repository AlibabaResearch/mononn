#pragma once

#include <string>
#include <vector>

#include "tensorflow/mononn_extra/proto/gemm_backend_config.pb.h"
#include "mononn_engine/core/common/proto_converter.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    struct GemmBackendConfig : public mononn_engine::core::common::ProtoConverter<tensorflow::mononn_extra::proto::GemmBackendConfig>{
        float alpha_real;
        float alpha_imag;
        float beta;
        int lhs_contracting_dimensions;
        int rhs_contracting_dimensions;
        int batch_size;
        std::vector<int> lhs_batch_dimensions;
        std::vector<int> rhs_batch_dimensions;
        int lhs_stride;
        int rhs_stride;
        int selected_algorithm;

        std::string to_string() const;

        std::unique_ptr<tensorflow::mononn_extra::proto::GemmBackendConfig> ConvertToProto() const override;
        void ParseFromProto(tensorflow::mononn_extra::proto::GemmBackendConfig const *gemm_backend_config) override;
    };
}
}
}
}