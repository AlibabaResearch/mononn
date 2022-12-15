#pragma once 

#include <string>
#include "mononn_engine/core/common/proto_converter.h"
#include "tensorflow/mononn_extra/proto/gemm_shape.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    class GemmShape : public mononn_engine::core::common::ProtoConverter<tensorflow::mononn_extra::proto::GemmShape> {
    public:
        GemmShape() {}
        GemmShape(int _m, int _n, int _k) : M(_m), N(_n), K(_k) {}
        
        std::string to_string() const;

        int m() const;
        int n() const;
        int k() const;
        int mn() const;
        int mk() const;
        int nk() const;
        int mnk() const;

        std::unique_ptr<tensorflow::mononn_extra::proto::GemmShape> ConvertToProto() const override;
        void ParseFromProto(tensorflow::mononn_extra::proto::GemmShape const *gemm_shape) override;
    private:
        int M, N, K;
    };
}
}
}
}