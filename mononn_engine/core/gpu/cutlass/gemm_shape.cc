#include "mononn_engine/core/gpu/cutlass/gemm_shape.h"
#include "mononn_engine/helpers/string_helpers.h"
namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    std::string GemmShape::to_string() const {
        return mononn_engine::helpers::string_format("cutlass::gemm::GemmShape<%d, %d, %d>", this->M, this->N, this->K);
    }

    int GemmShape::m() const {
        return this->M;
    }

    int GemmShape::n() const {
        return this->N;
    }
    
    int GemmShape::k() const {
        return this->K;
    }
    
    int GemmShape::mn() const {
        return this->M * this->N;
    }
    
    int GemmShape::mk() const {
        return this->M * this->K;
    }
    
    int GemmShape::nk() const {
        return this->N * this->K;
    }
    
    int GemmShape::mnk() const {
        return this->M * this->N * this->K;
    }

    std::unique_ptr<tensorflow::mononn_extra::proto::GemmShape> GemmShape::ConvertToProto() const {
        std::unique_ptr<tensorflow::mononn_extra::proto::GemmShape> gemm_shape = std::make_unique<tensorflow::mononn_extra::proto::GemmShape>();

        gemm_shape->set_m(this->M);
        gemm_shape->set_n(this->N);
        gemm_shape->set_k(this->K);

        return std::move(gemm_shape);
    }

    void GemmShape::ParseFromProto(const tensorflow::mononn_extra::proto::GemmShape *gemm_shape) {
        this->M = gemm_shape->m();
        this->N = gemm_shape->n();
        this->K = gemm_shape->k();
    }
}
}
}
}