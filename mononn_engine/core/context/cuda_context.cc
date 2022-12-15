#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace context {
    CUDAContext CUDAContext::get_cuda_context(Dim3 _grid_dim, Dim3 _block_dim, std::string _stream) {
        CUDADeviceContext cuda_device_context = CUDADeviceContext::get_cuda_device_context();
        int block_per_sm = (_grid_dim.XYZ() + cuda_device_context.sm_count - 1) / cuda_device_context.sm_count;

        CUDARuntimeContext cuda_runtime_context = CUDARuntimeContext(
            _grid_dim,
            _block_dim,
            get_max_smem_size_per_block(block_per_sm, _block_dim.XYZ(), cuda_device_context.max_configurable_smem_size),
            _stream
        );

        return CUDAContext(cuda_runtime_context, cuda_device_context);
    }

    CUDAContext CUDAContext::get_cuda_context(
            Dim3 _grid_dim,
            Dim3 _block_dim,
            std::string _stream,
            CUDADeviceContext _cuda_device_context) {
        int block_per_sm = (_grid_dim.XYZ() + _cuda_device_context.sm_count - 1) / _cuda_device_context.sm_count;

        CUDARuntimeContext cuda_runtime_context = CUDARuntimeContext(
                _grid_dim,
                _block_dim,
                get_max_smem_size_per_block(block_per_sm, _block_dim.XYZ(), _cuda_device_context.max_configurable_smem_size),
                _stream
        );

        return CUDAContext(cuda_runtime_context, _cuda_device_context);
    }

    int CUDAContext::get_block_per_sm() const {
        return (this->cuda_runtime_context.grid_dim.XYZ() + this->cuda_device_context.sm_count - 1) / this->cuda_device_context.sm_count;
    }

    std::unique_ptr<tensorflow::mononn_extra::proto::CUDAContext> CUDAContext::ConvertToProto() const {
        std::unique_ptr<tensorflow::mononn_extra::proto::CUDAContext> cuda_context = std::make_unique<tensorflow::mononn_extra::proto::CUDAContext>();

        cuda_context->set_allocated_cuda_runtime_context(this->cuda_runtime_context.ConvertToProto().release());
        cuda_context->set_allocated_cuda_device_context(this->cuda_device_context.ConvertToProto().release());

        return std::move(cuda_context);
    }

    void CUDAContext::ParseFromProto(const tensorflow::mononn_extra::proto::CUDAContext *cuda_context) {
        this->cuda_runtime_context.ParseFromProto(&cuda_context->cuda_runtime_context());
        this->cuda_device_context.ParseFromProto(&cuda_context->cuda_device_context());
    }
}
}
}