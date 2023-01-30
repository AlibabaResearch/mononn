#include "mononn_engine/core/context/cuda_runtime_context.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace context {
using Dim3 = mononn_engine::core::gpu::Dim3;

std::string CUDARuntimeContext::to_string() const {
  return mononn_engine::helpers::string_format(
      "CUDA runtime context:\n"
      "\tGrid dim: %s\n"
      "\tBlock dim %s\n"
      "\tsmem size %d\n"
      "\tstream %s\n",
      this->grid_dim.to_string().c_str(), this->block_dim.to_string().c_str(),
      this->smem_size, this->stream.c_str());
}

std::unique_ptr<tensorflow::mononn_extra::proto::CUDARuntimeContext>
CUDARuntimeContext::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::CUDARuntimeContext>
      cuda_runtime_context = std::make_unique<
          tensorflow::mononn_extra::proto::CUDARuntimeContext>();

  cuda_runtime_context->set_allocated_grid_dim(
      this->grid_dim.ConvertToProto().release());
  cuda_runtime_context->set_allocated_block_dim(
      this->block_dim.ConvertToProto().release());
  cuda_runtime_context->set_smem_size(this->smem_size);
  cuda_runtime_context->set_stream(this->stream);

  return std::move(cuda_runtime_context);
}

void CUDARuntimeContext::ParseFromProto(
    const tensorflow::mononn_extra::proto::CUDARuntimeContext*
        cuda_runtime_context) {
  this->grid_dim.ParseFromProto(&cuda_runtime_context->grid_dim());
  this->block_dim.ParseFromProto(&cuda_runtime_context->block_dim());
  this->smem_size = cuda_runtime_context->smem_size();
  this->stream = cuda_runtime_context->stream();
}
}  // namespace context
}  // namespace core
}  // namespace mononn_engine