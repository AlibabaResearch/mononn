#pragma once

#include <string>
#include <utility>

#include "mononn_engine/core/gpu/dim3.h"
#include "tensorflow/mononn_extra/proto/cuda_runtime_context.pb.h"

namespace mononn_engine {
namespace core {
namespace context {
using Dim3 = mononn_engine::core::gpu::Dim3;
struct CUDARuntimeContext
    : public mononn_engine::core::common::ProtoConverter<
          tensorflow::mononn_extra::proto::CUDARuntimeContext> {
  CUDARuntimeContext() {}
  CUDARuntimeContext(Dim3 _grid_dim, Dim3 _block_dim, int _smem_size,
                     std::string _stream)
      : grid_dim(_grid_dim),
        block_dim(_block_dim),
        smem_size(_smem_size),
        stream(std::move(_stream)) {}

  Dim3 grid_dim;
  Dim3 block_dim;
  int smem_size;  // smem size in bytes
  std::string stream;

  std::string to_string() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::CUDARuntimeContext>
  ConvertToProto() const override;
  void ParseFromProto(tensorflow::mononn_extra::proto::CUDARuntimeContext const*
                          cuda_runtime_context);
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine