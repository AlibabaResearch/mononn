#pragma once

#include <string>

#include "mononn_engine/core/common/proto_converter.h"
#include "tensorflow/mononn_extra/proto/dim3.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {

struct Dim3 : public mononn_engine::core::common::ProtoConverter<
                  tensorflow::mononn_extra::proto::Dim3> {
  Dim3() {}
  Dim3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
  int x, y, z;

  int XYZ() const;

  std::string to_string() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::Dim3> ConvertToProto()
      const override;
  void ParseFromProto(
      tensorflow::mononn_extra::proto::Dim3 const* dim3) override;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine