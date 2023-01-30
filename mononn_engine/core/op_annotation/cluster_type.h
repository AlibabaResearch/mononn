#pragma once

#include <string>
namespace mononn_engine {
namespace core {
namespace op_annotation {
class ClusterType {
 public:
  ClusterType() : name("None") {}
  ClusterType(std::string _name) : name(_name) {}
  ClusterType(const char* _name) : ClusterType(std::string(_name)) {}

  static const ClusterType None;
  static const ClusterType Reduce;
  static const ClusterType Elewise;
  static const ClusterType Gemm;
  static const ClusterType Conv;
  static const ClusterType GemmEpilogue;
  static const ClusterType ConvEpilogue;

  std::string to_string() const;

  bool operator==(ClusterType const& rhs) const;
  bool operator!=(ClusterType const& rhs) const;

 private:
  std::string name;
};
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine