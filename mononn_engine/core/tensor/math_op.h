#pragma once

#include <string>
#include <unordered_map>

namespace mononn_engine {
namespace core {
namespace tensor {
class MathOp {
 public:
  MathOp() {}
  MathOp(std::string _op) : op(_op) {}
  MathOp(const char* _op) : MathOp(std::string(_op)) {}
  std::string to_string() const;

  static const MathOp assign;
  static const MathOp plus_assign;
  static const MathOp equal_to;
  static const MathOp not_equal_to;
  static const MathOp greater_equal_than;
  static const MathOp greater_than;
  static const MathOp less_equal_than;
  static const MathOp less_than;

  static MathOp from_string(std::string str);
  static std::unordered_map<std::string, MathOp>* registry();

  bool operator==(MathOp const& rhs) const;

 private:
  std::string op;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine