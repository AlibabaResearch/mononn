#pragma once

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/all_operators.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace mononn_engine {
namespace parser {
class IRParser {
 public:
  using Graph = mononn_engine::core::graph::Graph;
  using Op = mononn_engine::core::op::Op;

  static std::shared_ptr<Graph> from_text(std::string text);
  static std::shared_ptr<Graph> from_file_stream(std::ifstream& file_stream);

 private:
  static std::shared_ptr<Op> get_node_from_hlo_instruction(
      std::shared_ptr<Graph> graph, xla::HloInstruction* instruction,
      std::unordered_map<std::string, xla::HloComputation*>* fused_computation);

#define JOIN_USING(x, y) x::y

#define USING_OP(op_name, op_code, op_class_code, ...) \
  using op_class_code = JOIN_USING(mononn_engine::core::op, op_class_code);

  OP_TYPE_LIST(USING_OP)

#undef USING_OP
#undef JOIN_USING

#define DECLARE_OP_PARSER(op_name, op_code, op_class_code, ...)              \
  static std::shared_ptr<Op> initialize_##op_code(                           \
      std::shared_ptr<op_class_code> node, xla::HloInstruction* instruction, \
      std::unordered_map<std::string, xla::HloComputation*>*                 \
          fused_computation);

  OP_TYPE_LIST(DECLARE_OP_PARSER)
#undef DECLARE_OP_PARSER

  template <typename T>
  static std::shared_ptr<T> get_sketchy_node(std::shared_ptr<Graph> graph,
                                             xla::HloInstruction* instruction);
};
}  // namespace parser
}  // namespace mononn_engine