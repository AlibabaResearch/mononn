// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/core/schedule/vectorizer.h"

#include <algorithm>
#include <stack>
#include <unordered_set>

#include "mononn_engine/core/op/broadcast.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/slice.h"
#include "mononn_engine/core/op/transpose.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace schedule {
using Op = mononn_engine::core::op::Op;
using Slice = mononn_engine::core::op::Slice;
using Transpose = mononn_engine::core::op::Transpose;
using Broadcast = mononn_engine::core::op::Broadcast;
using OpType = mononn_engine::core::op::OpType;
using Dtype = mononn_engine::core::tensor::Dtype;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

void Vectorizer::vectorize(std::shared_ptr<ClusterOp> cluster_op) {
  std::unordered_set<std::string> node_can_vectorize;
  for (auto const& node_name : cluster_op->get_graph()->get_node_list()) {
    node_can_vectorize.insert(node_name);
  }

  std::stack<std::string> trace;
  cluster_op->get_graph()->post_order_visit_all_nodes(
      [&](std::shared_ptr<Op> op) -> void {
        std::string node_name = op->get_name();
        if (op->get_type() == OpType::reduce ||
            op->get_type() == OpType::broadcast)
          trace.push(node_name);

        if (!trace.empty() &&
            cluster_op->get_graph()->get_node(trace.top())->get_type() ==
                OpType::reduce) {
          if (node_can_vectorize.find(node_name) != node_can_vectorize.end()) {
            node_can_vectorize.erase(node_name);
          }
        }
      },
      [&](std::shared_ptr<Op> op) -> void {
        std::string node_name = op->get_name();

        if (op->get_type() == OpType::reduce ||
            op->get_type() == OpType::broadcast) {
          EXPECT_TRUE(trace.top() == node_name, "Node name mismatch");

          trace.pop();
        }
      });

  cluster_op->get_graph()->reverse_post_order_visit_all_nodes(
      [&](std::shared_ptr<Op> op) -> void {
        std::string node_name = op->get_name();
        if (op->get_type() == OpType::broadcast) {
          std::shared_ptr<Broadcast> broadcast_op =
              std::static_pointer_cast<Broadcast>(op);

          TensorSpec output_shape = broadcast_op->get_output_spec(0);
          std::vector<int> broadcast_dims = broadcast_op->get_dimensions();

          // broadcast cannot be vectorized
          if (std::find(broadcast_dims.begin(), broadcast_dims.end(),
                        output_shape.rank() - 1) == broadcast_dims.end()) {
            if (node_can_vectorize.find(op->get_operand(0)->get_name()) !=
                node_can_vectorize.end()) {
              node_can_vectorize.erase(op->get_operand(0)->get_name());
            }
          }
        }

        if (op->get_type() == OpType::gather) {
          if (node_can_vectorize.find(op->get_operand(1)->get_name()) !=
              node_can_vectorize.end()) {
            node_can_vectorize.erase(op->get_operand(1)->get_name());
          }
        }

        if (op->get_type() != OpType::reduce &&
            node_can_vectorize.find(node_name) == node_can_vectorize.end()) {
          for (auto const& operand : op->get_operands()) {
            if (node_can_vectorize.find(operand->get_name()) !=
                node_can_vectorize.end()) {
              node_can_vectorize.erase(operand->get_name());
            }
          }
        }
      },
      [&](std::shared_ptr<Op> op) -> void {
        // empty
      });

  int vector_len = 16;  // 16 bytes per instruction
  cluster_op->get_graph()->wave_front_order([&](std::shared_ptr<Op> op)
                                                -> void {
    std::string node_name = op->get_name();

    if (node_can_vectorize.find(node_name) == node_can_vectorize.end()) return;

    std::vector<TensorSpec> output_specs = op->get_output_specs();
    EXPECT_TRUE(output_specs.size() == 1,
                "Node " + node_name + " have more than one outputs");
    Dtype type = output_specs[0].get_dtype();
    EXPECT_TRUE(128 % type.size_in_bits() == 0,
                "Node " + node_name + "have output type size in bits " +
                    std::to_string(type.size_in_bits()));
    vector_len = std::min(vector_len, 128 / type.size_in_bits());

    while (output_specs[0].get_shape(-1) % vector_len != 0) vector_len >>= 1;

    if (op->get_type() == OpType::slice) {
      std::shared_ptr<Slice> slice_op = std::static_pointer_cast<Slice>(op);
      std::vector<int> slice_starts = slice_op->get_slice_starts();
      std::vector<int> slice_strides = slice_op->get_slice_strides();
      std::vector<int> slice_limits = slice_op->get_slice_limits();

      int start_offset_in_bits = slice_starts.back() * type.size_in_bits();

      for (int candidate_vec_len = 16;; candidate_vec_len >>= 1) {
        if (candidate_vec_len == 0) {
          LOG(FATAL) << "Cannot vectorize node " << node_name;
        }

        if (start_offset_in_bits % (candidate_vec_len * type.size_in_bits()) ==
            0) {
          vector_len = std::min(vector_len, candidate_vec_len);
          break;
        }
      }

      if (slice_strides.back() != 1) {
        LOG(WARNING) << "Slice stride of " + node_name + " is not one";
        vector_len = 1;
      }

      int end_offset_in_bits = slice_limits.back() * type.size_in_bits();

      for (int candidate_vec_len = 16;; candidate_vec_len >>= 1) {
        if (candidate_vec_len == 0) {
          LOG(FATAL) << "Cannot vectorize node " << node_name;
        }

        if (end_offset_in_bits % (candidate_vec_len * type.size_in_bits()) ==
            0) {
          vector_len = std::min(vector_len, candidate_vec_len);
          break;
        }
      }
    }
  });

  cluster_op->get_graph()->wave_front_order([&](std::shared_ptr<Op> op)
                                                -> void {
    std::vector<TensorSpec> output_specs = op->get_output_specs();

    int vector_len_to_use = vector_len;
    if (node_can_vectorize.find(op->get_name()) == node_can_vectorize.end()) {
      vector_len_to_use = 1;
    }

    for (auto& spec : output_specs) {
      spec = TensorSpec(
          spec.get_dtype().get_primitive_type().vectorize(vector_len_to_use),
          spec.get_shape(), spec.get_layout());
    }
  });

  // vectorize loop schedule
  cluster_op->set_schedule(cluster_op->get_schedule().vectorize(vector_len));
}
}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine