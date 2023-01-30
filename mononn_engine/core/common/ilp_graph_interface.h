#pragma once

#include <unordered_map>
#include <vector>

#include "mononn_engine/core/context/index_trace_stamp.h"

namespace mononn_engine {
namespace core {
namespace common {
class ILPGraphInterface {
 public:
  using IndexTraceStamp = mononn_engine::core::context::IndexTraceStamp;

  virtual void set_instruction_parallel_factor(int _ilp_factor) = 0;
  virtual void trace_ilp_index(int ilp_id, const std::string& index,
                               const std::string& node_name,
                               std::string inverse_reduce_dimension = "") = 0;

  int get_instruction_parallel_factor();
  bool is_instruction_parallelized() const;
  //        void set_ilp_traced_index(int ilp_id, std::string node_name,
  //        std::vector<IndexTraceStamp> _traced_index_list);
  //        std::vector<IndexTraceStamp> get_ilp_traced_index(int ilp_id,
  //        std::string node_name) const;
  bool is_node_ilp_traced(std::string node_name, int ilp_id) const;
  void reset_ilp_traced_index();

 protected:
  //        std::unordered_map<std::string,
  //        std::vector<std::vector<IndexTraceStamp>>> ilp_traced_index;
  int ilp_factor = 1;
};
}  // namespace common
}  // namespace core
}  // namespace mononn_engine
