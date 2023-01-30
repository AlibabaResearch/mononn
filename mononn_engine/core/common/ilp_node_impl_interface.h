#pragma once

#include <vector>

#include "mononn_engine/core/common/ilp_node_interface.h"
#include "mononn_engine/core/context/index_trace_stamp.h"

namespace mononn_engine {
namespace core {
namespace common {
class ILPNodeImplInterface : public ILPNodeInterface {
 public:
  using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
  using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;
  std::vector<ConcreteIndexStamp> get_ilp_concrete_index(int ilp_id);
  ConcreteIndexStamp get_ilp_concrete_index(int ilp_id, int index_id);

 protected:
  std::vector<std::vector<ConcreteIndexStamp>> ilp_concrete_index_list;
};
}  // namespace common
}  // namespace core
}  // namespace mononn_engine
