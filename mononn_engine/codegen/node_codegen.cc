#include "mononn_engine/codegen/node_codegen.h"

#include <sstream>

#include "mononn_engine/config/config.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/helpers/env_variable.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace codegen {
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using OpType = mononn_engine::core::op::OpType;
using Config = mononn_engine::config::Config;
using Constant = mononn_engine::core::op::Constant;

std::string NodeCodegen::generate(
    std::shared_ptr<const CUDAContext> cuda_context,
    std::shared_ptr<const Op> node) {
  std::string result;

  if (Config::get()->print_hlo_text)
    result += "//" + node->get_hlo_text() + "\n";

  if (node->get_type() == OpType::parameter) {
    return result;
  }

  std::shared_ptr<OpImplBase> op_impl = node->get_implementation();

  if (node->get_type() == OpType::constant) {
    if (node->as<const Constant>()->is_scalar()) {
      return op_impl->generate();
    } else {
      return result;
    }
  }

  bool TF_MONONN_ENABLED =
      mononn_engine::helpers::EnvVar::is_true("TF_MONONN_ENABLED");

  if (node->get_type() == OpType::custom_call ||
      (node->get_type() == OpType::get_tuple_element &&
       !TF_MONONN_ENABLED) ||  // In TF_MONONN path, GET node will be generated
                               // in buffer pointer initialization.
      node->get_type() == OpType::global_sync) {
    return op_impl->generate();
  }

  if (TF_MONONN_ENABLED && node->get_type() == OpType::get_tuple_element) {
    return result;
  }

  LOG(FATAL) << "Found unclustered node " << node->get_name();

  //        Scalar key("idx", Dtype::from_string("int32"));
  //        TensorShape loop_shape = node->get_output_spec(0).get_shape();
  //
  //        Loop loop(
  //                loop_shape,
  //                key,
  //                CUDADefined::threadIdx_x_global,
  ////                Loop::Condition::less_than(key.get_name(),
  /// std::to_string(loop_shape.element_count())),
  //                CUDADefined::threadCnt_x_global);
  //
  //        IndexTracer index_tracer(loop.get_loop_key().get_name());
  //        index_tracer.trace(node);
  //
  //        std::stringstream ss;
  //        ss << loop.begin_loop();
  //
  //        if (node->get_type() == OpType::pad) {
  //            Dtype type = node->get_output_spec(0).get_dtype();
  //            ss << type.to_string() << " " << node->get_name() << ";\n";
  //
  //            std::string index_before_trace = loop.get_loop_key().get_name();
  //            std::vector<std::string> multi_index =
  //            IndexTransform::offset_to_multi_index(node->get_output_spec(0).get_shape(),
  //            index_before_trace); IndexTraceStamp its
  //            {loop.get_loop_key().get_name(), index_tracer.get_index(), "",
  //            "true", index_tracer.get_predictive()};
  //
  //            op_impl->set_traced_index({ its });
  //            op_impl->set_need_generate_with_index(true);
  //
  ////            ss <<
  /// op_impl->as<PadImpl>()->generate_if_statement_begin(multi_index);
  //            ss << op_impl->generate();
  ////            ss << op_impl->as<PadImpl>()->generate_if_statement_end();
  ////            ss << op_impl->as<PadImpl>()->generate_else();
  //
  //        } else {
  //            IndexTraceStamp its {loop.get_loop_key().get_name(),
  //            index_tracer.get_index(), ""};
  //
  //            op_impl->set_traced_index({ its });
  //            op_impl->set_need_generate_with_index(true);
  //            ss << op_impl->generate();
  //        }
  //
  //        ss << Memory::write(
  //                Memory::AccessFlavor::REGULAR,
  //                node->get_output_spec(0).get_dtype(),
  //                node->get_name(),
  //                BufferManager::get_buffer_name(node->get_name()),
  //                loop.get_loop_key().get_name());
  //
  //        ss << loop.end_loop();
  //
  //        return result + ss.str();
}
}  // namespace codegen
}  // namespace mononn_engine