#include "mononn_engine/core/op_annotation/op_attribute.h"

namespace mononn_engine {
namespace core {
namespace op_annotation {
std::string const OpAttribute::initial_cluster_tag = "initial_cluster_tag";
std::string const OpAttribute::sub_cluster_tag = "sub_cluster_tag";
std::string const OpAttribute::sub_cluster_type = "sub_cluster_type";
std::string const OpAttribute::on_chip_transfer_from_node =
    "on_chip_transfer_from_node";
std::string const OpAttribute::initial_cluster_type = "initial_cluster_type";
std::string const OpAttribute::intra_op_reschedule_factor =
    "intra_op_reschedule_factor";
std::string const OpAttribute::prefetch_predicate = "prefetch_predicate";
std::string const OpAttribute::is_parameter_streaming_access =
    "is_parameter_streaming_access";
std::string const OpAttribute::is_parameter_temporal_access =
    "is_parameter_temporal_access";
std::string const OpAttribute::is_broadcast_semi_vectorized =
    "is_broadcast_semi_vectorized";
std::string const OpAttribute::is_node_stop_vectorized =
    "is_node_stop_vectorized";
std::string const OpAttribute::is_parameter_async_prefetched =
    "is_parameter_async_prefetched";
std::string const OpAttribute::is_parameter_cache_prefetched =
    "is_parameter_cache_prefetched";
std::string const OpAttribute::async_pipeline_total_stage_count =
    "async_pipeline_total_stage_count";
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine