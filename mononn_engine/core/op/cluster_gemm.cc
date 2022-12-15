#include <sstream>

#include "mononn_engine/core/op/cluster_gemm.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "mononn_engine/core/schedule/schedule_factory.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace core {
namespace op {
    using Schedule = mononn_engine::core::schedule::Schedule;
    using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
    using Memory = mononn_engine::core::gpu::Memory;
    using BufferManager = mononn_engine::core::gpu::BufferManager;
    using Op = mononn_engine::core::op::Op;
    using ScheduleFactory = mononn_engine::core::schedule::ScheduleFactory;
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;

    TensorShape ClusterGemm::get_loop_shape() const {
        LOG(FATAL) << "Loop shape undefined";
    }

    Schedule ClusterGemm::construct_schedule(LocalityTier::Tier tier) {
        Schedule schedule;
        schedule.set_locality_tier(LocalityTier::kT3);

        return {schedule};
    }

    void ClusterGemm::setup_codegen() {

    }

    std::string ClusterGemm::generate_cluster_code() const {
        EXPECT_TRUE(this->get_graph()->get_node_list().size() == 1, "GEMM cluster should have only one node");
        std::stringstream ss;

        std::string node_name = this->get_graph()->get_node_list()[0];
        std::shared_ptr<const Op> node = this->get_graph()->get_node(node_name);
        std::shared_ptr<OpImplBase> op_impl = node->get_implementation();

        ss << "\\\\GEMM cluster\n";
        ss << "{\n";
        for (auto const &operand : node->get_operands()) {
            std::string operand_buffer_name = BufferManager::get_buffer_name(operand->get_name());
            ss << mononn_engine::helpers::string_format("void *%s = reinterpret_cast<void *>(%s);\n", operand->get_name().c_str(), operand_buffer_name.c_str());
        }

        // output_buffer
        {
            std::string node_name = node->get_name();
            std::string buffer_name = BufferManager::get_buffer_name(node_name);
            ss << mononn_engine::helpers::string_format("void *%s = reinterpret_cast<void *>(%s);\n", node_name.c_str(), buffer_name.c_str());
        }

        ss << op_impl->generate();

        ss << "}\n";

        return ss.str();
    }

    ClusterType ClusterGemm::get_cluster_type() const {
        return ClusterType::Gemm;
    }
}
}
}