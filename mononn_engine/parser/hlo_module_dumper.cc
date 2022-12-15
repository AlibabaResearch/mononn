#include <fstream>

#include "tensorflow/core/platform/env.h"
#include "mononn_engine/parser/hlo_module_dumper.h"
#include "mononn_engine/cnpy/cnpy.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/ops.h"
#include "mononn_engine/helpers/helpers.h"
#include "Eigen/Core"

#define COPY_DATA_TO_TENSOR(_type, _dst, _src) \
using RealT = typename tensorflow::Input::Initializer::RealType<_type>::type; \
for (int64_t idx = 0; idx < _dst.NumElements(); ++idx) { \
    _dst.flat<RealT>()(idx) = RealT(_src.data<_type>()[idx]); \
}

namespace mononn_engine {
namespace parser {

    tensorflow::DataType get_node_data_type(tensorflow::GraphDef &graph_def, std::string node_name) {
        for (auto const &node : graph_def.node()) {
            if (node.name() == node_name) {
                return node.attr().at("dtype").type();
            }
        }

        LOG(FATAL) << "Node not found in graph def: " << node_name;
    }

    void HloModuleDumper::dump(std::string binary_path, std::string text_path) {
        if (!text_path.empty()) {
            setenv("XLA_FLAGS", mononn_engine::helpers::string_format("--xla_dump_hlo_as_text --xla_dump_to=%s", text_path.c_str()).c_str(), 1);
        }

        tensorflow::GraphDef graph_def;
        TF_CHECK_OK(tensorflow::ReadBinaryProto(
                tensorflow::Env::Default(),
                this->frozen_pb_file,
                &graph_def));

//        std::ifstream model(this->frozen_pb_file);
//        if (model.fail()) {
//            LOG(FATAL) << "Failed load model from " << this->frozen_pb_file;
//        }
//
//        if (!graph_def.ParseFromIstream(&model)) {
//            LOG(FATAL) << "Failed parse model " <<;
//        }

        tensorflow::SessionOptions session_options;
        session_options.env = tensorflow::Env::Default();
        session_options.config.mutable_gpu_options()->set_allow_growth(true);
        session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(tensorflow::OptimizerOptions_GlobalJitLevel_ON_2);
        session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_cpu_global_jit(true);
        if (this->auto_mixed_precision) {
            session_options.config.mutable_graph_options()->mutable_rewrite_options()->set_auto_mixed_precision(tensorflow::RewriterConfig_Toggle_ON);
        }

        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
        TF_CHECK_OK(session->Create(graph_def));

        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

        int feed_count = (int)this->feeds.size();
        for (int idx = 0; idx < feed_count; ++idx) {
            tensorflow::TensorShape shape;
            std::string feed = this->feeds[idx];
            std::string input_file = this->input_files[idx];
            auto type = get_node_data_type(graph_def, feed);
            auto data = cnpy::npy_load(input_file);

            for (auto const &dim : data.shape) {
                shape.AddDim(dim);
            }

            tensorflow::Tensor input_tensor;
            TF_CHECK_OK(tensorflow::Tensor::BuildTensor(type, shape, &input_tensor));

            if (input_tensor.NumElements() != data.num_vals) {
                LOG(FATAL) << "Num element not match, " << input_tensor.NumElements() << " vs. " << data.num_vals;
            }

            if (type == tensorflow::DT_FLOAT) {
                COPY_DATA_TO_TENSOR(float, input_tensor, data);
            } else if (type == tensorflow::DT_INT32) {
                COPY_DATA_TO_TENSOR(int32_t, input_tensor, data);
            } else if (type == tensorflow::DT_HALF) {
                COPY_DATA_TO_TENSOR(Eigen::half, input_tensor, data);
            } else if (type == tensorflow::DT_DOUBLE) {
                COPY_DATA_TO_TENSOR(double, input_tensor, data);
            } else if (type == tensorflow::DT_UINT32) {
                COPY_DATA_TO_TENSOR(uint32_t , input_tensor, data);
            } else {
                LOG(FATAL) << "Unsupported type " << (int)type << " of operator " << feed;
            }

            inputs.emplace_back(feed, input_tensor);
        }

        std::vector<tensorflow::Tensor> outputs;

        TF_CHECK_OK(session->Run(inputs, this->fetches, {}, &outputs));

        TF_CHECK_OK(session->Close());
    }
}
}