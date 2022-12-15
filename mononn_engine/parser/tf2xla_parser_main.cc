#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "mononn_engine/helpers/helpers.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"

ABSL_FLAG(std::string, input_file, "/home/zhuangdonglin.zdl/workspace/models/experiments/bert_tiny/frozen.pb", "The input frozen pb file");
ABSL_FLAG(std::vector<std::string>, feeds, std::vector<std::string>({"token_type_ids", "input_ids", "attention_mask"}), "Input nodes");
ABSL_FLAG(std::vector<std::string>, fetches, std::vector<std::string>({"Identity"}), "Output nodes");

struct Options {
   std::string input_file;
   std::vector<std::string> feeds;
   std::vector<std::string> fetches;
};

tensorflow::tf2xla::Config get_config(const Options &options) {
   tensorflow::tf2xla::Config config;
   for (auto const &feed : options.feeds) {
       config.add_feed()->mutable_id()->set_node_name(feed);
   }

   for (auto const &fetch : options.fetches) {
       config.add_fetch()->mutable_id()->set_node_name(fetch);
   }

   return config;
}

int Main(const Options &options) {

    tensorflow::GraphDef graph_def;

    TF_CHECK_OK(tensorflow::ReadBinaryProto(
           tensorflow::Env::Default(),
           options.input_file,
           &graph_def));
//    SessionFactory::Register("DIRECT_SESSION", new DirectSessionFactory());
//    std::unique_ptr<tensorflow::Session> session;
//    {
//        tensorflow::Session *session_tmp;
//        auto session_options = tensorflow::SessionOptions();
//        auto s = tensorflow::NewSession(session_options, (&session_tmp));
//        session.reset(session_tmp);
//    }
    tensorflow::SessionOptions session_options;
    session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(tensorflow::OptimizerOptions_GlobalJitLevel_ON_2);
    session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_cpu_global_jit(true);
    session_options.config.mutable_graph_options()->mutable_rewrite_options()->set_auto_mixed_precision(tensorflow::RewriterConfig_Toggle_ON);
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
    TF_CHECK_OK(session->Create(graph_def));

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

    for (auto feed : options.feeds) {
        tensorflow::TensorShape shape;
        shape.AddDim(1);
        shape.AddDim(128);
        tensorflow::Input::Initializer input(1, shape);
        inputs.emplace_back(feed, input.tensor);
    }

    std::vector<tensorflow::Tensor> outputs;

    for (int i = 0; i < 2; ++i)
    TF_CHECK_OK(session->Run(inputs, options.fetches, {}, &outputs));

    session->Close();

    return 0;
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    Options options;

    options.input_file = absl::GetFlag(FLAGS_input_file);

    for (auto const &feed : absl::GetFlag(FLAGS_feeds)) {
       options.feeds.push_back(feed);
    }

    for (auto const &fetch : absl::GetFlag(FLAGS_fetches)) {
       options.fetches.push_back(fetch);
    }

    if (options.input_file.empty()) LOG(FATAL) << "Please specify model input file";
    if (options.feeds.empty()) LOG(FATAL) << "Please specify model graph input node";
    if (options.fetches.empty()) LOG(FATAL) << "Please specify model graph output node";

    return Main(options);
}