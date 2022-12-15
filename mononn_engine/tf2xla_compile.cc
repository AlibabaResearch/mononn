#include "absl/flags/flag.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/client/client_library.h"

ABSL_FLAG(std::string, model_file, "", "Input model protobuf file");

tensorflow::tf2xla::Config get_config() {
    tensorflow::tf2xla::Config config;
    config.add_feed()->mutable_id()->set_node_name("attention_mask");
    config.add_feed()->mutable_id()->set_node_name("input_ids");
    config.add_feed()->mutable_id()->set_node_name("token_type_ids");
    config.add_fetch()->mutable_id()->set_node_name("Identity");
    return config;
}

int main(int argc, char const *argv[])
{
    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(
            tensorflow::Env::Default(),
            "/home/zhuangdonglin.zdl/workspace/models/bert_en_uncased_L-2_H-128_A-2_2/frozen.pb",
            &graph_def));

    tensorflow::tf2xla::Config config = get_config();

    xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
    xla::XlaComputation computation;
    TF_CHECK_OK(ConvertGraphDefToXla(graph_def, config, client, &computation));


    return 0;
}
