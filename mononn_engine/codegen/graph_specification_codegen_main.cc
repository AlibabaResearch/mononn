#include <algorithm>
#include <fstream> 
#include <experimental/filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mononn_engine/codegen/graph_specification_codegen.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"
#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/protobuf.h"

ABSL_FLAG(std::string, graph_spec_file, "", "The input graph specification file");
ABSL_FLAG(std::string, output_dir, "", "The codegen directory");
ABSL_FLAG(std::string, format, "pb", "Input graph spec format");
ABSL_FLAG(std::vector<std::string>, host_codegen_disabled_pass, {}, "Host codegen disabled pass name");
ABSL_FLAG(std::vector<std::string>, optimization_disabled_pass, {}, "");
ABSL_FLAG(std::vector<std::string>, codegen_allow_list, {}, "");

struct Options {
    std::string graph_spec_file;
    std::string output_dir;
    std::string format;
    std::vector<std::string> host_codegen_disabled_pass;
    std::vector<std::string> optimization_disabled_pass;
    std::vector<std::string> codegen_allow_list;
};

using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
using GraphSpecificationCodegen = mononn_engine::codegen::GraphSpecificationCodegen;
using Config = mononn_engine::config::Config;
namespace fs = std::experimental::filesystem;

void override_codegen_allow_list(GraphSpecification *graph_spec, const std::vector<std::string> &codegen_allow_list) {
    std::vector<std::string> all_nodes;

    all_nodes.insert(all_nodes.end(), graph_spec->codegen_allow_list().begin(), graph_spec->codegen_allow_list().end());
    all_nodes.insert(all_nodes.end(), graph_spec->codegen_reject_list().begin(), graph_spec->codegen_reject_list().end());

    graph_spec->mutable_codegen_allow_list()->Clear();
    graph_spec->mutable_codegen_allow_list()->Clear();

    std::vector<std::string> overrided_allow_list;

    for (auto const &node_name : all_nodes) {
        if (std::find(codegen_allow_list.begin(), codegen_allow_list.end(), node_name) != codegen_allow_list.end()) {
            *graph_spec->add_codegen_allow_list() = node_name;
            overrided_allow_list.push_back(node_name);
        } else {
            *graph_spec->add_codegen_reject_list() = node_name;
        }
    }

    std::sort(overrided_allow_list.begin(), overrided_allow_list.end());

    LOG(INFO) << "Codegen allow list override: " << mononn_engine::helpers::join(",", codegen_allow_list);
    LOG(INFO) << "Oerrided: " << mononn_engine::helpers::join(",", overrided_allow_list);

    if (overrided_allow_list != codegen_allow_list) {
        LOG(WARNING) << "Codegen allow list and overrided mismatch.";
    }
}

int Main(Options const &options) {
    Config::get()->output_dir = options.output_dir;
    Config::get()->host_codegen_disabled_pass = options.host_codegen_disabled_pass;
    Config::get()->optimization_disabled_pass = options.optimization_disabled_pass;

    std::unique_ptr<GraphSpecification> graph_spec = std::make_unique<GraphSpecification>();
    if (!fs::exists(options.graph_spec_file)) {
        LOG(FATAL) << "Graph specification file " << options.graph_spec_file << " do not exists";
    }

    if (options.format == "pb") {
        std::ifstream file;
        file.open(options.graph_spec_file);

        graph_spec->ParseFromIstream(&file);
        file.close();
    } else if (options.format == "json") {
        mononn_engine::helpers::load_proto_from_json_file(graph_spec.get(), options.graph_spec_file);
    } else {
        LOG(FATAL) << "Unsupported format " << options.format;
    }

    if (!options.codegen_allow_list.empty()) {
        override_codegen_allow_list(graph_spec.get(), options.codegen_allow_list);
    }

    Config::get()->feeds = std::vector<std::string>(graph_spec->feeds().begin(), graph_spec->feeds().end());
    Config::get()->fetches = std::vector<std::string>(graph_spec->fetches().begin(), graph_spec->fetches().end());
    Config::get()->input_data_files = std::vector<std::string>(graph_spec->input_data_files().begin(), graph_spec->input_data_files().end());

    std::unique_ptr<CUDAProgram> cuda_program = std::move(GraphSpecificationCodegen::generate(graph_spec.get()));

    cuda_program->generate(options.output_dir);

    return 0;
}

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    Options options;

    options.graph_spec_file = absl::GetFlag(FLAGS_graph_spec_file);
    options.output_dir = absl::GetFlag(FLAGS_output_dir);
    options.format = absl::GetFlag(FLAGS_format);
    options.host_codegen_disabled_pass = absl::GetFlag(FLAGS_host_codegen_disabled_pass);
    options.optimization_disabled_pass = absl::GetFlag(FLAGS_optimization_disabled_pass);
    options.codegen_allow_list = absl::GetFlag(FLAGS_codegen_allow_list);
    std::sort(options.codegen_allow_list.begin(), options.codegen_allow_list.end());

    return Main(options);
}