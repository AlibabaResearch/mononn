#pragma once

#include <memory>
#include "google/protobuf/message.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace helpers {
    using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;

    template<typename T>
    std::unique_ptr<T> deep_copy_proto(const google::protobuf::Message *message) {
        std::unique_ptr<T> copied_message = std::make_unique<T>();
        std::string __buf;
        message->SerializeToString(&__buf);
        copied_message->ParseFromString(__buf);
        return std::move(copied_message);
    }

    std::unique_ptr<GraphSpecification> deep_copy_graph_specification(const GraphSpecification *graph_spec);

    void save_proto_to_json_file(const google::protobuf::Message *message, std::string file_name);
    void load_proto_from_json_file(google::protobuf::Message *message, std::string file_name);

    void save_proto_to_binary_file(const google::protobuf::Message *message, std::string file_name);
    void load_proto_from_binary_file(google::protobuf::Message *message, std::string file_name);
}
}

