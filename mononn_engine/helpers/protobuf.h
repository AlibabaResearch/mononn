// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include "google/protobuf/message.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace helpers {
using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;

template <typename T>
std::unique_ptr<T> deep_copy_proto(const google::protobuf::Message* message) {
  std::unique_ptr<T> copied_message = std::make_unique<T>();
  std::string __buf;
  message->SerializeToString(&__buf);
  copied_message->ParseFromString(__buf);
  return std::move(copied_message);
}

std::unique_ptr<GraphSpecification> deep_copy_graph_specification(
    const GraphSpecification* graph_spec);

void save_proto_to_json_file(const google::protobuf::Message* message,
                             std::string file_name);
void load_proto_from_json_file(google::protobuf::Message* message,
                               std::string file_name);

void save_proto_to_binary_file(const google::protobuf::Message* message,
                               std::string file_name);
void load_proto_from_binary_file(google::protobuf::Message* message,
                                 std::string file_name);
}  // namespace helpers
}  // namespace mononn_engine
