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

#include "mononn_engine/helpers/protobuf.h"

#include <fstream>

#include "google/protobuf/util/json_util.h"
#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace helpers {
std::unique_ptr<GraphSpecification> deep_copy_graph_specification(
    const GraphSpecification* graph_spec) {
  std::unique_ptr<GraphSpecification> copied_graph_spec =
      std::make_unique<GraphSpecification>();
  std::string __buf;
  graph_spec->SerializeToString(&__buf);
  copied_graph_spec->ParseFromString(__buf);

  return std::move(copied_graph_spec);
}

void save_proto_to_json_file(const google::protobuf::Message* message,
                             std::string file_name) {
  google::protobuf::util::JsonPrintOptions json_options;
  json_options.add_whitespace = true;
  json_options.always_print_primitive_fields = true;
  json_options.preserve_proto_field_names = true;

  std::string json_string;
  google::protobuf::util::MessageToJsonString(*message, &json_string,
                                              json_options);
  std::ofstream ofs;
  ofs.open(file_name);
  ofs << json_string;
  ofs.close();
}

void load_proto_from_json_file(google::protobuf::Message* message,
                               std::string file_name) {
  google::protobuf::util::JsonParseOptions json_parse_options;
  json_parse_options.ignore_unknown_fields = false;

  std::ifstream ifs;
  ifs.open(file_name);

  if (!ifs.is_open()) LOG(FATAL) << "Cannot open file: " << file_name;

  std::stringstream json_stream;
  json_stream << ifs.rdbuf();

  google::protobuf::util::JsonStringToMessage(json_stream.str(), message,
                                              json_parse_options);

  ifs.close();
}

void save_proto_to_binary_file(const google::protobuf::Message* message,
                               std::string file_name) {
  std::string __buf;
  message->SerializeToString(&__buf);

  std::ofstream ofs;
  ofs.open(file_name);
  ofs << __buf;
  ofs.close();
}

void load_proto_from_binary_file(google::protobuf::Message* message,
                                 std::string file_name) {
  std::ifstream ifs;
  ifs.open(file_name);

  if (!ifs.is_open()) LOG(FATAL) << "Cannot open file: " << file_name;

  message->ParseFromIstream(&ifs);

  ifs.close();
}
}  // namespace helpers
}  // namespace mononn_engine