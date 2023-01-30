#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"

std::unordered_map<std::string, xla::HloComputation*> fused_computations;
std::string to_neo4j_node_name(std::string name) {
  std::stringstream result;
  for (char c : name) {
    if (c == '.' || c == '-')
      result << "_";
    else
      result << c;
  }

  return result.str();
}

std::string emit_node(std::string node_name, std::string label,
                      std::string properties) {
  std::stringstream cmd;
  cmd << "CREATE(" + node_name;
  if (!label.empty()) {
    cmd << ":" + label;
  }

  cmd << " ";
  cmd << properties;
  cmd << ")\n";

  return cmd.str();
}

std::string emit_edges(std::string node_name,
                       const xla::HloInstruction::InstructionVector& ops) {
  std::stringstream cmd;
  for (const xla::HloInstruction* op : ops) {
    cmd << "CREATE(" + to_neo4j_node_name(op->name()) + ")-[:INPUT]->(" +
               node_name + ")\n";
    // cmd << "CREATE(" + node_name + ")-[:REVERSE_INPUT]->(" +
    // to_neo4j_node_name(op->name()) + ")\n";
  }

  return cmd.str();
}

std::string to_properties_string(
    const std::map<std::string, std::string>& properties) {
  std::stringstream prop;

  bool first = true;
  for (const std::pair<std::string, std::string>& kv : properties) {
    if (first) {
      prop << "{";
      first = false;
    } else
      prop << ", ";

    prop << kv.first << ":";

    prop << "'" << kv.second << "'";
  }

  prop << "}";

  return prop.str();
}

std::string get_node_label(std::string name) {
  if (name.find("constant") == 0) {
    return "constant";
  }

  return to_neo4j_node_name(name.substr(0, name.find(".")));
}

std::string emit_fused_computation(xla::HloInstruction* fused_instruction,
                                   std::string model_name) {
  std::stringstream cmd;

  std::map<std::string, std::string> param_mapping;
  for (size_t op_id = 0; op_id < fused_instruction->operands().size();
       ++op_id) {
    param_mapping["param_" + std::to_string(op_id)] =
        fused_instruction->operands()[op_id]->name();
  }

  for (xla::HloInstruction* instruction :
       fused_computations[fused_instruction->fused_instructions_computation()
                              ->name()]
           ->instructions()) {
    std::string name = instruction->name();
    std::string node_label = get_node_label(name);

    if (name.find("param_", 0) == 0) continue;

    std::string node_name = to_neo4j_node_name(name);
    std::string shape = instruction->shape().ToString(true);
    std::string op_name = instruction->metadata().op_name();
    std::string op_type = instruction->metadata().op_type();

    std::map<std::string, std::string> properties;
    properties["name"] = name;
    properties["shape"] = shape;
    properties["op_name"] = op_name;
    properties["op_type"] = op_type;
    properties["model"] = model_name;
    properties["fused_computation"] =
        fused_instruction->fused_instructions_computation()->name();

    cmd << "CREATE(" + node_name + ":" + node_label + " " +
               to_properties_string(properties)
        << ")\n";

    for (const xla::HloInstruction* op : instruction->operands()) {
      std::string input_op_name = op->name();

      if (input_op_name.find("param_") == 0) {
        input_op_name = input_op_name.substr(0, input_op_name.find("."));
        input_op_name = param_mapping[input_op_name];
      }

      cmd << "CREATE(" + to_neo4j_node_name(input_op_name) + ")-[:INPUT]->(" +
                 node_name + ")\n";
      // cmd << "CREATE(" + node_name + ")-[:REVERSE_INPUT]->(" +
      // to_neo4j_node_name(input_op_name) + ")\n";
    }
  }

  xla::HloInstruction* root_instruction =
      fused_computations[fused_instruction->fused_instructions_computation()
                             ->name()]
          ->root_instruction();
  std::string name = fused_instruction->name();
  std::string node_name = to_neo4j_node_name(name);
  std::string shape = fused_instruction->shape().ToString(true);
  auto fusion_kind = fused_instruction->fusion_kind();

  std::map<std::string, std::string> properties;
  properties["name"] = name;
  properties["shape"] = shape;
  properties["kind"] = std::to_string((int)fusion_kind);
  properties["model"] = model_name;

  cmd << "CREATE(" + node_name + ":ROOT" + " " +
             to_properties_string(properties)
      << ")\n";

  cmd << "CREATE(" + to_neo4j_node_name(root_instruction->name()) +
             ")-[:INPUT]->(" + node_name + ")\n";
  // cmd << "CREATE(" + node_name + ")-[:REVERSE_INPUT]->(" +
  // to_neo4j_node_name(root_instruction->name()) + ")\n";

  return cmd.str();
}

std::string emit_computation(xla::HloInstruction* instruction,
                             std::string model_name) {
  std::string name = instruction->name();
  std::string node_label = get_node_label(name);
  std::string node_name = to_neo4j_node_name(name);
  std::string shape = instruction->shape().ToString(true);
  std::map<std::string, std::string> properties;

  properties["name"] = name;
  properties["shape"] = shape;
  properties["model"] = model_name;

  // constant
  if (instruction->IsConstant()) {
    std::string op_type = instruction->metadata().op_type();
    std::string op_name = instruction->metadata().op_name();
    properties["op_type"] = op_type;
    properties["op_name"] = op_name;

    return emit_node(node_name, node_label, to_properties_string(properties));
  }

  // args
  if (name.find("arg", 0) == 0) {
    std::string op_name = instruction->metadata().op_name();
    properties["op_name"] = op_name;
    return emit_node(node_name, node_label, to_properties_string(properties));
  }

  if (name.find("custom-call", 0) == 0) {
    std::string custom_call_target = instruction->custom_call_target();
    std::string op_type = instruction->metadata().op_type();
    std::string op_name = instruction->metadata().op_name();

    properties["custom_call_target"] = custom_call_target;
    properties["op_type"] = op_type;
    properties["op_name"] = op_name;

    return emit_node(node_name, node_label, to_properties_string(properties)) +
           emit_edges(node_name, instruction->operands());
  }

  if (name.find("tuple", 0) == 0) {
    std::string op_name = instruction->metadata().op_name();
    properties["op_name"] = op_name;

    return emit_node(node_name, node_label, to_properties_string(properties)) +
           emit_edges(node_name, instruction->operands());
  }

  return emit_node(node_name, node_label, to_properties_string(properties)) +
         emit_edges(node_name, instruction->operands());
}

int main(int argc, char const* argv[]) {
  std::stringstream hlo_text_stream;
  std::string model_name = argv[1];
  std::ifstream in(argv[2]);

  hlo_text_stream << in.rdbuf();

  tensorflow::StatusOr<std::unique_ptr<xla::HloModule>> ret =
      xla::ParseAndReturnUnverifiedModule(
          absl::string_view(hlo_text_stream.str()));

  xla::HloModule* mod = ret->get();
  xla::HloComputation* entry_computation = ret->get()->entry_computation();

  {
    std::set<std::string> se;
    for (xla::HloInstruction* instruction : entry_computation->instructions()) {
      std::string inst_name = instruction->name();
      std::string inst_type;
      if (inst_name.find("constant") == 0) {
        inst_type = inst_name.substr(0, inst_name.find("_"));
      } else {
        inst_type = inst_name.substr(0, inst_name.find("."));
      }

      se.insert(inst_type);
    }

    for (std::string s : se) {
      std::cout << s << std::endl;
    }
  }
  return 0;

  for (xla::HloComputation* computation : mod->computations()) {
    fused_computations[computation->name()] = computation;
  }

  std::stringstream cypher_stream;
  for (xla::HloInstruction* instruction : entry_computation->instructions()) {
    if (instruction->IsFusedComputation()) {
      std::cerr << "Unexpected fused instruction: " << instruction->name()
                << std::endl;
      cypher_stream << emit_fused_computation(instruction, model_name);
    } else {
      cypher_stream << emit_computation(instruction, model_name);
    }
  }

  cypher_stream << ";";

  std::ofstream out(argv[3]);
  out << cypher_stream.str() << std::endl;

  return 0;
}