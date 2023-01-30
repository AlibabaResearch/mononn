#include "mononn_engine/core/semantic/function_invocation.h"

#include <sstream>

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace semantic {
void FunctionInvocation::add_template_arg(std::string template_arg) {
  this->template_args.push_back(template_arg);
}

void FunctionInvocation::add_arg(std::string arg) { this->args.push_back(arg); }

FunctionInvocation FunctionInvocation::get_ilp_function_invocation(
    int ilp_id) const {
  FunctionInvocation ilp_invocation(this->get_func_name());

  for (auto const& template_arg : this->template_args) {
    ilp_invocation.add_template_arg(template_arg);
  }

  for (auto const& arg : this->args) {
    ilp_invocation.add_arg(
        mononn_engine::helpers::get_node_ilp_name(arg, ilp_id));
  }

  return ilp_invocation;
}

std::string FunctionInvocation::template_args_to_string() const {
  if (this->template_args.empty()) return "";

  std::stringstream ss;

  for (int idx = 0; idx < (int)this->template_args.size(); ++idx) {
    if (idx == 0)
      ss << "<";
    else
      ss << ", ";

    ss << this->template_args[idx];
  }

  ss << ">";
  return ss.str();
}

std::string FunctionInvocation::args_to_string() const {
  if (this->args.empty()) return "()";

  std::stringstream ss;

  for (int idx = 0; idx < (int)this->args.size(); ++idx) {
    if (idx == 0)
      ss << "(";
    else {
      ss << ", ";
      if ((int)this->args.size() > 4) {
        ss << "\n";
      }
    }

    ss << this->args[idx];
  }

  ss << ")";

  return ss.str();
}

std::string FunctionInvocation::get_func_name() const {
  return this->func_name;
}

void FunctionInvocation::set_func_name(std::string _func_name) {
  this->func_name = _func_name;
}

void FunctionInvocation::set_arg(int arg_id, std::string arg_name) {
  this->args[arg_id] = arg_name;
}

std::string FunctionInvocation::to_string() const {
  return mononn_engine::helpers::string_format(
      "%s%s%s", this->get_func_name().c_str(),
      this->template_args_to_string().c_str(), this->args_to_string().c_str());
}
}  // namespace semantic
}  // namespace core
}  // namespace mononn_engine