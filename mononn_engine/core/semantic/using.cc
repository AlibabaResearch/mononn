#include "mononn_engine/core/semantic/using.h"

#include <sstream>

namespace mononn_engine {
namespace core {
namespace semantic {
void Using::add_template_arg(std::string arg) {
  this->template_args.push_back(arg);
}

std::string Using::get_name() const { return this->name; }

std::string Using::to_string() const {
  std::stringstream ss;

  ss << "using " << this->name << " =";

  if (this->is_typename())
    ss << " "
       << "typename";

  ss << " " << this->class_name;

  if (this->template_args.empty()) {
    ss << ";\n";
    return ss.str();
  }

  ss << "<"
     << "\n";

  for (int idx = 0; idx < (int)this->template_args.size(); ++idx) {
    ss << this->template_args[idx];

    if (idx < (int)this->template_args.size() - 1) {
      ss << ",\n";
    } else {
      ss << ">";
    }
  }

  for (auto const& type : this->with_type) {
    ss << "::" << type;
  }

  ss << ";\n";

  return ss.str();
}

bool Using::is_typename() const { return !this->with_type.empty(); }

void Using::with(std::string _type) { this->with_type.push_back(_type); }
}  // namespace semantic
}  // namespace core
}  // namespace mononn_engine