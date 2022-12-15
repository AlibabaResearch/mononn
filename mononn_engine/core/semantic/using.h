#pragma once
#include <string>
#include <vector>

namespace mononn_engine {
namespace core {
namespace semantic {
    class Using {
    public:
        Using(std::string _name, std::string _class_name) : name(_name), class_name(_class_name) {}
        Using(std::string _name, const char *_class_name) : Using(_name, std::string(_class_name)) {}
        Using(const char *_name, std::string _class_name) : Using(std::string(_name), _class_name) {}
        Using(const char *_name, const char *_class_name) : Using(std::string(_name), std::string(_class_name)) {}
        
        std::string get_name() const;
        void add_template_arg(std::string arg);
        std::string to_string() const;

        bool is_typename() const;
        void with(std::string _type);
    private:
        std::string name;
        std::string class_name;
        std::vector<std::string> template_args;
        std::vector<std::string> with_type;
    };
}
}
}