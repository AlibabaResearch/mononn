#pragma once
#include <vector>
#include <functional>

namespace mononn_engine {
namespace helpers {
    class Transform {
    public:
//        template<typename Tin, typename Tout>
//        static std::vector<Tout> map(std::vector<Tin> const &in, std::function<Tout(Tin const &element)> fn) {
//            std::vector<Tout> result;
//
//            for (auto const &e : in) {
//                result.push_back(fn(e));
//            }
//
//            return result;
//        }

        template<typename Tin, typename Tout>
        static std::vector<Tout> map(std::vector<Tin> &in, std::function<Tout(Tin &element)> fn) {
            std::vector<Tout> result;

            for (auto &e : in) {
                result.push_back(fn(e));
            }

            return result;
        }

    private:
    };
}
}

