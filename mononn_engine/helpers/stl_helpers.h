#pragma once
#include <vector>
#include <memory>
#include <functional>

namespace mononn_engine {
namespace helpers {
    template<typename T1, typename T2, typename TRes>
    std::vector<TRes> cartesian_join(const std::vector<T1> &vec1, const std::vector<T2> &vec2, std::function<TRes(const T1&, const T2&)> func) {
        std::vector<TRes> res;

        for (auto const &elem1 : vec1) {
            for (auto const &elem2 : vec2) {
                res.push_back(func(elem1, elem2));
            }
        }

        return res;
    }

    template<typename T>
    std::vector<T> vector_concat(std::vector<T> const &vec1, std::vector<T> const &vec2) {
        std::vector<T> result = vec1;
        result.insert(result.end(), vec2.begin(), vec2.end());
        return result;
    }

    template<typename T, typename ... TArgs>
    std::vector<T> vector_concat(std::vector<T> const &vec1, std::vector<T> const &vec2, TArgs ... additional_vec) {
        return vector_concat(vector_concat(vec1, vec2), additional_vec...);
    }

    template<typename Derived, typename Base, typename Del>
    std::unique_ptr<Derived, Del>
    static_unique_ptr_cast(std::unique_ptr<Base, Del> && p)
    {
        auto d = static_cast<Derived *>(p.release());
        return std::unique_ptr<Derived, Del>(d);
    }
}
}
