#include "mononn_engine/core/context/index_transform_lazy.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace core {
namespace context {
    std::function<std::vector<std::string>(std::string)> IndexTransformLazy::offset_to_multi_index_lazy(TensorShape tensor_shape) {
        std::function<std::vector<std::string>(std::string)> ret = [=](std::string offset) -> std::vector<std::string> {
            std::vector<std::string> multi_index;
            multi_index.resize(tensor_shape.rank());

            for (int idx = 0; idx < (int)multi_index.size(); ++idx) {
                std::string mod = mononn_engine::helpers::join(" * ", tensor_shape.slice_dim(idx, -1).get_shape());
                std::string div = mononn_engine::helpers::join(" * ", tensor_shape.slice_dim(idx + 1, -1).get_shape());

                if (idx == 0) {
                    multi_index[idx] = mononn_engine::helpers::string_format("((%s) / (%s))", offset.c_str(), div.c_str());
                } else if (idx == (int)multi_index.size() - 1) {
                    multi_index[idx] = mononn_engine::helpers::string_format("((%s) %% (%s))", offset.c_str(), mod.c_str());
                } else {
                    multi_index[idx] = mononn_engine::helpers::string_format("(((%s) %% (%s)) / (%s))", offset.c_str(), mod.c_str(), div.c_str());
                }
            }

            return multi_index;
        };

        return ret;
    }

    std::function<std::string(std::vector<std::string>)> IndexTransformLazy::multi_index_to_offset_lazy(TensorShape tensor_shape) {
        std::function<std::string(std::vector<std::string>)> ret = [=](std::vector<std::string> multi_index) -> std::string {
            std::string offset;
            offset = multi_index[0];
            for (int idx = 1; idx < (int)multi_index.size(); ++idx) {
                offset = mononn_engine::helpers::string_format("(%s * %s + %s)",
                                                          offset.c_str(),
                                                          std::to_string(tensor_shape.get_shape(idx)).c_str(),
                                                          multi_index[idx].c_str());
            }

            return offset;
        };

        return ret;
    }
}
}
}