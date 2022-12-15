#include "mononn_engine/core/context/index_tracer_lazy.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "mononn_engine/core/context/index_transform_lazy.h"

namespace mononn_engine {
namespace core {
namespace context {
    using OpType = mononn_engine::core::op::OpType;
    using Constant = mononn_engine::core::op::Constant;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using IndexTransformLazy = mononn_engine::core::context::IndexTransformLazy;

    void IndexTracerLazy::trace(std::shared_ptr<const Op> op) {
        if (op->get_type() == OpType::broadcast) {
            this->trace_broadcast(std::static_pointer_cast<const Broadcast>(op));
        } else if (op->get_type() == OpType::gather) {
            LOG(FATAL) << "Gather operator should be traced explicitly for operand or indices";
        } else if (op->get_type() == OpType::pad) {
            this->trace_pad(std::static_pointer_cast<const Pad>(op));
        } else if (op->get_type() == OpType::transpose) {
            this->trace_transpose(std::static_pointer_cast<const Transpose>(op));
        } else if (op->get_type() == OpType::slice) {
            this->trace_slice(std::static_pointer_cast<const Slice>(op));
        } else if (op->get_type() == OpType::reduce) {
            LOG(FATAL) << "Reduce operator should be traced explicitly";
        }
    }

    std::string IndexTracerLazy::get_index(std::string input_index) const {
        return this->index(input_index);
    }

    void IndexTracerLazy::trace_broadcast(std::shared_ptr<const Broadcast> op) {
        static int __already_logged = false;
        if (!__already_logged) {
            __already_logged = true;
            LOG(INFO) << "Trace broadcast can perform additional optimizations for lesser index calculation";
        }

        if (op->get_operand(0)->get_type() == OpType::constant &&
            std::static_pointer_cast<Constant>(op->get_operand(0))->is_scalar()) {
            return;
        }

        std::function<std::string(std::string)> trace_index = [=](std::string input_index) -> std::string {
            std::vector<int> broadcast_dims = op->get_dimensions();

            TensorShape shape = op->get_output_spec(0).get_shape();
            TensorShape shape_before_broadcast = op->get_operand(0)->get_output_spec(0).get_shape();

            std::function<std::vector<std::string>(std::string)> per_dim_index = IndexTransformLazy::offset_to_multi_index_lazy(shape);

            EXPECT_TRUE(broadcast_dims.size() == shape_before_broadcast.rank(), "Rank not match");

            std::vector<std::string> per_dim_index_before_broadcast;
            for (auto const &dim : broadcast_dims) {
                per_dim_index_before_broadcast.push_back(per_dim_index(input_index)[dim]);
            }

            return IndexTransformLazy::multi_index_to_offset_lazy(shape_before_broadcast)(per_dim_index_before_broadcast);
        };

        this->index = [=](std::string input_index) -> std::string {
            return trace_index(this->index(input_index));
        };
    }

    void IndexTracerLazy::trace_gather_operand(std::shared_ptr<const Gather> op) {
        int index_vector_dim = op->get_index_vector_dim();
        std::vector<int> offset_dims = op->get_offset_dims();
        std::vector<int> slice_sizes = op->get_slice_sizes();
        std::vector<int> collapsed_slice_dims = op->get_collapsed_slice_dims();
        std::vector<int> start_index_map = op->get_start_index_map();

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_gather = op->get_operand(0)->get_output_spec(0).get_shape();
        TensorShape start_indices_shape = op->get_operand(1)->get_output_spec(0).get_shape();

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        std::vector<int> batch_dims;
        for (int idx = 0; idx < shape.rank(); ++idx) {
            if (std::find(offset_dims.begin(), offset_dims.end(), idx) == offset_dims.end()) {
                batch_dims.push_back(idx);
            }
        }

        std::vector<std::string> batch_dim_index, offset_dim_index;

        for (auto const &dim : batch_dims) {
            batch_dim_index.push_back(per_dim_index[dim]);
        }

        for (auto const &dim : offset_dims) {
            offset_dim_index.push_back(per_dim_index[dim]);
        }

        std::vector<std::string> operand_index;
        operand_index.push_back(op->get_operand(1)->get_name());

        for (auto const &idx : offset_dim_index) {
            operand_index.push_back(idx);
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_gather, operand_index);
    }

    void IndexTracerLazy::trace_gather_indices(std::shared_ptr<const Gather> op) {
        int index_vector_dim = op->get_index_vector_dim();
        std::vector<int> offset_dims = op->get_offset_dims();
        std::vector<int> slice_sizes = op->get_slice_sizes();
        std::vector<int> collapsed_slice_dims = op->get_collapsed_slice_dims();
        std::vector<int> start_index_map = op->get_start_index_map();

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_gather = op->get_operand(0)->get_output_spec(0).get_shape();
        TensorShape start_indices_shape = op->get_operand(1)->get_output_spec(0).get_shape();

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        std::vector<int> batch_dims;
        for (int idx = 0; idx < shape.rank(); ++idx) {
            if (std::find(offset_dims.begin(), offset_dims.end(), idx) == offset_dims.end()) {
                batch_dims.push_back(idx);
            }
        }

        std::vector<std::string> batch_dim_index, offset_dim_index;

        for (auto const &dim : batch_dims) {
            batch_dim_index.push_back(per_dim_index[dim]);
        }

        for (auto const &dim : offset_dims) {
            offset_dim_index.push_back(per_dim_index[dim]);
        }

        this->index = IndexTransform::multi_index_to_offset(start_indices_shape, batch_dim_index);
    }

    void IndexTracerLazy::trace_pad(std::shared_ptr<const Pad> op) {
        std::vector<int> padding_high = op->get_padding_high();
        std::vector<int> padding_low = op->get_padding_low();

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_pad = op->get_operand(0)->get_output_spec(0).get_shape();

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);
        std::vector<std::string> per_dim_index_before_pad(per_dim_index.size());

        for (int idx = 0; idx < (int)per_dim_index.size(); ++idx) {
            per_dim_index_before_pad[idx] = mononn_engine::helpers::string_format("(%s - %s)",
                                                                             per_dim_index[idx].c_str(),
                                                                             std::to_string(padding_low[idx]).c_str());
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_pad, per_dim_index_before_pad);
    }

    void IndexTracerLazy::trace_transpose(std::shared_ptr<const Transpose> op) {

        static bool __already_logged = false;
        if (!__already_logged) {
            __already_logged = true;
            LOG(WARNING) << "Use shared memory for fast transpose";
        }

        std::vector<int> perm = op->get_permute();
        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_transpose = op->get_operand(0)->get_output_spec(0).get_shape();

        if (op->need_smem()) {
            LOG(WARNING) << "Transpose" << op->get_name() << " cannot be efficiently handled with simple index mapping";
        }

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);
        std::vector<std::string> per_dim_index_before_transpose(per_dim_index.size());

        EXPECT_TRUE(perm.size() == per_dim_index_before_transpose.size(), "Rank mismatch");

        for (int idx = 0; idx < (int)perm.size(); ++idx) {
            per_dim_index_before_transpose[perm[idx]] = per_dim_index[idx];
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_transpose, per_dim_index_before_transpose);
    }

    void IndexTracerLazy::trace_slice(std::shared_ptr<const Slice> op) {
        std::vector<int> slice_starts = op->get_slice_starts();
        std::vector<int> slice_limits = op->get_slice_limits();

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_slice = op->get_operand(0)->get_output_spec(0).get_shape();

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);
        std::vector<std::string> per_dim_index_before_slice(per_dim_index.size());

        for (int idx = 0; idx < (int)per_dim_index_before_slice.size(); ++idx) {
            per_dim_index_before_slice[idx] = mononn_engine::helpers::string_format("(%s + %s)",
                                                                               per_dim_index[idx].c_str(),
                                                                               std::to_string(slice_starts[idx]).c_str());
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_slice, per_dim_index_before_slice);
    }

    void IndexTracerLazy::trace_reduce(std::shared_ptr<const Reduce> op, std::string const &inverse_reduce_dim) {
        this->index = mononn_engine::helpers::string_format("((%s * %s) + %s)",
                                                       this->index.c_str(),
                                                       std::to_string(op->get_operand(0)->get_output_spec(0).get_shape(-1)).c_str(),
                                                       inverse_reduce_dim.c_str());
    }
}
}
}