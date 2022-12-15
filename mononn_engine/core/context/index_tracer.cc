#include "mononn_engine/core/context/index_tracer.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "mononn_engine/core/context/index_transform.h"

namespace mononn_engine {
namespace core {
namespace context {
    using OpType = mononn_engine::core::op::OpType;
    using Constant = mononn_engine::core::op::Constant;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using IndexTransform = mononn_engine::core::context::IndexTransform;

    const std::string IndexSymbols::linear_index = "linear_index";
    const std::string IndexSymbols::strided_index = "strided_index";
    const std::string IndexSymbols::ilp_variable_suffix = "ilp_variable_suffix";
    const std::string IndexSymbols::ilp_index_id = "ilp_index_id";


    void IndexTracer::trace(const std::shared_ptr<const Op> &op) {
        if (op->get_type() == OpType::broadcast) {
            this->trace_broadcast(std::static_pointer_cast<const Broadcast>(op));
        } else if (op->get_type() == OpType::dynamic_slice) {
            this->trace_dynamic_slice(std::static_pointer_cast<const DynamicSlice>(op));
        } else if (op->get_type() == OpType::dynamic_update_slice) {
            LOG(FATAL) << "DynamicUpdateSlice operator should be traced explicitly for operand or indices";
        }else if (op->get_type() == OpType::gather) {
            LOG(FATAL) << "Gather operator should be traced explicitly for operand or indices";
        } else if (op->get_type() == OpType::pad) {
            this->trace_pad(std::static_pointer_cast<const Pad>(op));
        } else if (op->get_type() == OpType::transpose) {
            this->trace_transpose(std::static_pointer_cast<const Transpose>(op));
        } else if (op->get_type() == OpType::slice) {
            this->trace_slice(std::static_pointer_cast<const Slice>(op));
        } else if (op->get_type() == OpType::reduce) {
            LOG(FATAL) << "Reduce operator should be traced explicitly";
        } else if (op->get_type() == OpType::reduce_window) {
            this->trace_reduce_window(std::static_pointer_cast<const ReduceWindow>(op));  
        } else if (op->get_type() == OpType::concatenate) {
            LOG(FATAL) << "Concatenate operator should be trace explicitly";
        }
    }

    void IndexTracer::set_index(std::string _index) {
        this->index = _index;
    }

    void IndexTracer::set_pred(std::string _pred) {
        this->pred = _pred;
    }

    std::string IndexTracer::get_index() const {
        return this->index;
    }

    std::string IndexTracer::get_predictive() const {
        return this->pred;
    }

    void IndexTracer::trace_broadcast(const std::shared_ptr<const Broadcast> &op) {
        static int __already_logged = false;
        if (!__already_logged) {
            __already_logged = true;
            LOG(INFO) << "Trace broadcast can perform additional optimizations for lesser index calculation";
        }

        if (op->get_operand(0)->get_type() == OpType::constant &&
        std::static_pointer_cast<Constant>(op->get_operand(0))->is_scalar()) {
            return;
        }

        std::vector<int> broadcast_dims = op->get_dimensions();

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_broadcast = op->get_operand(0)->get_output_spec(0).get_shape();

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        EXPECT_TRUE(broadcast_dims.size() == shape_before_broadcast.rank() 
            || (broadcast_dims.size() == 0 && shape_before_broadcast.element_count() == 1 /*scalar to tensor broadcast*/), 
                "Rank not match for node " + op->get_name() + 
                ". Broadcast dimension size " + std::to_string(broadcast_dims.size()) + " shape before broadcast: " + shape_before_broadcast.to_string());

        std::vector<std::string> per_dim_index_before_broadcast;

        if (broadcast_dims.empty()) {
            this->index = "0";
        } else {
            for (auto const &dim : broadcast_dims) {
                        per_dim_index_before_broadcast.push_back(per_dim_index[dim]);
            }

            this->index = IndexTransform::multi_index_to_offset(shape_before_broadcast, per_dim_index_before_broadcast);
        }
    }

    void IndexTracer::trace_dynamic_slice(const std::shared_ptr<const DynamicSlice> &op) {
        std::vector<int> dynamic_slice_sizes = op->get_dynamic_slice_sizes();
        std::vector<std::string> start_indices_list;

        for (int idx = 1; idx < op->get_operand_count(); ++idx) {
            start_indices_list.push_back(op->get_operand(idx)->get_name());
        }

        if (op->get_operand(0)->get_output_spec(0).rank() != start_indices_list.size()) {
            LOG(FATAL) << "Rank mismatch " << op->get_operand(0)->get_output_spec(0).rank() 
                << " vs " << start_indices_list.size();
        }

        if (op->get_operand(0)->get_output_spec(0).rank() != dynamic_slice_sizes.size()) {
            LOG(FATAL) << "Rank mismatch " << op->get_operand(0)->get_output_spec(0).rank() 
                << " vs " << dynamic_slice_sizes.size();
        }

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_dynamic_slice = op->get_operand(0)->get_output_spec(0).get_shape();
        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        for (int idx = 0; idx < per_dim_index.size(); ++idx) {
            per_dim_index[idx] = mononn_engine::helpers::string_format("clamp(%s{ilp_variable_suffix}, 0, %d) + (%s)",
                start_indices_list[idx].c_str(), shape_before_dynamic_slice.get_shape(idx) - dynamic_slice_sizes[idx], per_dim_index[idx].c_str());
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_dynamic_slice, per_dim_index);
    }

    void IndexTracer::trace_dynamic_update_slice_operand(const std::shared_ptr<const DynamicUpdateSlice> &op) {
        // Same logic as trace_dynamic_update_slice_update, but only record predicate.
        std::vector<std::string> start_indices_list;
        for (int idx = 2; idx < op->get_operand_count(); ++idx) {
            start_indices_list.push_back(op->get_operand(idx)->get_name());
        }

        if (op->get_operand(0)->get_output_spec(0).rank() != start_indices_list.size()) {
            LOG(FATAL) << "Rank mismatch " << op->get_operand(0)->get_output_spec(0).rank() 
                << " vs " << start_indices_list.size();
        }

        if (op->get_operand(1)->get_output_spec(0).rank() != start_indices_list.size()) {
            LOG(FATAL) << "Rank mismatch " << op->get_operand(1)->get_output_spec(0).rank() 
                << " vs " << start_indices_list.size();
        }

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape update_shape = op->get_operand(1)->get_output_spec(0).get_shape();
        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        for (int idx = 0; idx < start_indices_list.size(); ++idx) {
            per_dim_index[idx] = mononn_engine::helpers::string_format("(%s) - clamp(%s{ilp_variable_suffix}, 0, %d)", per_dim_index[idx].c_str(), start_indices_list[idx].c_str(), shape.get_shape(idx) - update_shape.get_shape(idx));
            // Predicate here is opposite to trace_dynamic_update_slice_update
            this->pred = mononn_engine::helpers::string_format("(%s) && (!(%s >= 0) && (%s < %d))", this->pred.c_str(), per_dim_index[idx].c_str(), per_dim_index[idx].c_str(), update_shape.get_shape(idx));
        }
    }

    void IndexTracer::trace_dynamic_update_slice_update(const std::shared_ptr<const DynamicUpdateSlice> &op) {
        std::vector<std::string> start_indices_list;
        for (int idx = 2; idx < op->get_operand_count(); ++idx) {
            start_indices_list.push_back(op->get_operand(idx)->get_name());
        }

        if (op->get_operand(0)->get_output_spec(0).rank() != start_indices_list.size()) {
            LOG(FATAL) << "Rank mismatch " << op->get_operand(0)->get_output_spec(0).rank() 
                << " vs " << start_indices_list.size();
        }

        if (op->get_operand(1)->get_output_spec(0).rank() != start_indices_list.size()) {
            LOG(FATAL) << "Rank mismatch " << op->get_operand(1)->get_output_spec(0).rank() 
                << " vs " << start_indices_list.size();
        }

        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape update_shape = op->get_operand(1)->get_output_spec(0).get_shape();
        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        for (int idx = 0; idx < start_indices_list.size(); ++idx) {
            per_dim_index[idx] = mononn_engine::helpers::string_format("(%s) - clamp(%s{ilp_variable_suffix}, 0, %d)", per_dim_index[idx].c_str(), start_indices_list[idx].c_str(), shape.get_shape(idx) - update_shape.get_shape(idx));
            this->pred = mononn_engine::helpers::string_format("(%s) && (%s >= 0) && (%s < %d)", this->pred.c_str(), per_dim_index[idx].c_str(), per_dim_index[idx].c_str(), update_shape.get_shape(idx));
        }

        this->index = IndexTransform::multi_index_to_offset(update_shape, per_dim_index);
    }

    void IndexTracer::trace_gather_operand(const std::shared_ptr<const Gather> &op) {
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
        operand_index.push_back(op->get_operand(1)->get_name() + "{ilp_variable_suffix}");

        for (auto const &idx : offset_dim_index) {
            operand_index.push_back(idx);
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_gather, operand_index);

    }

    void IndexTracer::trace_gather_operand_ilp(const std::shared_ptr<const Gather> &op, int ilp_id) {
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
        operand_index.push_back(mononn_engine::helpers::get_node_ilp_name(op->get_operand(1)->get_name(), ilp_id));

        for (auto const &idx : offset_dim_index) {
            operand_index.push_back(idx);
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_gather, operand_index);
    }

    void IndexTracer::trace_gather_indices(const std::shared_ptr<const Gather> &op) {
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

    void IndexTracer::trace_pad(const std::shared_ptr<const Pad> &op) {
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

            if (padding_low[idx] != 0) {
                this->pred = mononn_engine::helpers::string_format("(%s) && (%s >= %d)", this->pred.c_str(), per_dim_index[idx].c_str(), padding_low[idx]);
            }

            if (padding_high[idx] != 0) {
                this->pred = mononn_engine::helpers::string_format("(%s) && (%s < %d)", this->pred.c_str(), per_dim_index[idx].c_str(),
                                                         shape.get_shape(idx) - padding_high[idx]);
            }
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_pad, per_dim_index_before_pad);
    }

    void IndexTracer::trace_transpose(const std::shared_ptr<const Transpose> &op) {

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

    void IndexTracer::trace_slice(const std::shared_ptr<const Slice> &op) {
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

    void IndexTracer::trace_reduce(const std::shared_ptr<const Reduce> &op, std::string const &inverse_reduce_dim) {
        if (inverse_reduce_dim.empty()) {
            // trace reduce node in kLoop cluster;
            return;
        }

        this->index = mononn_engine::helpers::string_format("((%s * %s) + %s)",
               this->index.c_str(),
               std::to_string(op->get_operand(0)->get_output_spec(0).get_shape(-1)).c_str(),
               inverse_reduce_dim.c_str());
    }

    void IndexTracer::trace_reduce_window(const std::shared_ptr<const ReduceWindow> &op) {
        TensorShape shape = op->get_output_spec(0).get_shape();
        TensorShape shape_before_reduce_window = op->get_operand(0)->get_output_spec(0).get_shape();
        
        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);
        std::vector<int> filter_size = op->get_filter_size();
        std::vector<int> padding_low = op->get_padding_low();
        std::vector<int> padding_high = op->get_padding_high();
        std::vector<int> stride = op->get_filter_stride();

        for (int idx = 0; idx < per_dim_index.size(); ++idx) {

            if (filter_size[idx] != 1) {
                // per dim start (0 - padding low) + idx * stride
                per_dim_index[idx] = mononn_engine::helpers::string_format("((0 - %d) + (%s * %d) + {window_position_%d})", 
                    padding_low[idx], per_dim_index[idx].c_str(), stride[idx], idx);
            }

            if (padding_low[idx] != 0) {
                // index * stride - padding low >= 0
                this->pred = mononn_engine::helpers::string_format("(%s) && (%s >= 0)", 
                    this->pred.c_str(),
                    per_dim_index[idx].c_str());
            }

            if (padding_high[idx] != 0) {
                // index * stride + window <= size + padding high
                this->pred = mononn_engine::helpers::string_format("(%s) && (%s < %d)", 
                    this->pred.c_str(),
                    per_dim_index[idx].c_str(),
                    shape_before_reduce_window.get_shape(idx));
            }
        }

        this->index = IndexTransform::multi_index_to_offset(shape_before_reduce_window, per_dim_index);
    }

    void IndexTracer::trace_concatenate(const std::shared_ptr<const Concatenate> &op, int operand_id) {
        int dimension = op->get_dimension();

        TensorShape shape = op->get_output_spec(0).get_shape();
        std::vector<TensorShape> operand_shapes;

        for (int idx = 0; idx < op->get_operand_count(); ++idx) {
            operand_shapes.push_back(op->get_operand(idx)->get_output_spec(0).get_shape());
        }

        std::vector<std::string> per_dim_index = IndexTransform::offset_to_multi_index(shape, this->index);

        std::string index_for_current_concat_branch;

        TensorShape left_shape = operand_shapes[0];
        left_shape.set_shape(dimension, 0);

        ///////////////// Step 1: Get prediction for operand with operand_id /////////////////
        std::string branch_prediction;
        if (operand_id != 0) {
            for (int shape_id = 0; shape_id < operand_id; ++shape_id) {
                left_shape = left_shape.concat_on_dim(operand_shapes[shape_id], dimension);
            }

            branch_prediction = mononn_engine::helpers::string_format("(%s) >= (%d)",
                                    per_dim_index[dimension].c_str(), left_shape.get_shape(dimension));
        }

        if (operand_id != op->get_operand_count() - 1) {
            if (!branch_prediction.empty()) branch_prediction += " && ";

            TensorShape left_shape_plus_one_shape = left_shape.concat_on_dim(operand_shapes[operand_id], dimension);

            branch_prediction += mononn_engine::helpers::string_format("(%s) < (%d)",
                                    per_dim_index[dimension].c_str(), left_shape_plus_one_shape.get_shape(dimension));
        }

        this->pred = mononn_engine::helpers::string_format("(%s) && (%s)", this->pred.c_str(), branch_prediction.c_str());

        ///////////////// Step 1: Get index for each branch /////////////////
        std::vector<std::string> multi_index = per_dim_index;
        multi_index[dimension] = mononn_engine::helpers::string_format("((%s) - (%d))",
                                per_dim_index[dimension].c_str(), left_shape.get_shape(dimension));
        index_for_current_concat_branch = IndexTransform::multi_index_to_offset(operand_shapes.back(), multi_index);

        this->index = index_for_current_concat_branch;
    }
}
}
}