#include "mononn_engine/codegen/model_data.h"

#include "mononn_engine/cnpy/cnpy.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace codegen {
void ModelData::add_float_data(std::vector<float> const& _data_float) {
  this->data_float = _data_float;
}

void ModelData::add_half_data(std::vector<Eigen::half> const& _data_half) {
  this->data_half = _data_half;
}

void ModelData::add_int_data(const std::vector<int>& _data_int) {
  this->data_int = _data_int;
}

void ModelData::add_data(void* _data) {
  if (this->data_type == Dtype::FLOAT16) {
    this->add_half_data(*reinterpret_cast<std::vector<Eigen::half>*>(_data));
  } else if (this->data_type == Dtype::FLOAT32) {
    this->add_float_data(*reinterpret_cast<std::vector<float>*>(_data));
  } else if (this->data_type == Dtype::INT32) {
    this->add_int_data(*reinterpret_cast<std::vector<int>*>(_data));
  } else {
    LOG(FATAL) << "";
  }
}

void ModelData::save_to_dir(std::string save_dir) const {
  std::string file_name = this->file_name;
  std::string save_file =
      mononn_engine::helpers::Path::join(save_dir, file_name);

  if (this->data_type == Dtype::FLOAT16) {
    cnpy::npy_save(save_file, this->data_half.data(), this->shape, "w");
  } else if (this->data_type == Dtype::FLOAT32) {
    cnpy::npy_save(save_file, this->data_float.data(), this->shape, "w");
  } else if (this->data_type == Dtype::INT32) {
    cnpy::npy_save(save_file, this->data_int.data(), this->shape, "w");
  } else {
    LOG(FATAL) << "Unsupported type " << this->data_type.to_string();
  }
}

}  // namespace codegen
}  // namespace mononn_engine