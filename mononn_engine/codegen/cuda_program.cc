// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/codegen/cuda_program.h"

#include <experimental/filesystem>
#include <fstream>

#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/directory.h"
#include "mononn_engine/helpers/env_variable.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/path.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace codegen {
using Config = mononn_engine::config::Config;
using EnvVar = mononn_engine::helpers::EnvVar;

void CUDAProgram::append_main(std::string const& content) {
  this->append_file("main.cu", content);
}

void CUDAProgram::append_file(std::string const& file_name,
                              std::string const& content) {
  if (this->files.find(file_name) == this->files.end()) {
    this->files[file_name] = std::stringstream();
  } else {
    this->files[file_name] << content;
  }
}

std::vector<std::string> CUDAProgram::get_file_list() const {
  std::vector<std::string> file_list;
  file_list.reserve(this->files.size());

  for (auto const& [file_name, _] : this->files) {
    file_list.push_back(file_name);
  }

  return file_list;
}

std::stringstream& CUDAProgram::file_ref(const std::string& file_name) {
  if (this->files.find(file_name) == this->files.end()) {
    this->files[file_name] = std::stringstream();
  }

  return this->files[file_name];
}

const std::stringstream& CUDAProgram::file_ref(
    const std::string& file_name) const {
  if (this->files.find(file_name) == this->files.end()) {
    LOG(FATAL) << file_name << " not found.";
  }

  return this->files.at(file_name);
}

void CUDAProgram::generate(std::string const& directory,
                           CompileOutputType type) const {
  namespace fs = std::experimental::filesystem;

  if (!fs::is_directory(directory) || !fs::exists(directory)) {
    fs::create_directory(directory);
  }

  std::ofstream ofs;
  std::string make_file =
      mononn_engine::helpers::Path::join(directory, "Makefile");
  ofs.open(make_file);
  EXPECT_TRUE(ofs.is_open(), "Cannot open file " + make_file);
  ofs << this->get_makefile(type);
  ofs.close();

  for (auto const& [file_name, file_content] : this->files) {
    std::string full_file_name =
        mononn_engine::helpers::Path::join(directory, file_name);
    ofs.open(full_file_name);
    EXPECT_TRUE(ofs.is_open(), "Cannot open file " + full_file_name);
    ofs << file_content.str();
    ofs.close();
  }

  for (auto const& model_data : this->model_data_list) {
    model_data->save_to_dir(directory);
  }

  for (auto const& data_file : this->data_files) {
    std::string file_name =
        mononn_engine::helpers::string_split(data_file, '/').back();
    std::string dst_file_name =
        mononn_engine::helpers::Path::join(directory, file_name);
    if (fs::exists(dst_file_name)) {
      fs::remove(dst_file_name);
    }

    fs::copy(data_file,
             mononn_engine::helpers::Path::join(directory, file_name));
  }
}

bool CUDAProgram::build(std::string const& directory) const {
  LOG(FATAL) << "Not implemented";
}

bool CUDAProgram::generate_and_build(std::string const& directory) const {
  this->generate(directory);
  return this->build(directory);
}

void CUDAProgram::add_model_data(std::unique_ptr<ModelData> model_data) {
  this->model_data_list.push_back(std::move(model_data));
}

void CUDAProgram::add_data_file(std::string data_file) {
  this->data_files.push_back(data_file);
}

std::string CUDAProgram::get_makefile(CompileOutputType type) const {
  std::string cuda_arch = this->cuda_context->cuda_device_context.cuda_arch;
  cuda_arch = cuda_arch.substr(0, (int)cuda_arch.length() - 1);

  if (!EnvVar::defined("MONONN_HOME")) {
    LOG(FATAL) << "MONONN_HOME not defined.";
  }

  std::string MONONN_HOME = EnvVar::get("MONONN_HOME");

  std::string cutlass_include =
      mononn_engine::helpers::Path::join(MONONN_HOME, "cutlass_mononn/include");
  std::string cutlass_util_include = mononn_engine::helpers::Path::join(
      MONONN_HOME, "cutlass_mononn/tools/util/include");
  std::string onefuser_include =
      mononn_engine::helpers::Path::join(MONONN_HOME, "mononn_engine");

  std::stringstream ss;
  ss << "all : main.cu\n";
  ss << "\tnvcc -gencode=arch=compute_" << cuda_arch << ",code=sm_" << cuda_arch
     << " \\\n";
  ss << "\t-std=c++17 -w -use_fast_math \\\n";
  ss << "\t-I" << cutlass_include << " \\\n";
  ss << "\t-I" << cutlass_util_include << " \\\n";
  ss << "\t-I" << onefuser_include << " \\\n";
  ss << "\t-lz main.cu \\\n";
  ss << "\t-o mononn";

  if (type == CompileOutputType::COMPILE_OUTPUT_TYPE_CUBIN) {
    ss << ".cubin -cubin";
  } else if (type == CompileOutputType::COMPILE_OUTPUT_TYPE_PTX) {
    ss << ".ptx -ptx";
  } else if (type != CompileOutputType::COMPILE_OUTPUT_TYPE_BINARY) {
    LOG(FATAL) << "Unsupport type " << type;
  }

  return ss.str();
}

//     std::string CUDAProgram::get_cmakefile(bool build_cubin) const {
//         namespace fs = std::experimental::filesystem;

//         std::string cutlass_include =
//         mononn_engine::helpers::Path::join(Config::get()->onefuser_home,
//         "cutlass/include"); std::string cutlass_util_include =
//         mononn_engine::helpers::Path::join(Config::get()->onefuser_home,
//         "cutlass/tools/util/include"); std::string onefuser_include =
//         mononn_engine::helpers::Path::join(Config::get()->onefuser_home,
//         "tensorflow/onefuser");

//         std::string cmakefile_template = R"(
// cmake_minimum_required(VERSION 3.20)
// project(onefuser CUDA)

// set(CMAKE_CUDA_STANDARD 17)

// set(CMAKE_CUDA_FLAGS "-w -use_fast_math")
// set(CMAKE_CUDA_ARCHITECTURES %s)

// include_directories(%s
//     %s
//     %s)

// link_libraries(-lz)

// add_executable(onefuser %s)

// set_target_properties(onefuser PROPERTIES
//         CUDA_SEPARABLE_COMPILATION ON)
// )";

//         std::string cuda_arch =
//         this->cuda_context->cuda_device_context.cuda_arch; cuda_arch =
//         cuda_arch.substr(0, (int)cuda_arch.length() - 1);
//         std::vector<std::string> file_list;

//         file_list.push_back("main.cu");
//         for (auto const &[file_name, file_context] : this->files) {
//             file_list.push_back(file_name);
//         }

//         std::string file_list_str =
//         mononn_engine::helpers::join(std::string("\n"), file_list);

//         return mononn_engine::helpers::string_format(cmakefile_template,
//         cuda_arch.c_str(), cutlass_include.c_str(),
//         cutlass_util_include.c_str(), onefuser_include.c_str(),
//         file_list_str.c_str());
//     }
}  // namespace codegen
}  // namespace mononn_engine