# Introduction

This repository holds the artifact for the MonoNN compiler. ["MonoNN: Enabling a New Monolithic Optimization Space for Neural Network
Inference Tasks on Modern GPU-Centric Architectures"](https://www.usenix.org/conference/osdi24/presentation/zhuang).


MonoNN is a deep learning compiler that can automatically generate the entire neural network into a single CUDA kernel launch for minimized non-computation overhead.
MonoNN collective optimizes various neural network operators (e.g., compute-intensive and memory-intensive operators) in a unified optimization space, which we call **monolithic optimization space**.  
Various techniques are used to reconcile incompatibility between different types of operators, optimize memory access patterns, and accelerate the tuning process. 
Please refer to our pre-print paper for an in-depth view of MonoNN.

# Prerequisite
- Bazel 5.0.0  
- GCC 9.4  
- Python 3.8  
- Ubuntu 20.04  
- CUDA 11.6
- Libuuid (apt install uuid-dev)  
- Zlib (apt install libz-dev)
- Conda  
- Python 3.8  
- NVIDIA GPU available. We recommend use T4, A100, or A10.  
# Build from source

## Clone the code
```
git clone --recursive git@github.com:AlibabaResearch/mononn.git
cd mononn/tensorflow_mononn
```
## Prepare Python environment
```
conda create -n mononn python=3.8
conda activate mononn
pip install numpy wheel packaging requests opt_einsum transformers==4.20.0
pip install keras_preprocessing --no-deps
```

## Configure the build

```
./configure
```

Make sure enable CUDA support:
```
Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.
```

Below is a completed configuration session:  


<details>
  <summary>Click me to expand</summary>

```
(mononn) root@i39a12200:/home/zhuangdonglin.zdl/workspace/mononn/tensorflow_mononn# ./configure 
You have bazel 5.0.0 installed.
Please specify the location of python. [Default is /home/zhuangdonglin.zdl/workspace/.conda/envs/mononn/bin/python3]: 


Found possible Python library paths:
  /home/zhuangdonglin.zdl/workspace/.conda/envs/mononn/lib/python3.8/site-packages
Please input the desired Python library path to use.  Default is [/home/zhuangdonglin.zdl/workspace/.conda/envs/mononn/lib/python3.8/site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: N
No TensorRT support will be enabled for TensorFlow.

Found CUDA 11.6 in:
    /usr/local/cuda-11.6/targets/x86_64-linux/lib
    /usr/local/cuda-11.6/targets/x86_64-linux/include
Found cuDNN 8 in:
    /usr/lib/x86_64-linux-gnu
    /usr/include


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 8.6


Do you want to use clang as CUDA compiler? [y/N]: N
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=mkl_aarch64 	# Build with oneDNN and Compute Library for the Arm Architecture (ACL).
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
	--config=v1          	# Build with TensorFlow 1 API instead of TF 2 API.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=nogcp       	# Disable GCP support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```
</details>

## Build and install
This may take a while depends on your available CPU cores.

```
bazel build //tensorflow/tools/pip_package:build_pip_package --nocheck_visibility
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-2.9.2-cp38-cp38-linux_x86_64.whl
```

# Example: Compile a BERT model
See here: [Compile a BERT model](./examples/README.md)

# Enable the MonoNN compiler in your model
MonoNN can be enabled and seamlessly optimize the Tensorflow model by setting the following environment variables.
```
export TF_MONONN_ENABLED=true
export MONONN_HOME=/path/to/mononn/
# XLA compiler should be enabled as well. 
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Then MonoNN compiler should automatically optimize your Tensorflow model.
python run_your_tensorflow_model.py
```

It may take a while for the MonoNN tuner to explore the optimization space defined by the MonoNN compiler during the first inference iteration. 
One can optionally save tuning results using the below environment variable.

```
export TF_MONONN_DUMP_DIR=/path/to/tuning/spec
```

In this way MonoNN can load existing tuning specifications for subsequent inference, No tuning is needed.
```
export TF_MONONN_EXISTING_TUNING_SPEC_DIR=/path/to/tuning/spec
python run_your_tensorflow_model.py
```

# (Optional) Inspect MonoNN-generated code 

MonoNN will dump generated CUDA and PTX code to TF_MONONN_DUMP_DIR if specified.
Here is a sample directory structure for dumped files.
```
└── cluster_0
   ├── 72_1_1_128_1_1_src
   │   ├── headers.cuh
   │   ├── main.cu
   ├── 144_1_1_128_1_1_src
   │   ├── headers.cuh
   │   ├── main.cu
   ├── 72_1_1_128_1_1.json
   ├── 72_1_1_128_1_1.ptx
   ├── 144_1_1_128_1_1.json
   ├── 144_1_1_128_1_1.ptx
   ├── ......
   ├── best_tuning_spec.json
   ├── best_tuning_spec.ptx
   ├── performance_report.csv
   ├── tuning_log.log
```

Ideally, there will exist only one cluster if the XLA clustering phase runs smoothly.
x_1_1_y_1_1.json is the tuning result in JSON format under specific TLP settings: x thread blocks and y threads for each thread block.
x_1_1_y_1_1_src holds the corresponding CUDA source code and x_1_1_y_1_1.ptx holds the corresponding PTX assembly code. 
tuning_log.log contains measured inference latency for each x_y combination. performance_report.csv contains detailed subgraph-level performance data, which is useful to identify execution hotspots within the monolithic kernel. 

## (Advanced) Directly use MonoNN generated CUDA/PTX code.

Cumbersome Tensorflow dependency is not necessary in many cases. 
To fully get rid of Tensorflow dependency, one can directly use MonoNN-generated CUDA/PTX code.
For example, use **best_tuning_spec.ptx** with **cuModuleLoadData** in your CUDA/C++ applications or in other deep learning frameworks such as PyTorch.
MonoNN complies following convention when defining the CUDA kernel.

- Kernel Name: mononn_cluster_x, where x is the cluster id shown in the dumped directory name. 
- Kernel parameter: Suppose your model has n inputs, the kernel parameter would be (input<sub>1</sub>, input<sub>n</sub>,...,input<sub>n</sub>, output_buffer, temporary_buffer). The input buffer should be filled with input data before kernel invocation. The size of each buffer could be found in the MonoNN output log:
  <pre>
  2023-01-30 11:19:12.846397: I tensorflow/mononn_extra/hlo_memory_scheduler_mononn.cc:250] ScheduleComputationMonoNN for module cluster_0__XlaCompiledKernel_true__XlaHasReferenceVars_false__XlaNumConstantArgs_1__XlaNumResourceArgs_0_.626 complete.
  2023-01-30 11:19:12.927441: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1750] MonoNN Buffer allocation summary:
  2023-01-30 11:19:12.927493: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1751] 	Parameter buffers:
  2023-01-30 11:19:12.927505: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1754] 		Parameter 0: 512 bytes.
  2023-01-30 11:19:12.927510: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1754] 		Parameter 1: 512 bytes.
  2023-01-30 11:19:12.927522: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1754] 		Parameter 2: 512 bytes.
  2023-01-30 11:19:12.927530: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1757] 	Liveout(output) buffer size: 65536 bytes.
  2023-01-30 11:19:12.927540: I tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1759] 	Temporary buffer size: 1607832 bytes.
  </pre>

- Grid and block setting: Can be found in **best_tuning_spec.json** under **grid_dim** and **block_dim** keyword.
- Dynamic shared memory setting: Can be found in **best_tuning_spec.json** under **smem_size** keyword.


# MonoNN code structure 
```
├── mononn_engine 
│   ├── cnpy                # Library to read/write .npy file in C++
│   ├── codegen             # Code for MonoNN code generation
│   ├── config              # Data structure that holds configurations.
│   ├── core                # MonoNN core data structures
│   │   ├── common          # Commonly used utilities & interfaces (e.g., concurrent queue)
│   │   ├── context         # Codegen context management (e.g., CUDA context, codegen index calculation.)
│   │   ├── edge            # Edges in the graph
│   │   ├── gpu             # GPU information & functionality wrapper.
│   │   ├── graph           # Computation graph definition.
│   │   ├── op              # Operator definition.
│   │   ├── op_annotation   # Operator annotation.
│   │   ├── op_impl         # Operator implementation definition.
│   │   ├── schedule        # Loop schedule
│   │   ├── semantic        # CUDA C/C++ semantic wrapper
│   │   ├── tensor          # Tensor and data type specification.
│   ├── module              # Entry point of MonoNN codegen engine. Including a self-contained MonoNN module and MonoNN module tuner.
│   ├── helpers             # Helpers classes for string, file, directory, protobuf, multi-process etc.
│   ├── optimization        # Optimization passes.
│   ├── parser              # Parser converts XLA HLO IR to MonoNN computation graph.
│   ├── proto               # Protocol buffers.
│   ├── tuning              # Core functionalities for MonoNN module tuning.
├── tensorflow_mononn       # Tensorflow used by MonoNN. Code for MonoNN-TF integration.
└── cutlass_mononn          # CUTLASS used by MonoNN
```

# Miscellaneous

MonoNN compiler is designed for neural network inference. 
Please do not enable it in Tensorflow training workloads.
# Acknowledgement
MonoNN depends on the below repositories for its core functionality:

- [Tensorflow](https://github.com/tensorflow/tensorflow): MonoNN heavily depends on TF XLA execution pipeline. Including graph execution, IR lowering, buffer management, etc.
- [CUTLASS](https://github.com/nvidia/cutlass): MonoNN adopt CUTLASS as basic building blocks for GEMMs and Convolutions in the monolithic kernel.
- [Huggingface Transformers](https://github.com/huggingface/transformers): We use neural network implementation from Huggingface to evaluate the effectiveness of MonoNN optimization. 
- [Cnpy](https://github.com/rogersce/cnpy): MonoNN use cnpy to read and write .npy file.

# Citation 

```
@inproceedings{DBLP:conf/osdi/ZhuangZXQB0S24,
  author       = {Donglin Zhuang and
                  Zhen Zheng and
                  Haojun Xia and
                  Xiafei Qiu and
                  Junjie Bai and
                  Wei Lin and
                  Shuaiwen Leon Song},
  editor       = {Ada Gavrilovska and
                  Douglas B. Terry},
  title        = {MonoNN: Enabling a New Monolithic Optimization Space for Neural Network
                  Inference Tasks on Modern GPU-Centric Architectures},
  booktitle    = {18th {USENIX} Symposium on Operating Systems Design and Implementation,
                  {OSDI} 2024, Santa Clara, CA, USA, July 10-12, 2024},
  pages        = {989--1005},
  publisher    = {{USENIX} Association},
  year         = {2024},
  url          = {https://www.usenix.org/conference/osdi24/presentation/zhuang},
  timestamp    = {Tue, 16 Jul 2024 16:42:27 +0200},
  biburl       = {https://dblp.org/rec/conf/osdi/ZhuangZXQB0S24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```