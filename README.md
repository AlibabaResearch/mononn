
# Environmental Variables

## TF MonoNN 
TF_MONONN_ENABLED
TF_MONONN_DUMP_DIR
TF_MONONN_EXISTING_TUNING_SPEC_DIR
TF_MONONN_THREADPOOL_SIZE
TF_MONONN_ENABLE_WORKER_THRAED_LOGGING
TF_USE_MONONN_INST_SCHEDULE
TF_MONONN_LAUNCH_BLOCK_OVERRIDE
TF_MONONN_LAUNCH_GPU_SM_AFFINITY
TF_MONONN_LOAD_ALTERNATIVE_TUNING_SPEC
TF_MONONN_INFERENCE_DUMP_DIR
TF_MONONN_NO_USE_PTX_CACHE

## MonoNN Standalone
MONONN_STANDALONE_ENABLED

## Apply to all path
MONONN_HOME
MONONN_OPTIMIZATION_DISABLED_PASS
MONONN_CODEGEN_ALLOW_LIST_OVERRIDE

# Actionable Items:

## Model support
* GPT2
* GPT3

# Additional experiments

1. Large model GPT-2B or larger
2. Throughput experiments (for example, batched inference)
3. ILP result analyss, gain on instruction per cycle? Gain on pipeline stall time?
4. On-chip cache exploration performance breakdown? memory stall? 
5. Use origional Astitch artifact.
6. Per-kernel analysis use table or timeline, speed, resource usage, instruction stall. 
7. Use Tensorlfow as baseline?
8. CUDA grpah vs Monolithic execution.

# Other concerns
1. Shall we remove Wide&Deep?
2. 

pip install numpy wheel packaging requests 
pip install keras_preprocessing --no-deps

apt install uuid-dev