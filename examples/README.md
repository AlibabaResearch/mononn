# Example: Compile a BERT model
We provide scripts to save, freeze and compile Tensorflow models. 
We take the BERT-Base model from Huggingface to demonstrate the optimization flow of MonoNN.

## Step 0
Download the BERT model from Huggingface and save it to **bert_base** directory using the script in ./utils
```
>> export MODEL=bert_base # Alternatively, MODEL can be set to vit/t5_small/t5_base/bert_large/clip/opt_125m
>> python utils/save_model.py --model $MODEL --model_dir $MODEL
...Some outputs...
>> ls $MODEL
assets  config.json  keras_metadata.pb  saved_model.pb  variables
```
## Step 1 
Convert Tensorflow saved model to the frozen graph.
```
>> python utils/savedmodel2frozengraph.py --model_dir $MODEL
```

After this step, **frozen.pb** file should exists under **bert_base** directory.

## Step 2
Begin MonoNN tuning.
```
>> python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task tuning \
  --mononn_home path_to_mononn_home \
  --mononn_dump_dir ./"$MODEL"_mononn_bs1 
```

*path_to_mononn_home* is the home directory of MonoNN. After tuning, the tuning result will be saved in *bert_base_mononn_bs1* and it can be loaded in subsequent inference.

Hint: model tuning may take a while depending on your machine's available resources. By default, MonoNN greedily uses all CPU cores for tuning. If you would like to limit the CPU usage of the MonoNN tuner, set **TF_MONONN_THREADPOOL_SIZE** environment variable to any positive value to limit the number of CPU threads in use.

## Step 3
Using MonoNN in inference is similar to the tuning procedure. Just need to specify from which directory should MonoNN compiler load the tuning result.
```
>> python run_mononn.py \
  --model $MODEL \
  --data_file data/bert_bs1.npy \
  --task inference \
  --mononn_home path_to_mononn_home \
  --mononn_spec_dir ./"$MODEL"_mononn_bs1 
```