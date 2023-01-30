import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to model directory.')
parser.add_argument('--data_file', type=str, required=True, help='Numpy data file.')
parser.add_argument('--task', type=str, required=True, choices=['tuning', 'inference'], help='Run in tuning or inference mode.')
parser.add_argument('--mononn_home', type=str, required=True, help='MonoNN home directory.')
parser.add_argument('--mononn_dump_dir', type=str, default=None, help='Directory to save MonoNN tuning result. Required for tuning task.')
parser.add_argument('--mononn_spec_dir', type=str, default=None, help='Directory to load MonoNN tuning result. Required for inference task.')

args = parser.parse_args()

import os
import numpy as np
import time
os.environ['TF_MONONN_ENABLED'] = 'true'
os.environ['MONONN_HOME'] = args.mononn_home
os.environ['TF_XLA_FLAGS'] = ' --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

if args.task == 'tuning':
    assert args.mononn_dump_dir != None, 'Please specify mononn_dump_dir'
    os.environ['TF_MONONN_DUMP_DIR'] = args.mononn_dump_dir
if args.task == 'inference':
    assert args.mononn_spec_dir != None, 'Please specify mononn_spec_dir'
    os.environ['TF_MONONN_EXISTING_TUNING_SPEC_DIR'] = args.mononn_spec_dir

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def load_frozen_pb(model_file):
    graph_def = tf.compat.v1.GraphDef()
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def

def get_default_sess_config():
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.auto_mixed_precision = True
    config.experimental.mlir_bridge_rollout = 1
    config.experimental.enable_mlir_bridge = True
    return config

def inference():
    feed_dict = np.load(args.data_file, allow_pickle=True, encoding='bytes').item()
    config = get_default_sess_config()
    graph_def = load_frozen_pb(os.path.join(args.model, 'frozen.pb'))

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=graph, config=config)
    
    def do_inference():
        tic = time.time()
        result = sess.run(feed_dict.keys(), feed_dict)
        return time.time() - tic, result


    warmup_num = 10
    infer_num = 100
    print('Performing warm-up {} times'.format(warmup_num))
    _ = [do_inference() for i in range(warmup_num)]


    print('Performing inference {} times'.format(infer_num))
    runtimes = [t for t, ret in [do_inference() for i in range(infer_num)]]
        
    rt_str = ' '.join(['{:.2f}'.format(rt * 1000) for rt in runtimes])
    print('Runtimes(ms): {}'.format(rt_str))
    print('Runtime(ms): {:.2f}'.format(np.mean(runtimes) * 1000))

if __name__ == '__main__':
    inference()
    
