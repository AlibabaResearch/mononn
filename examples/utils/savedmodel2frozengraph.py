import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--signature', type=str, default='serving_default')
args = parser.parse_args()

import tensorflow as tf
import os
from tensorflow.compat.v1.saved_model import tag_constants
from tensorflow.compat.v1.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants

saved_model_loaded = tf.saved_model.load(args.model_dir)

graph_func = saved_model_loaded.signatures[
    args.signature]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

inputs = [input.name for input in frozen_func.inputs]
outputs = [output.name for output in frozen_func.outputs]

inputs_shape = [str(input.shape) for input in frozen_func.inputs]
outputs_shape = [str(output.shape) for output in frozen_func.outputs]

tf.io.write_graph(frozen_func.graph, args.model_dir, name='frozen.pb', as_text=False)

output_str = 'inputs: ' + ','.join(inputs) + '\noutputs: ' + ','.join(outputs) + '\ninputs shape: ' + ' '.join(inputs_shape) + '\noutputs shape: ' + ' '.join(outputs_shape)
tf.io.write_file(os.path.join(args.model_dir, 'frozen.pb.info'), output_str)