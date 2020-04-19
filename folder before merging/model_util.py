from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def extract_submodel(model, inputs, outputs, name=None):
  output_to_layer = {}
  output_to_layer_input = {}
  for layer in model.layers:
    layer_output = layer.output
    layer_inputs = layer.input
    output_to_layer[layer_output] = layer
    output_to_layer_input[layer_output] = layer_inputs

  model_inputs_dict = {}
  memoized_results = {}

  # Relies on recursion, very low limit in python
  def _recurse_in_model(tensor):
    if tensor in memoized_results:
      return memoized_results[tensor]
    if (tensor == inputs) or (isinstance(inputs, list) and tensor in inputs):
      if tensor not in model_inputs_dict:
        model_inputs_dict[tensor] = tf.keras.layers.Input(tensor=tensor)
      out = model_inputs_dict[tensor]
    else:
      cur_inputs = output_to_layer_input[tensor]
      cur_layer = output_to_layer[tensor]
      if isinstance(cur_inputs, list):
        out = cur_layer([_recurse_in_model(inp) for inp in cur_inputs])
      else:
        out = cur_layer(_recurse_in_model(cur_inputs))
    memoized_results[tensor] = out
    return out

  if isinstance(outputs, list):
    model_outputs = [_recurse_in_model(tensor) for tensor in outputs]
  else:
    model_outputs = _recurse_in_model(outputs)

  if isinstance(inputs, list):
    model_inputs = [model_inputs_dict[tensor] for tensor in inputs]
  else:
    model_inputs = model_inputs_dict[inputs]

  return tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)
