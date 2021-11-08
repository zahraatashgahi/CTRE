# coding=utf-8
# Copyright 2020 RigL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module has helper functions for the interpolation experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os; import sys; sys.path.append(os.getcwd())
import re
import numpy as np

import tensorflow.compat.v1 as tf



DEFAULT_ERK_SCALE = 1.0


def mask_extract_name_fn(mask_name):
  return re.findall('(.+)/mask:0', mask_name)[0]


def get_n_zeros(size, sparsity):
  return int(np.floor(sparsity * size))


def calculate_sparsity(masks):
  dense_params = tf.constant(0.)
  sparse_params = tf.constant(0.)
  for mask in masks:
    dense_params += tf.cast(tf.size(mask), dtype=dense_params.dtype)
    sparse_params += tf.cast(tf.reduce_sum(mask), dtype=sparse_params.dtype)
  return 1. - sparse_params / dense_params


def get_mask_random_numpy(mask_shape, sparsity, random_state=None):
  """Creates a random sparse mask with deterministic sparsity.

  Args:
    mask_shape: list, used to obtain shape of the random mask.
    sparsity: float, between 0 and 1.
    random_state: np.random.RandomState, if given the shuffle call is made using
      the RandomState

  Returns:
    numpy.ndarray
  """
  flat_ones = np.ones(mask_shape).flatten()
  n_zeros = get_n_zeros(flat_ones.size, sparsity)
  flat_ones[:n_zeros] = 0
  if random_state:
    random_state.shuffle(flat_ones)
  else:
    np.random.shuffle(flat_ones)
  new_mask = flat_ones.reshape(mask_shape)
  return new_mask


def get_mask_random(mask, sparsity, dtype, random_state=None):
  """Creates a random sparse mask with deterministic sparsity.

  Args:
    mask: tf.Tensor, used to obtain shape of the random mask.
    sparsity: float, between 0 and 1.
    dtype: tf.dtype, type of the return value.
    random_state: np.random.RandomState, if given the shuffle call is made using
      the RandomState

  Returns:
    tf.Tensor
  """
  new_mask_numpy = get_mask_random_numpy(
      mask.shape.as_list(), sparsity, random_state=random_state)
  new_mask = tf.constant(new_mask_numpy, dtype=dtype)
  return new_mask


def get_sparsities_erdos_renyi(all_masks,
                               default_sparsity,
                               custom_sparsity_map,
                               include_kernel,
                               extract_name_fn=mask_extract_name_fn,
                               erk_power_scale=DEFAULT_ERK_SCALE):
  """Given the method, returns the sparsity of individual layers as a dict.

  It ensures that the non-custom layers have a total parameter count as the one
  with uniform sparsities. In other words for the layers which are not in the
  custom_sparsity_map the following equation should be satisfied.

  # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
  Args:
    all_masks: list, of all mask Variables.
    default_sparsity: float, between 0 and 1.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    include_kernel: bool, if True kernel dimension are included in the scaling.
    extract_name_fn: function, extracts the variable name.
    erk_power_scale: float, if given used to take power of the ratio. Use
      scale<1 to make the erdos_renyi softer.

  Returns:
    sparsities, dict of where keys() are equal to all_masks and individiual
      masks are mapped to the their sparsities.
  """
  # We have to enforce custom sparsities and then find the correct scaling
  # factor.

  is_eps_valid = False
  # # The following loop will terminate worst case when all masks are in the
  # custom_sparsity_map. This should probably never happen though, since once
  # we have a single variable or more with the same constant, we have a valid
  # epsilon. Note that for each iteration we add at least one variable to the
  # custom_sparsity_map and therefore this while loop should terminate.
  dense_layers = set()
  while not is_eps_valid:
    # We will start with all layers and try to find right epsilon. However if
    # any probablity exceeds 1, we will make that layer dense and repeat the
    # process (finding epsilon) with the non-dense layers.
    # We want the total number of connections to be the same. Let say we have
    # for layers with N_1, ..., N_4 parameters each. Let say after some
    # iterations probability of some dense layers (3, 4) exceeded 1 and
    # therefore we added them to the dense_layers set. Those layers will not
    # scale with erdos_renyi, however we need to count them so that target
    # paratemeter count is achieved. See below.
    # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
    #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
    # eps * (p_1 * N_1 + p_2 * N_2) =
    #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
    # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

    divisor = 0
    rhs = 0
    raw_probabilities = {}
    for mask in all_masks:
      var_name = extract_name_fn(mask.name)
      shape_list = mask.shape.as_list()
      n_param = np.prod(shape_list)
      n_zeros = get_n_zeros(n_param, default_sparsity)
      if var_name in dense_layers:
        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
        rhs -= n_zeros
      elif var_name in custom_sparsity_map:
        # We ignore custom_sparsities in erdos-renyi calculations.
        pass
      else:
        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
        # equation above.
        n_ones = n_param - n_zeros
        rhs += n_ones
        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
        if include_kernel:
          raw_probabilities[mask.name] = (np.sum(shape_list) /
                                          np.prod(shape_list))**erk_power_scale
        else:
          n_in, n_out = shape_list[-2:]
          raw_probabilities[mask.name] = (n_in + n_out) / (n_in * n_out)
        # Note that raw_probabilities[mask] * n_param gives the individual
        # elements of the divisor.
        divisor += raw_probabilities[mask.name] * n_param
    # By multipliying individual probabilites with epsilon, we should get the
    # number of parameters per layer correctly.
    eps = rhs / divisor
    # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
    # mask to 0., so they become part of dense_layers sets.
    max_prob = np.max(list(raw_probabilities.values()))
    max_prob_one = max_prob * eps
    if max_prob_one > 1:
      is_eps_valid = False
      for mask_name, mask_raw_prob in raw_probabilities.items():
        if mask_raw_prob == max_prob:
          var_name = extract_name_fn(mask_name)
          tf.logging.info('Sparsity of var: %s had to be set to 0.', var_name)
          dense_layers.add(var_name)
    else:
      is_eps_valid = True

  sparsities = {}
  # With the valid epsilon, we can set sparsities of the remaning layers.
  for mask in all_masks:
    var_name = extract_name_fn(mask.name)
    shape_list = mask.shape.as_list()
    n_param = np.prod(shape_list)
    if var_name in custom_sparsity_map:
      sparsities[mask.name] = custom_sparsity_map[var_name]
      tf.logging.info('layer: %s has custom sparsity: %f', var_name,
                      sparsities[mask.name])
    elif var_name in dense_layers:
      sparsities[mask.name] = 0.
    else:
      probability_one = eps * raw_probabilities[mask.name]
      sparsities[mask.name] = 1. - probability_one
    tf.logging.info('layer: %s, shape: %s, sparsity: %f', var_name, mask.shape,
                    sparsities[mask.name])
  return sparsities


def get_sparsities_uniform(all_masks,
                           default_sparsity,
                           custom_sparsity_map,
                           extract_name_fn=mask_extract_name_fn):
  """Given the method, returns the sparsity of individual layers as a dict.

  Args:
    all_masks: list, of all mask Variables.
    default_sparsity: float, between 0 and 1.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    extract_name_fn: function, extracts the variable name.

  Returns:
    sparsities, dict of where keys() are equal to all_masks and individiual
      masks are mapped to the their sparsities.
  """
  sparsities = {}
  for mask in all_masks:
    var_name = extract_name_fn(mask.name)
    if var_name in custom_sparsity_map:
      sparsities[mask.name] = custom_sparsity_map[var_name]
    else:
      sparsities[mask.name] = default_sparsity
  print("final sparsities = ", sparsities)
  return sparsities



def get_sparsities(all_masks,
                   method,
                   default_sparsity,
                   custom_sparsity_map,
                   extract_name_fn=mask_extract_name_fn,
                   erk_power_scale=DEFAULT_ERK_SCALE):
  """Given the method, returns the sparsity of individual layers as a dict.

  Args:
    all_masks: list, of all mask Variables.
    method: str, 'random' or 'erdos_renyi'.
    default_sparsity: float, between 0 and 1.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    extract_name_fn: function, extracts the variable name.
    erk_power_scale: float, passed to the erdos_renyi function.

  Returns:
    sparsities, dict of where keys() are equal to all_masks and individiual
      masks are mapped to the their sparsities.

  Raises:
    ValueError: when a key from custom_sparsity not found in all_masks.
    ValueError: when an invalid initialization option is given.
  """
  # (1) Ensure all keys are valid and processed.
  keys_found = set()
  for mask in all_masks:
    var_name = extract_name_fn(mask.name)
    if var_name in custom_sparsity_map:
      keys_found.add(var_name)
  keys_given = set(custom_sparsity_map.keys())
  if keys_found != keys_given:
    diff = keys_given - keys_found
    raise ValueError('No masks are found for the following names: %s' %
                     str(diff))

  if method in ('erdos_renyi', 'erdos_renyi_kernel'):
    include_kernel = method == 'erdos_renyi_kernel'
    sparsities = get_sparsities_erdos_renyi(
        all_masks,
        default_sparsity,
        custom_sparsity_map,
        include_kernel=include_kernel,
        extract_name_fn=extract_name_fn,
        erk_power_scale=erk_power_scale)
  elif method == 'random':
    sparsities = get_sparsities_uniform(
        all_masks,
        default_sparsity,
        custom_sparsity_map,
        extract_name_fn=extract_name_fn)
  elif method == 'str':
    sparsities = get_sparsities_str(all_masks, default_sparsity)
  else:
    raise ValueError('Method: %s is not valid mask initialization method' %
                     method)
  return sparsities


def get_mask_init_fn(all_masks,
                     method,
                     default_sparsity,
                     custom_sparsity_map,
                     mask_fn=get_mask_random,
                     erk_power_scale=DEFAULT_ERK_SCALE,
                     extract_name_fn=mask_extract_name_fn):
  """Returns a function for initializing masks randomly.

  Args:
    all_masks: list, of all masks to be updated.
    method: str, method to initialize the masks, passed to the
      sparse_utils.get_mask() function.
    default_sparsity: float, if 0 mask left intact, if greater than one, a
      fraction of the ones in each mask is flipped to 0.
    custom_sparsity_map: dict, sparsity of individual variables can be
      overridden here. Key should point to the correct variable name, and value
      should be in [0, 1].
    mask_fn: function, to initialize masks with given sparsity.
    erk_power_scale: float, passed to get_sparsities.
    extract_name_fn: function, used to grab names from the variable.

  Returns:
    A callable to run after an init op. See `init_fn` of
    `tf.train.Scaffold`. Returns None if no `preinitialize_checkpoint` field
    is set in `RunnerSpec`.
  Raise:
    ValueError: when there is no mask corresponding to a key in the
      custom_sparsity_map.
  """
  sparsities = get_sparsities(
      all_masks,
      method,
      default_sparsity,
      custom_sparsity_map,
      erk_power_scale=erk_power_scale,
      extract_name_fn=extract_name_fn)
  print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
  print("*********************************************************")
  print(str(sparsities))
  tf.logging.info('Per layer sparsities are like the following: %s',
                  str(sparsities))
  assign_ops = []
  for mask in all_masks:
    new_mask = mask_fn(mask, sparsities[mask.name], mask.dtype)
    assign_op = tf.assign(mask, new_mask)
    assign_ops.append(assign_op)

  return tf.group(assign_ops)


## Calculating flops and parameters using a list of Keras layers.
def _get_kernel(layer):
  """Given the Keras layer returns the weights."""
  if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
    return layer.depthwise_kernel
  else:
    return layer.kernel


