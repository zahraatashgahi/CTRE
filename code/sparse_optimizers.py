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

"""This module implements some common and new sparse training algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import sparse_utils
import six

from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import moving_averages
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import training_util

import tensorflow as tf
def extract_number(token):
  """Strips the number from the end of the token if it exists.

  Args:
    token: str, s or s_d where d is a number: a float or int. `foo_.5`,
      `foo_foo.5`, `foo_0.5`, `foo_4` are all valid strings.

  Returns:
    float, d if exists otherwise 1.
  """
  regexp = re.compile(r'.*_(\d*\.?\d*)$')
  if regexp.search(token):
    return float(regexp.search(token).group(1))
  else:
    return 1.


class SparseSETOptimizer(tf_optimizer.Optimizer):
  """Implementation of dynamic sparsity optimizers.

  Implementation of SET.
  See https://www.nature.com/articles/s41467-018-04316-3
  This optimizer wraps a regular optimizer and performs updates on the masks
  according to schedule given.

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    name: bool, passed to the super.
    use_stateless: bool, if True stateless operations are used. This is
      important for multi-worker jobs not to diverge.
    stateless_seed_offset: int, added to the seed of stateless operations. Use
      this to create randomness without divergence across workers.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               name='SparseSETOptimizer',
               use_stateless=True,
               stateless_seed_offset=0):
    super(SparseSETOptimizer, self).__init__(use_locking, name)
    self._optimizer = optimizer
    self._grow_init = grow_init
    self._drop_fraction_anneal = drop_fraction_anneal
    self._drop_fraction_initial_value = ops.convert_to_tensor(
        float(drop_fraction),
        name='%s_drop_fraction' % self._drop_fraction_anneal)
    self._begin_step = ops.convert_to_tensor(begin_step, name='begin_step')
    self._end_step = ops.convert_to_tensor(end_step, name='end_step')
    self._frequency = ops.convert_to_tensor(frequency, name='frequency')
    self._frequency_val = frequency
    self._use_stateless = use_stateless
    self._stateless_seed_offset = stateless_seed_offset

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    result = self._optimizer.compute_gradients(loss, **kwargs)
    return result

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    pre_op = self._before_apply_gradients(grads_and_vars)
    with ops.control_dependencies([pre_op]):
      optimizer_update = self._optimizer.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)
    # We get the default one after calling the super.apply_gradient(), since
    # we want to preserve original behavior of the optimizer: don't increment
    # anything if no global_step is passed. But we need the global step for
    # the mask_update.
    global_step = (
        global_step if global_step is not None else
        training_util.get_or_create_global_step())
    self._global_step = global_step
    with ops.control_dependencies([optimizer_update]):
      return self.cond_mask_update_op(global_step, control_flow_ops.no_op)

  def _before_apply_gradients(self, grads_and_vars):
    """Called before applying gradients."""
    return control_flow_ops.no_op('before_apply_grad')

  def cond_mask_update_op(self, global_step, false_branch):
    """Creates the conditional mask update operation.

    All masks are updated when it is an update iteration
    (checked by self.is_mask_update_iter()).
    Arguments:
      global_step: tf.Variable, current training iteration.
      false_branch: function, called when it is not a mask update iteration.

    Returns:
      conditional update operation
    """
    # Initializing to -freq so that last_update_step+freq=0. This enables early
    # mask_updates.
    last_update_step = variable_scope.get_variable(
        'last_mask_update_step', [],
        initializer=init_ops.constant_initializer(
            -self._frequency_val, dtype=global_step.dtype),
        trainable=False,
        dtype=global_step.dtype)

    def mask_update_op():
      update_ops = []
      for mask, weights in zip(self.get_masks(), self.get_weights()):
        update_ops.append(self.generic_mask_update(mask, weights))

      with ops.control_dependencies(update_ops):
        assign_op = state_ops.assign(
            last_update_step, global_step, name='last_mask_update_step_assign')
        with ops.control_dependencies([assign_op]):
          return control_flow_ops.no_op('mask_update')

    maybe_update = control_flow_ops.cond(
        self.is_mask_update_iter(global_step, last_update_step), mask_update_op,
        false_branch)
    return maybe_update

  def get_weights(self):
    return pruning.get_weights()

  def get_masks(self):
    return pruning.get_masks()

  def get_masked_weights(self):
    return pruning.get_masked_weights()

  def is_mask_update_iter(self, global_step, last_update_step):
    """Function for checking if the current step is a mask update step.

    It also creates the drop_fraction op and assigns it to the self object.

    Args:
      global_step: tf.Variable(int), current training step.
      last_update_step: tf.Variable(int), holding the last iteration the mask is
        updated. Used to determine whether current iteration is a mask update
        step.

    Returns:
      bool, whether the current iteration is a mask_update step.
    """
    gs_dtype = global_step.dtype
    self._begin_step = math_ops.cast(self._begin_step, gs_dtype)
    self._end_step = math_ops.cast(self._end_step, gs_dtype)
    self._frequency = math_ops.cast(self._frequency, gs_dtype)
    is_step_within_update_range = math_ops.logical_and(
        math_ops.greater_equal(global_step, self._begin_step),
        math_ops.logical_or(
            math_ops.less_equal(global_step, self._end_step),
            # If _end_step is negative, we never stop updating the mask.
            # In other words we update the mask with given frequency until the
            # training ends.
            math_ops.less(self._end_step, 0)))
    is_update_step = math_ops.less_equal(
        math_ops.add(last_update_step, self._frequency), global_step)
    is_mask_update_iter_op = math_ops.logical_and(is_step_within_update_range,
                                                  is_update_step)
    self.drop_fraction = self.get_drop_fraction(global_step,
                                                is_mask_update_iter_op)
    return is_mask_update_iter_op

  def get_drop_fraction(self, global_step, is_mask_update_iter_op):
    """Returns a constant or annealing drop_fraction op."""
    if self._drop_fraction_anneal == 'constant':
      drop_frac = self._drop_fraction_initial_value
    elif self._drop_fraction_anneal == 'cosine':
      decay_steps = self._end_step - self._begin_step
      drop_frac = learning_rate_decay.cosine_decay(
          self._drop_fraction_initial_value,
          global_step,
          decay_steps,
          name='cosine_drop_fraction')
    elif self._drop_fraction_anneal.startswith('exponential'):
      exponent = extract_number(self._drop_fraction_anneal)
      div_dtype = self._drop_fraction_initial_value.dtype
      power = math_ops.divide(
          math_ops.cast(global_step - self._begin_step, div_dtype),
          math_ops.cast(self._end_step - self._begin_step, div_dtype),
      )
      drop_frac = math_ops.multiply(
          self._drop_fraction_initial_value,
          math_ops.pow(1 - power, exponent),
          name='%s_drop_fraction' % self._drop_fraction_anneal)
    else:
      raise ValueError('drop_fraction_anneal: %s is not valid' %
                       self._drop_fraction_anneal)
    return array_ops.where(is_mask_update_iter_op, drop_frac,
                           array_ops.zeros_like(drop_frac))

  def generic_mask_update(self, mask, weights, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    masked_weights = mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=(hash(weights.name + 'drop')))
    # Randomly revive n_prune many connections from non-existing connections.
    score_grow = self._random_uniform(
        weights.shape, seed=hash(weights.name + 'grow'))
    return self._get_update_op(score_drop, score_grow, mask, weights)

  def _get_update_op(self,
                     score_drop,
                     score_grow,
                     mask,
                     weights,
                     reinit_when_same=False):
    """Prunes+grows connections, all tensors same shape."""
    old_dtype = mask.dtype
    mask_casted = math_ops.cast(mask, dtypes.float32)
    n_total = array_ops.size(score_drop)
    n_ones = math_ops.cast(math_ops.reduce_sum(mask_casted), dtype=dtypes.int32)
    n_prune = math_ops.cast(
        math_ops.cast(n_ones, dtype=dtypes.float32) * self.drop_fraction,
        dtypes.int32)
    n_keep = n_ones - n_prune

    # Sort the entire array since the k needs to be constant for TPU.
    _, sorted_indices = nn_ops.top_k(
        array_ops.reshape(score_drop, [-1]), k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    # We will have zeros after having `n_keep` many ones.
    new_values = array_ops.where(
        math_ops.range(n_total) < n_keep,
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask1 = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
    # Flatten the scores
    score_grow = array_ops.reshape(score_grow, [-1])
    # Set scores of the enabled connections(ones) to min(s) - 1, so that they
    # have the lowest scores.
    score_grow_lifted = array_ops.where(
        math_ops.equal(mask1, 1),
        array_ops.ones_like(mask1) * (math_ops.reduce_min(score_grow) - 1),
        score_grow)
    _, sorted_indices = nn_ops.top_k(score_grow_lifted, k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    new_values = array_ops.where(
        math_ops.range(n_total) < n_prune,
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask2 = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
    # Ensure masks are disjoint.
    assert_op = control_flow_ops.Assert(
        math_ops.equal(math_ops.reduce_sum(mask1 * mask2), 0.), [mask1, mask2])

    with ops.control_dependencies([assert_op]):
      # Let's set the weights of the growed connections.
      mask2_reshaped = array_ops.reshape(mask2, mask.shape)
    # Set the values of the new connections.
    grow_tensor = self.get_grow_tensor(weights, self._grow_init)
    if reinit_when_same:
      # If dropped and grown, we re-initialize.
      new_connections = math_ops.equal(mask2_reshaped, 1)
    else:
      new_connections = math_ops.logical_and(
          math_ops.equal(mask2_reshaped, 1), math_ops.equal(mask_casted, 0))
    new_weights = array_ops.where(new_connections, grow_tensor, weights)
    weights_update = state_ops.assign(weights, new_weights)
    # Ensure there is no momentum value for new connections
    reset_op = self.reset_momentum(weights, new_connections)

    with ops.control_dependencies([weights_update, reset_op]):
      mask_combined = array_ops.reshape(mask1 + mask2, mask.shape)
    mask_combined = math_ops.cast(mask_combined, dtype=old_dtype)
    new_mask = state_ops.assign(mask, mask_combined)
    return new_mask

  def reset_momentum(self, weights, new_connections):
    reset_ops = []
    for s_name in self._optimizer.get_slot_names():
      # Momentum variable for example, we reset the aggregated values to zero.
      optim_var = self._optimizer.get_slot(weights, s_name)
      new_values = array_ops.where(new_connections,
                                   array_ops.zeros_like(optim_var), optim_var)
      reset_ops.append(state_ops.assign(optim_var, new_values))
    return control_flow_ops.group(reset_ops)

  def get_grow_tensor(self, weights, method):
    """Different ways to initialize new connections.

    Args:
      weights: tf.Tensor or Variable.
      method: str, available options: 'zeros', 'random_normal', 'random_uniform'
        and 'initial_value'

    Returns:
      tf.Tensor same shape and type as weights.

    Raises:
      ValueError, when the method is not valid.
    """
    if not isinstance(method, six.string_types):
      raise ValueError('Grow-Init: %s is not a string' % method)

    if method == 'zeros':
      grow_tensor = array_ops.zeros_like(weights, dtype=weights.dtype)
    elif method.startswith('initial_dist'):
      original_shape = weights.initial_value.shape
      divisor = extract_number(method)
      grow_tensor = array_ops.reshape(
          random_ops.random_shuffle(
              array_ops.reshape(weights.initial_value, [-1])),
          original_shape) / divisor
    elif method.startswith('random_normal1'):
      stddev = math_ops.reduce_std(weights)
      divisor = extract_number(method)
      grow_tensor = self._random_normal(
          weights.shape,
          mean=0,
          stddev=1, #stddev,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_n')) / 10 #divisor
    elif method.startswith('random_normal2'):
      stddev = math_ops.reduce_std(weights)
      divisor = extract_number(method)
      grow_tensor = self._random_normal(
          weights.shape,
          mean=0,
          stddev=stddev,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_n')) /2
    elif method.startswith('random_uniform'):
      mean = math_ops.reduce_mean(math_ops.abs(weights))

      mean = np.sqrt(6. / weights.get_shape().as_list()[0])
      divisor = extract_number(method)
      grow_tensor = self._random_uniform(
          weights.shape,
          minval=-mean,
          maxval=mean,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_u')) / divisor
    else:
      raise ValueError('Grow-Init: %s is not a valid option.' % method)
    return grow_tensor

  def _random_uniform(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = math_ops.cast(
          array_ops.stack([c_seed, self._global_step]), dtypes.int32)
      return stateless_random_ops.stateless_random_uniform(*args, **kwargs)
    else:
      return random_ops.random_uniform(*args, **kwargs)

  def _random_normal(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = math_ops.cast(
          array_ops.stack([c_seed, self._global_step]), dtypes.int32)
      return stateless_random_ops.stateless_random_normal(*args, **kwargs)
    else:
      return random_ops.random_normal(*args, **kwargs)



class SparseStaticOptimizer(SparseSETOptimizer):
  """Sparse optimizer that re-initializes weak connections during training.

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    momentum: float, for the exponentialy moving average.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               name='SparseStaticOptimizer',
               stateless_seed_offset=0):
    super(SparseStaticOptimizer, self).__init__(
        optimizer,
        begin_step,
        end_step,
        frequency,
        drop_fraction=drop_fraction,
        drop_fraction_anneal=drop_fraction_anneal,
        grow_init=grow_init,
        use_locking=use_locking,
        name=name,
        stateless_seed_offset=stateless_seed_offset)

  def generic_mask_update(self, mask, weights, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    masked_weights = mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=hash(weights.name + 'drop'))
    # Revive n_prune many connections using momentum.
    score_grow = mask
    return self._get_update_op(
        score_drop, score_grow, mask, weights, reinit_when_same=True)


class SparseMomentumOptimizer(SparseSETOptimizer):
  """Sparse optimizer that grows connections with the expected gradients.

  A simplified implementation of Momentum based sparse optimizer. No
  redistribution of sparsity.
  Original implementation:
  https://github.com/TimDettmers/sparse_learning/blob/master/mnist_cifar/main.py

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    momentum: float, for the exponentialy moving average.
    use_tpu: bool, if true the masked_gradients are aggregated.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               momentum=0.9,
               use_tpu=False,
               name='SparseMomentumOptimizer',
               stateless_seed_offset=0):
    super(SparseMomentumOptimizer, self).__init__(
        optimizer,
        begin_step,
        end_step,
        frequency,
        drop_fraction=drop_fraction,
        drop_fraction_anneal=drop_fraction_anneal,
        grow_init=grow_init,
        use_locking=use_locking,
        name='SparseMomentumOptimizer',
        stateless_seed_offset=stateless_seed_offset)
    self._ema_grads = moving_averages.ExponentialMovingAverage(decay=momentum)
    self._use_tpu = use_tpu

  def set_masked_grads(self, grads, weights):
    if self._use_tpu:
      grads = [tpu_ops.cross_replica_sum(g) for g in grads]
    self._masked_grads = grads
    # Using names since better to hash.
    self._weight2masked_grads = {w.name: m for w, m in zip(weights, grads)}

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    grads_and_vars = self._optimizer.compute_gradients(loss, **kwargs)
    # Need to update the EMA of the masked_weights. This is a bit hacky and
    # might not work as expected if the gradients are not applied after every
    # calculation. However, it should be fine if only .minimize() call is used.
    masked_grads_vars = self._optimizer.compute_gradients(
        loss, var_list=self.get_masked_weights())
    masked_grads = [g for g, _ in masked_grads_vars]
    self.set_masked_grads(masked_grads, self.get_weights())
    return grads_and_vars

  def _before_apply_gradients(self, grads_and_vars):
    """Updates momentum before updating the weights with gradient."""
    return self._ema_grads.apply(self._masked_grads)

  def generic_mask_update(self, mask, weights, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    casted_mask = math_ops.cast(mask, dtypes.float32)
    masked_weights = casted_mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=hash(weights.name + 'drop'))
    # Revive n_prune many connections using momentum.
    masked_grad = self._weight2masked_grads[weights.name]
    score_grow = math_ops.abs(self._ema_grads.average(masked_grad))
    return self._get_update_op(score_drop, score_grow, mask, weights)


class SparseRigLOptimizer(SparseSETOptimizer):
  """Sparse optimizer that grows connections with the pre-removal gradients.

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    init_avg_scale: float, used to scale the gradient when initializing the,
      momentum values of new connections. We hope this will improve training,
      compare to starting from 0 for the new connections. Set this to something
      between 0 and 1 / (1 - momentum). This is because in the current
      implementation of MomentumOptimizer, aggregated values converge to 1 / (1
      - momentum) with constant gradients.
    use_tpu: bool, if true the masked_gradients are aggregated.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               initial_acc_scale=0.,
               use_tpu=False,
               name='SparseRigLOptimizer',
               stateless_seed_offset=0):
    super(SparseRigLOptimizer, self).__init__(
        optimizer,
        begin_step,
        end_step,
        frequency,
        drop_fraction=drop_fraction,
        drop_fraction_anneal=drop_fraction_anneal,
        grow_init=grow_init,
        use_locking=use_locking,
        name='SparseRigLOptimizer',
        stateless_seed_offset=stateless_seed_offset)
    self._initial_acc_scale = initial_acc_scale
    self._use_tpu = use_tpu

  def set_masked_grads(self, grads, weights):
    if self._use_tpu:
      grads = [tpu_ops.cross_replica_sum(g) for g in grads]
    self._masked_grads = grads
    # Using names since better to hash.
    self._weight2masked_grads = {w.name: m for w, m in zip(weights, grads)}

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    grads_and_vars = self._optimizer.compute_gradients(loss, **kwargs)
    masked_grads_vars = self._optimizer.compute_gradients(
        loss, var_list=self.get_masked_weights())
    masked_grads = [g for g, _ in masked_grads_vars]
    self.set_masked_grads(masked_grads, self.get_weights())
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    pre_op = self._before_apply_gradients(grads_and_vars)
    with ops.control_dependencies([pre_op]):
      # Call this to create slots.
      _ = self._optimizer.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)

      def apply_gradient_op():
        optimizer_update = self._optimizer.apply_gradients(
            grads_and_vars, global_step=global_step, name=name)
        return optimizer_update

      # We get the default one after calling the super.apply_gradient(), since
      # we want to preserve original behavior of the optimizer: don't increment
      # anything if no global_step is passed. But we need the global step for
      # the mask_update.
      global_step = (
          global_step if global_step is not None else
          training_util.get_or_create_global_step())
      self._global_step = global_step
      return self.cond_mask_update_op(global_step, apply_gradient_op)

  def generic_mask_update(self, mask, weights, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    casted_mask = math_ops.cast(mask, dtype=dtypes.float32)
    masked_weights = casted_mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=hash(weights.name + 'drop'))
    # Revive n_prune many connections using gradient.
    score_grow = math_ops.abs(self._weight2masked_grads[weights.name])
    with ops.control_dependencies([score_grow]):
      return self._get_update_op(score_drop, score_grow, mask, weights)

  def get_grow_tensor(self, weight, method):
    if method.startswith('grad_scale'):
      masked_grad = self._weight2masked_grads[weight.name]
      divisor = extract_number(method)
      grow_tensor = masked_grad / divisor
    elif method.startswith('grad_sign'):
      masked_grad_sign = math_ops.sign(self._weight2masked_grads[weight.name])
      divisor = extract_number(method)
      grow_tensor = masked_grad_sign / divisor
    else:
      grow_tensor = super(SparseRigLOptimizer,
                          self).get_grow_tensor(weight, method)
    return grow_tensor

  def reset_momentum(self, weights, new_connections):
    reset_ops = []
    for s_name in self._optimizer.get_slot_names():
      # Momentum variable for example, we reset the aggregated values to zero.
      optim_var = self._optimizer.get_slot(weights, s_name)
      accum_grad = (
          self._weight2masked_grads[weights.name] * self._initial_acc_scale)
      new_values = array_ops.where(new_connections, accum_grad, optim_var)
      reset_ops.append(state_ops.assign(optim_var, new_values))
    return control_flow_ops.group(reset_ops)


class SparseSnipOptimizer(tf_optimizer.Optimizer):
  """Implementation of dynamic sparsity optimizers.

  Implementation of Snip
  https://arxiv.org/abs/1810.02340

  Attributes:
    optimizer: tf.train.Optimizer
    default_sparsity: float, between 0 and 1.
    mask_init_method: str, used to determine mask initializations.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    use_locking: bool, passed to the super.
    use_tpu: bool, if true the masked_gradients are aggregated.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               default_sparsity,
               mask_init_method,
               custom_sparsity_map=None,
               use_locking=False,
               use_tpu=False,
               name='SparseSnipOptimizer'):
    super(SparseSnipOptimizer, self).__init__(use_locking, name)
    if not custom_sparsity_map:
      custom_sparsity_map = {}
    self._optimizer = optimizer
    self._use_tpu = use_tpu
    self._default_sparsity = default_sparsity
    self._mask_init_method = mask_init_method
    self._custom_sparsity_map = custom_sparsity_map
    self.is_snipped = variable_scope.get_variable(
        'is_snipped', initializer=lambda: False, trainable=False)

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    return self._optimizer.compute_gradients(loss, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """

    def apply_gradient_op():
      return self._optimizer.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)

    maybe_reduce = lambda x: x
    if self._use_tpu:
      maybe_reduce = tpu_ops.cross_replica_sum
    grads_and_vars_dict = {
        re.findall('(.+)/weights:0', var.name)[0]: (maybe_reduce(grad), var)
        for grad, var in grads_and_vars
        if var.name.endswith('weights:0')
    }

    def snip_fn(mask, sparsity, dtype):
      """Creates a random sparse mask with deterministic sparsity.

      Args:
        mask: tf.Tensor, used to obtain correct corresponding gradient.
        sparsity: float, between 0 and 1.
        dtype: tf.dtype, type of the return value.

      Returns:
        tf.Tensor
      """
      del dtype
      var_name = sparse_utils.mask_extract_name_fn(mask.name)
      g, v = grads_and_vars_dict[var_name]
      score_drop = math_ops.abs(g * v)
      n_total = np.prod(score_drop.shape.as_list())
      n_prune = sparse_utils.get_n_zeros(n_total, sparsity)
      n_keep = n_total - n_prune

      # Sort the entire array since the k needs to be constant for TPU.
      _, sorted_indices = nn_ops.top_k(
          array_ops.reshape(score_drop, [-1]), k=n_total)
      sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
      # We will have zeros after having `n_keep` many ones.
      new_values = array_ops.where(
          math_ops.range(n_total) < n_keep,
          array_ops.ones_like(sorted_indices, dtype=mask.dtype),
          array_ops.zeros_like(sorted_indices, dtype=mask.dtype))
      new_mask = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                      new_values.shape)
      return array_ops.reshape(new_mask, mask.shape)

    def snip_op():
      all_masks = pruning.get_masks()
      assigner = sparse_utils.get_mask_init_fn(
          all_masks,
          self._mask_init_method,
          self._default_sparsity,
          self._custom_sparsity_map,
          mask_fn=snip_fn)
      with ops.control_dependencies([assigner]):
        assign_op = state_ops.assign(
            self.is_snipped, True, name='assign_true_after_snipped')
      return assign_op

    maybe_snip_op = control_flow_ops.cond(
        math_ops.logical_and(
            math_ops.equal(global_step, 0),
            math_ops.logical_not(self.is_snipped)), snip_op, apply_gradient_op)

    return maybe_snip_op


class SparseDNWOptimizer(tf_optimizer.Optimizer):
  """Implementation of DNW optimizer.

  Implementation of DNW.
  See https://arxiv.org/pdf/1906.00586.pdf
  This optimizer ensures the mask is updated at every iteration, according to
  the current set of weights. It uses dense gradient to update weights.

  Attributes:
    optimizer: tf.train.Optimizer
    default_sparsity: float, between 0 and 1.
    mask_init_method: str, used to determine mask initializations.
    custom_sparsity_map: dict, <str, float> key/value pairs where the mask
      correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
    use_tpu: bool, if true the masked_gradients are aggregated.
    use_locking: bool, passed to the super.
    name: bool, passed to the super.
  """

  def __init__(self,
               optimizer,
               default_sparsity,
               mask_init_method,
               custom_sparsity_map=None,
               use_tpu=False,
               use_locking=False,
               name='SparseDNWOptimizer'):
    super(SparseDNWOptimizer, self).__init__(use_locking, name)
    self._optimizer = optimizer
    self._use_tpu = use_tpu
    self._default_sparsity = default_sparsity
    self._mask_init_method = mask_init_method
    self._custom_sparsity_map = custom_sparsity_map

  def compute_gradients(self, loss, var_list=None, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    # Replace masked variables with masked_weights so that the gradient is dense
    # and not masked
    if var_list is None:
      var_list = (
          variables.trainable_variables() +
          ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
    var_list = self.replace_with_masked_weights(var_list)
    grads_and_vars = self._optimizer.compute_gradients(
        loss, var_list=var_list, **kwargs)
    return self.replace_masked_weights(grads_and_vars)

  def replace_with_masked_weights(self, var_list):
    """Replaces masked variables with masked weights."""
    weight2masked_weights = {
        w.name: mw
        for w, mw in zip(self.get_weights(), self.get_masked_weights())
    }
    updated_var_list = [weight2masked_weights.get(w.name, w) for w in var_list]
    return updated_var_list

  def replace_masked_weights(self, grads_and_vars):
    """Replaces masked weight tensords with weight variables."""
    masked_weights2weight = {
        mw.name: w
        for w, mw in zip(self.get_weights(), self.get_masked_weights())
    }
    updated_grads_and_vars = [
        (g, masked_weights2weight.get(w.name, w)) for g, w in grads_and_vars
    ]
    return updated_grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    optimizer_update = self._optimizer.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)
    vars_dict = {
        re.findall('(.+)/weights:0', var.name)[0]: var
        for var in self.get_weights()
    }

    def dnw_fn(mask, sparsity, dtype):
      """Creates a mask with smallest magnitudes with deterministic sparsity.

      Args:
        mask: tf.Tensor, used to obtain correct corresponding gradient.
        sparsity: float, between 0 and 1.
        dtype: tf.dtype, type of the return value.

      Returns:
        tf.Tensor
      """
      del dtype
      var_name = sparse_utils.mask_extract_name_fn(mask.name)
      v = vars_dict[var_name]
      score_drop = math_ops.abs(v)
      n_total = np.prod(score_drop.shape.as_list())
      n_prune = sparse_utils.get_n_zeros(n_total, sparsity)
      n_keep = n_total - n_prune

      # Sort the entire array since the k needs to be constant for TPU.
      _, sorted_indices = nn_ops.top_k(
          array_ops.reshape(score_drop, [-1]), k=n_total)
      sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
      # We will have zeros after having `n_keep` many ones.
      new_values = array_ops.where(
          math_ops.range(n_total) < n_keep,
          array_ops.ones_like(sorted_indices, dtype=mask.dtype),
          array_ops.zeros_like(sorted_indices, dtype=mask.dtype))
      new_mask = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                      new_values.shape)
      return array_ops.reshape(new_mask, mask.shape)

    with ops.control_dependencies([optimizer_update]):
      all_masks = self.get_masks()
      mask_update_op = sparse_utils.get_mask_init_fn(
          all_masks,
          self._mask_init_method,
          self._default_sparsity,
          self._custom_sparsity_map,
          mask_fn=dnw_fn)

    return mask_update_op

  def get_weights(self):
    return pruning.get_weights()

  def get_masks(self):
    return pruning.get_masks()

  def get_masked_weights(self):
    return pruning.get_masked_weights()



class SparseCTEOptimizer(tf_optimizer.Optimizer):
  """Implementation of dynamic sparsity optimizers.

  Implementation of SET.
  See https://www.nature.com/articles/s41467-018-04316-3
  This optimizer wraps a regular optimizer and performs updates on the masks
  according to schedule given.

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    name: bool, passed to the super.
    use_stateless: bool, if True stateless operations are used. This is
      important for multi-worker jobs not to diverge.
    stateless_seed_offset: int, added to the seed of stateless operations. Use
      this to create randomness without divergence across workers.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               name='SparseCTEOptimizer',
               use_stateless=True,
               stateless_seed_offset=0):
    super(SparseCTEOptimizer, self).__init__(use_locking, name)
    self._optimizer = optimizer
    self._grow_init = grow_init
    self._drop_fraction_anneal = drop_fraction_anneal
    self._drop_fraction_initial_value = ops.convert_to_tensor(
        float(drop_fraction),
        name='%s_drop_fraction' % self._drop_fraction_anneal)
    self._begin_step = ops.convert_to_tensor(begin_step, name='begin_step')
    self._end_step = ops.convert_to_tensor(end_step, name='end_step')
    self._frequency = ops.convert_to_tensor(frequency, name='frequency')
    self._frequency_val = frequency
    self._use_stateless = use_stateless
    self._stateless_seed_offset = stateless_seed_offset
    self._score_cosine = []
    
    #self._custom_indices= tf.zeros(10000)
    cnt = 0
    for mask, weights in zip(self.get_masks(), self.get_weights()):
        print("****----------------------------------****")
        print(weights)
        print(weights.name)
        print(weights.shape, flush=True)
        nm = "sim_"+weights.name.replace("/","_")
        nm = nm.replace(":","_")
        print(nm, flush=True)
        cnt += 1
        self._score_cosine.append(tf.Variable(tf.zeros(weights.shape), trainable =False,
                                            dtype=tf.float32, 
                                            name="cosine_score"+str(cnt)))
        print("done", flush=True)
    

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    result = self._optimizer.compute_gradients(loss, **kwargs)
    return result

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    pre_op = self._before_apply_gradients(grads_and_vars)
    with ops.control_dependencies([pre_op]):
      optimizer_update = self._optimizer.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)
    # We get the default one after calling the super.apply_gradient(), since
    # we want to preserve original behavior of the optimizer: don't increment
    # anything if no global_step is passed. But we need the global step for
    # the mask_update.
    global_step = (
        global_step if global_step is not None else
        training_util.get_or_create_global_step())
    self._global_step = global_step
    with ops.control_dependencies([optimizer_update]):
      return self.cond_mask_update_op(global_step, control_flow_ops.no_op)

  def _before_apply_gradients(self, grads_and_vars):
    """Called before applying gradients."""
    return control_flow_ops.no_op('before_apply_grad')

  def cond_mask_update_op(self, global_step, false_branch):
    """Creates the conditional mask update operation.

    All masks are updated when it is an update iteration
    (checked by self.is_mask_update_iter()).
    Arguments:
      global_step: tf.Variable, current training iteration.
      false_branch: function, called when it is not a mask update iteration.

    Returns:
      conditional update operation
    """
    # Initializing to -freq so that last_update_step+freq=0. This enables early
    # mask_updates.
    last_update_step = variable_scope.get_variable(
        'last_mask_update_step', [],
        initializer=init_ops.constant_initializer(
            -self._frequency_val, dtype=global_step.dtype),
        trainable=False,
        dtype=global_step.dtype)

    def mask_update_op():
      update_ops = []
      cnt = 0
      for mask, weights in zip(self.get_masks(), self.get_weights()):
        update_ops.append(self.generic_mask_update(mask, weights, cnt))
        cnt+=1

      with ops.control_dependencies(update_ops):
        assign_op = state_ops.assign(
            last_update_step, global_step, name='last_mask_update_step_assign')
        with ops.control_dependencies([assign_op]):
          return control_flow_ops.no_op('mask_update')

    maybe_update = control_flow_ops.cond(
        self.is_mask_update_iter(global_step, last_update_step), mask_update_op,
        false_branch)
    return maybe_update

  def get_weights(self):
    return pruning.get_weights()

  def get_masks(self):
    return pruning.get_masks()

  def get_masked_weights(self):
    return pruning.get_masked_weights()

  def is_mask_update_iter(self, global_step, last_update_step):
    """Function for checking if the current step is a mask update step.

    It also creates the drop_fraction op and assigns it to the self object.

    Args:
      global_step: tf.Variable(int), current training step.
      last_update_step: tf.Variable(int), holding the last iteration the mask is
        updated. Used to determine whether current iteration is a mask update
        step.

    Returns:
      bool, whether the current iteration is a mask_update step.
    """
    gs_dtype = global_step.dtype
    self._begin_step = math_ops.cast(self._begin_step, gs_dtype)
    self._end_step = math_ops.cast(self._end_step, gs_dtype)
    self._frequency = math_ops.cast(self._frequency, gs_dtype)
    is_step_within_update_range = math_ops.logical_and(
        math_ops.greater_equal(global_step, self._begin_step),
        math_ops.logical_or(
            math_ops.less_equal(global_step, self._end_step),
            # If _end_step is negative, we never stop updating the mask.
            # In other words we update the mask with given frequency until the
            # training ends.
            math_ops.less(self._end_step, 0)))
    is_update_step = math_ops.less_equal(
        math_ops.add(last_update_step, self._frequency), global_step)
    is_mask_update_iter_op = math_ops.logical_and(is_step_within_update_range,
                                                  is_update_step)
    self.drop_fraction = self.get_drop_fraction(global_step,
                                                is_mask_update_iter_op)
    return is_mask_update_iter_op

  def get_drop_fraction(self, global_step, is_mask_update_iter_op):
    """Returns a constant or annealing drop_fraction op."""
    if self._drop_fraction_anneal == 'constant':
      drop_frac = self._drop_fraction_initial_value
    elif self._drop_fraction_anneal == 'cosine':
      decay_steps = self._end_step - self._begin_step
      drop_frac = learning_rate_decay.cosine_decay(
          self._drop_fraction_initial_value,
          global_step,
          decay_steps,
          name='cosine_drop_fraction')
    elif self._drop_fraction_anneal.startswith('exponential'):
      exponent = extract_number(self._drop_fraction_anneal)
      div_dtype = self._drop_fraction_initial_value.dtype
      power = math_ops.divide(
          math_ops.cast(global_step - self._begin_step, div_dtype),
          math_ops.cast(self._end_step - self._begin_step, div_dtype),
      )
      drop_frac = math_ops.multiply(
          self._drop_fraction_initial_value,
          math_ops.pow(1 - power, exponent),
          name='%s_drop_fraction' % self._drop_fraction_anneal)
    else:
      raise ValueError('drop_fraction_anneal: %s is not valid' %
                       self._drop_fraction_anneal)
    return array_ops.where(is_mask_update_iter_op, drop_frac,
                           array_ops.zeros_like(drop_frac))

  def generic_mask_update(self, mask, weights, cnt = 0, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    masked_weights = mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=(hash(weights.name + 'drop')))
    # Randomly revive n_prune many connections from non-existing connections.
    #masked_weights = mask * weights
    #score_drop = math_ops.abs(masked_weights)
    
    
    #score_grow = self._score_grow
    if cnt <5:
        score_grow = self._score_cosine[cnt]
    else:
        score_grow = self._random_uniform(
       weights.shape, seed=hash(weights.name + 'grow'))
    return self._get_update_op(score_drop, score_grow, mask, weights)

  def _get_update_op(self,
                     score_drop,
                     score_grow,
                     mask,
                     weights,
                     reinit_when_same=False):
    """Prunes+grows connections, all tensors same shape."""
    old_dtype = mask.dtype
    mask_casted = math_ops.cast(mask, dtypes.float32)
    n_total = array_ops.size(score_drop)
    n_ones = math_ops.cast(math_ops.reduce_sum(mask_casted), dtype=dtypes.int32)
    n_prune = math_ops.cast(
        math_ops.cast(n_ones, dtype=dtypes.float32) * self.drop_fraction,
        dtypes.int32)
    n_keep = n_ones - n_prune

    # Sort the entire array since the k needs to be constant for TPU.
    _, sorted_indices = nn_ops.top_k(
        array_ops.reshape(score_drop, [-1]), k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    # We will have zeros after having `n_keep` many ones.
    new_values = array_ops.where(
        math_ops.range(n_total) < n_keep,
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask1 = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
    
    # Flatten the scores
    score_grow = array_ops.reshape(score_grow, [-1])
    # Set scores of the enabled connections(ones) to min(s) - 1, so that they
    # have the lowest scores.
    score_grow_lifted = array_ops.where(
        math_ops.equal(mask1, 1),
        array_ops.ones_like(mask1) * (math_ops.reduce_min(score_grow) - 1),
        score_grow)
    _, sorted_indices = nn_ops.top_k(score_grow_lifted, k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    new_values = array_ops.where(
        math_ops.range(n_total) < n_prune,
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask2 = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
    # Ensure masks are disjoint.
    assert_op = control_flow_ops.Assert(
        math_ops.equal(math_ops.reduce_sum(mask1 * mask2), 0.), [mask1, mask2])

    with ops.control_dependencies([assert_op]):
      # Let's set the weights of the growed connections.
      mask2_reshaped = array_ops.reshape(mask2, mask.shape)
    # Set the values of the new connections.
    grow_tensor = self.get_grow_tensor(weights, self._grow_init)
    if reinit_when_same:
      # If dropped and grown, we re-initialize.
      new_connections = math_ops.equal(mask2_reshaped, 1)
    else:
      new_connections = math_ops.logical_and(
          math_ops.equal(mask2_reshaped, 1), math_ops.equal(mask_casted, 0))
    new_weights = array_ops.where(new_connections, grow_tensor, weights)
    weights_update = state_ops.assign(weights, new_weights)
    # Ensure there is no momentum value for new connections
    reset_op = self.reset_momentum(weights, new_connections)

    with ops.control_dependencies([weights_update, reset_op]):
      mask_combined = array_ops.reshape(mask1 + mask2, mask.shape)
    mask_combined = math_ops.cast(mask_combined, dtype=old_dtype)
    new_mask = state_ops.assign(mask, mask_combined)
    return new_mask

  def reset_momentum(self, weights, new_connections):
    reset_ops = []
    for s_name in self._optimizer.get_slot_names():
      # Momentum variable for example, we reset the aggregated values to zero.
      optim_var = self._optimizer.get_slot(weights, s_name)
      new_values = array_ops.where(new_connections,
                                   array_ops.zeros_like(optim_var), optim_var)
      reset_ops.append(state_ops.assign(optim_var, new_values))
    return control_flow_ops.group(reset_ops)

  def get_grow_tensor(self, weights, method):
    """Different ways to initialize new connections.

    Args:
      weights: tf.Tensor or Variable.
      method: str, available options: 'zeros', 'random_normal', 'random_uniform'
        and 'initial_value'

    Returns:
      tf.Tensor same shape and type as weights.

    Raises:
      ValueError, when the method is not valid.
    """
    if not isinstance(method, six.string_types):
      raise ValueError('Grow-Init: %s is not a string' % method)

    if method == 'zeros':
      grow_tensor = array_ops.zeros_like(weights, dtype=weights.dtype)
    elif method.startswith('initial_dist'):
      original_shape = weights.initial_value.shape
      divisor = extract_number(method)
      grow_tensor = array_ops.reshape(
          random_ops.random_shuffle(
              array_ops.reshape(weights.initial_value, [-1])),
          original_shape) / divisor
    elif method.startswith('random_normal1'):
      stddev = math_ops.reduce_std(weights)
      divisor = extract_number(method)
      grow_tensor = self._random_normal(
          weights.shape,
          mean=0,
          stddev=1, #stddev,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_n')) / 10 #divisor
    elif method.startswith('random_normal2'):
      stddev = math_ops.reduce_std(weights)
      divisor = extract_number(method)
      grow_tensor = self._random_normal(
          weights.shape,
          mean=0,
          stddev=stddev,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_n'))  /2
          
    elif method.startswith('random_uniform'):
      mean = math_ops.reduce_mean(math_ops.abs(weights))
      mean = np.sqrt(6. / weights.get_shape().as_list()[0])
      divisor = extract_number(method)
      grow_tensor = self._random_uniform(
          weights.shape,
          minval=-mean,
          maxval=mean,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_u')) / divisor
    else:
      raise ValueError('Grow-Init: %s is not a valid option.' % method)
    return grow_tensor

  def _random_uniform(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = math_ops.cast(
          array_ops.stack([c_seed, self._global_step]), dtypes.int32)
      return stateless_random_ops.stateless_random_uniform(*args, **kwargs)
    else:
      return random_ops.random_uniform(*args, **kwargs)

  def _random_normal(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = math_ops.cast(
          array_ops.stack([c_seed, self._global_step]), dtypes.int32)
      return stateless_random_ops.stateless_random_normal(*args, **kwargs)
    else:
      return random_ops.random_normal(*args, **kwargs)







class SparseCTREsimOptimizer(tf_optimizer.Optimizer):
  """Implementation of dynamic sparsity optimizers.

  Implementation of SET.
  See https://www.nature.com/articles/s41467-018-04316-3
  This optimizer wraps a regular optimizer and performs updates on the masks
  according to schedule given.

  Attributes:
    optimizer: tf.train.Optimizer
    begin_step: int, first iteration where masks are updated.
    end_step: int, iteration after which no mask is updated.
    frequency: int, of mask update operations.
    drop_fraction: float, of connections to drop during each update.
    drop_fraction_anneal: str or None, if supplied used to anneal the drop
      fraction.
    use_locking: bool, passed to the super.
    grow_init: str, name of the method used to initialize new connections.
    name: bool, passed to the super.
    use_stateless: bool, if True stateless operations are used. This is
      important for multi-worker jobs not to diverge.
    stateless_seed_offset: int, added to the seed of stateless operations. Use
      this to create randomness without divergence across workers.
  """

  def __init__(self,
               optimizer,
               begin_step,
               end_step,
               frequency,
               drop_fraction=0.1,
               drop_fraction_anneal='constant',
               use_locking=False,
               grow_init='zeros',
               name='SparseCTREsimOptimizer',
               use_stateless=True,
               stateless_seed_offset=0):
    super(SparseCTREsimOptimizer, self).__init__(use_locking, name)
    self._optimizer = optimizer
    self._grow_init = grow_init
    self._drop_fraction_anneal = drop_fraction_anneal
    self._drop_fraction_initial_value = ops.convert_to_tensor(
        float(drop_fraction),
        name='%s_drop_fraction' % self._drop_fraction_anneal)
    self._begin_step = ops.convert_to_tensor(begin_step, name='begin_step')
    self._end_step = ops.convert_to_tensor(end_step, name='end_step')
    self._frequency = ops.convert_to_tensor(frequency, name='frequency')
    self._frequency_val = frequency
    self._use_stateless = use_stateless
    self._stateless_seed_offset = stateless_seed_offset
    self._score_cosine = []
    
    #self._custom_indices= tf.zeros(10000)
    cnt = 0
    for mask, weights in zip(self.get_masks(), self.get_weights()):
        print("****----------------------------------****")
        print(weights)
        print(weights.name)
        print(weights.shape, flush=True)
        nm = "sim_"+weights.name.replace("/","_")
        nm = nm.replace(":","_")
        print(nm, flush=True)
        cnt += 1
        self._score_cosine.append(tf.Variable(tf.zeros(weights.shape), trainable =False,
                                            dtype=tf.float32, 
                                            name="cosine_score"+str(cnt)))
        print("done", flush=True)
    

  def compute_gradients(self, loss, **kwargs):
    """Wraps the compute gradient of passed optimizer."""
    result = self._optimizer.compute_gradients(loss, **kwargs)
    return result

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Wraps the original apply_gradient of the optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    pre_op = self._before_apply_gradients(grads_and_vars)
    with ops.control_dependencies([pre_op]):
      optimizer_update = self._optimizer.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)
    # We get the default one after calling the super.apply_gradient(), since
    # we want to preserve original behavior of the optimizer: don't increment
    # anything if no global_step is passed. But we need the global step for
    # the mask_update.
    global_step = (
        global_step if global_step is not None else
        training_util.get_or_create_global_step())
    self._global_step = global_step
    with ops.control_dependencies([optimizer_update]):
      return self.cond_mask_update_op(global_step, control_flow_ops.no_op)

  def _before_apply_gradients(self, grads_and_vars):
    """Called before applying gradients."""
    return control_flow_ops.no_op('before_apply_grad')

  def cond_mask_update_op(self, global_step, false_branch):
    """Creates the conditional mask update operation.

    All masks are updated when it is an update iteration
    (checked by self.is_mask_update_iter()).
    Arguments:
      global_step: tf.Variable, current training iteration.
      false_branch: function, called when it is not a mask update iteration.

    Returns:
      conditional update operation
    """
    # Initializing to -freq so that last_update_step+freq=0. This enables early
    # mask_updates.
    last_update_step = variable_scope.get_variable(
        'last_mask_update_step', [],
        initializer=init_ops.constant_initializer(
            -self._frequency_val, dtype=global_step.dtype),
        trainable=False,
        dtype=global_step.dtype)

    def mask_update_op():
      update_ops = []
      cnt = 0
      for mask, weights in zip(self.get_masks(), self.get_weights()):
        update_ops.append(self.generic_mask_update(mask, weights, cnt))
        cnt+=1

      with ops.control_dependencies(update_ops):
        assign_op = state_ops.assign(
            last_update_step, global_step, name='last_mask_update_step_assign')
        with ops.control_dependencies([assign_op]):
          return control_flow_ops.no_op('mask_update')

    maybe_update = control_flow_ops.cond(
        self.is_mask_update_iter(global_step, last_update_step), mask_update_op,
        false_branch)
    return maybe_update

  def get_weights(self):
    return pruning.get_weights()

  def get_masks(self):
    return pruning.get_masks()

  def get_masked_weights(self):
    return pruning.get_masked_weights()

  def is_mask_update_iter(self, global_step, last_update_step):
    """Function for checking if the current step is a mask update step.

    It also creates the drop_fraction op and assigns it to the self object.

    Args:
      global_step: tf.Variable(int), current training step.
      last_update_step: tf.Variable(int), holding the last iteration the mask is
        updated. Used to determine whether current iteration is a mask update
        step.

    Returns:
      bool, whether the current iteration is a mask_update step.
    """
    gs_dtype = global_step.dtype
    self._begin_step = math_ops.cast(self._begin_step, gs_dtype)
    self._end_step = math_ops.cast(self._end_step, gs_dtype)
    self._frequency = math_ops.cast(self._frequency, gs_dtype)
    is_step_within_update_range = math_ops.logical_and(
        math_ops.greater_equal(global_step, self._begin_step),
        math_ops.logical_or(
            math_ops.less_equal(global_step, self._end_step),
            # If _end_step is negative, we never stop updating the mask.
            # In other words we update the mask with given frequency until the
            # training ends.
            math_ops.less(self._end_step, 0)))
    is_update_step = math_ops.less_equal(
        math_ops.add(last_update_step, self._frequency), global_step)
    is_mask_update_iter_op = math_ops.logical_and(is_step_within_update_range,
                                                  is_update_step)
    self.drop_fraction = self.get_drop_fraction(global_step,
                                                is_mask_update_iter_op)
    return is_mask_update_iter_op

  def get_drop_fraction(self, global_step, is_mask_update_iter_op):
    """Returns a constant or annealing drop_fraction op."""
    if self._drop_fraction_anneal == 'constant':
      drop_frac = self._drop_fraction_initial_value
    elif self._drop_fraction_anneal == 'cosine':
      decay_steps = self._end_step - self._begin_step
      drop_frac = learning_rate_decay.cosine_decay(
          self._drop_fraction_initial_value,
          global_step,
          decay_steps,
          name='cosine_drop_fraction')
    elif self._drop_fraction_anneal.startswith('exponential'):
      exponent = extract_number(self._drop_fraction_anneal)
      div_dtype = self._drop_fraction_initial_value.dtype
      power = math_ops.divide(
          math_ops.cast(global_step - self._begin_step, div_dtype),
          math_ops.cast(self._end_step - self._begin_step, div_dtype),
      )
      drop_frac = math_ops.multiply(
          self._drop_fraction_initial_value,
          math_ops.pow(1 - power, exponent),
          name='%s_drop_fraction' % self._drop_fraction_anneal)
    else:
      raise ValueError('drop_fraction_anneal: %s is not valid' %
                       self._drop_fraction_anneal)
    return array_ops.where(is_mask_update_iter_op, drop_frac,
                           array_ops.zeros_like(drop_frac))

  def generic_mask_update(self, mask, weights, cnt = 0, noise_std=1e-5):
    """True branch of the condition, updates the mask."""
    # Ensure that the weights are masked.
    masked_weights = mask * weights
    score_drop = math_ops.abs(masked_weights)
    # Add noise for slight bit of randomness.
    score_drop += self._random_normal(
        score_drop.shape,
        stddev=noise_std,
        dtype=score_drop.dtype,
        seed=(hash(weights.name + 'drop')))
    # Randomly revive n_prune many connections from non-existing connections.
    #masked_weights = mask * weights
    #score_drop = math_ops.abs(masked_weights)
    
    
    #score_grow = self._score_grow
    if cnt <5:
        score_grow = self._score_cosine[cnt]
    else:
        score_grow = self._random_uniform(
       weights.shape, seed=hash(weights.name + 'grow'))
    return self._get_update_op(score_drop, score_grow, mask, weights)

  def _get_update_op(self,
                     score_drop,
                     score_grow,
                     mask,
                     weights,
                     reinit_when_same=False):
    """Prunes+grows connections, all tensors same shape."""
    old_dtype = mask.dtype
    mask_casted = math_ops.cast(mask, dtypes.float32)
    n_total = array_ops.size(score_drop)
    n_ones = math_ops.cast(math_ops.reduce_sum(mask_casted), dtype=dtypes.int32)
    n_prune = math_ops.cast(
        math_ops.cast(n_ones, dtype=dtypes.float32) * self.drop_fraction,
        dtypes.int32)
    n_keep = n_ones - n_prune

    # Sort the entire array since the k needs to be constant for TPU.
    _, sorted_indices = nn_ops.top_k(
        array_ops.reshape(score_drop, [-1]), k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    # We will have zeros after having `n_keep` many ones.
    new_values = array_ops.where(
        math_ops.range(n_total) < n_keep,
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask1 = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
                                 
        
        
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #$$$$$ Me
    sorted_indices = tf.range(tf.size(mask1))
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    mask_casted_reshaped = array_ops.reshape(mask_casted, mask1.shape)
    #print("mask_casted_reshaped ", mask_casted_reshaped)
    #print("mask1", mask1)
    values_deleted = array_ops.where(
        tf.math.logical_and(tf.equal(mask_casted_reshaped,1), tf.equal(mask1,0)), #(mask_casted_reshaped > 0) and (mask1 < 1),
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask_deleted = array_ops.scatter_nd(sorted_indices_ex, values_deleted,
                                 new_values.shape)
    n_deleted = math_ops.cast(math_ops.reduce_sum(mask_deleted), dtype=dtypes.int32)
    ### mask deleted connections
    #score_grow =  tf.Print(score_grow , ["n_deleted = ", n_deleted, " - n_ones = ", n_ones,\
    #                                          "+++ ",  self.drop_fraction , n_deleted/n_ones ])
    
    #score_grow =  tf.Print(score_grow , ["intersect deleted and mask 1= ", math_ops.reduce_sum(mask_deleted*mask1), " - n_ones = ", n_ones,\
    #                                          "+++ intersect deleted and mask_casted", math_ops.reduce_sum(mask_deleted*mask_casted_reshaped) ])
    
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    
    
    # Flatten the scores
    score_grow = array_ops.reshape(score_grow, [-1])
    # Set scores of the enabled connections(ones) to min(s) - 1, so that they
    # have the lowest scores.
    score_grow_lifted = array_ops.where(
        math_ops.equal(mask1, 1),
        array_ops.ones_like(mask1) * (math_ops.reduce_min(score_grow) - 1),
        score_grow)
    _, sorted_indices = nn_ops.top_k(score_grow_lifted, k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    new_values = array_ops.where(
        math_ops.range(n_total) < n_prune,
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask2 = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
    
    #mask2 =  tf.Print(mask2 , ["mask2 = ", mask2])
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #$$$$$ Me
    ### mask readded connections
    sorted_indices = tf.range(tf.size(mask2))
    sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    values_readded = array_ops.where(
        tf.math.logical_and(tf.equal(mask2,1), tf.equal(mask_deleted,1)), #(mask_casted_reshaped > 0) and (mask1 < 1),
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    mask_readded = array_ops.scatter_nd(sorted_indices_ex, values_readded,
                                 new_values.shape)
    n_readded = math_ops.cast(math_ops.reduce_sum(mask_readded), dtype=dtypes.int32)
    
    #mask2 =  tf.Print(mask2 , ["n_readded = ", n_readded])
    
    
    #values_readded = array_ops.where(
    #    mask2 == 1 and values_deleted == 1,
    #    array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
    #    array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    #mask_readded = array_ops.scatter_nd(sorted_indices_ex, new_values,
    #                             new_values.shape)                             
                                 
    #n_readded = math_ops.cast(math_ops.reduce_sum(values_readded), dtype=dtypes.int32, name= "n_readded" )                             
    #indices_deleted = array_ops.where(mask_casted == 1 and mask1 == 0)
    #print("mask_deleted = ", values_deleted, flush= True)
    #print("mask_readded = ", values_readded, flush= True)
    #print("n_readded = ", n_readded, flush= True)
    #mask2 = tf.Print(mask2, ["n_readded = ", n_readded, " - n_total = ", n_total,\
    #                                    " - weights=", weights.shape])
  
    ### delete readded values from mask 2
    
    n_current_mask2 = math_ops.cast(math_ops.reduce_sum(mask2), dtype=dtypes.int32)
    #mask2 = tf.Print(mask2, ["n_before_mask2 = ", n_current_mask2, " - weights=", weights.shape])
    
    values_update_mask2 = array_ops.where(
        tf.math.logical_and(tf.equal(mask2,1), tf.equal(mask_deleted,0)),
        array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    #mask2 = tf.Print(mask2, ["values_update_mask2 = ", values_update_mask2, " - weights=", weights.shape])
    #mask2 =  array_ops.scatter_nd( values_update_mask2, tf.range(n_readded), new_values.shape)
    #_, sorted_indices = nn_ops.top_k(values_update_mask2, k=n_total)
    #sorted_indices_ex = array_ops.expand_dims(sorted_indices, 1)
    
    mask3    = array_ops.scatter_nd(sorted_indices_ex, values_update_mask2,
                                 new_values.shape)
    
    n_current_mask2 = math_ops.cast(math_ops.reduce_sum(mask3), dtype=dtypes.int32)
    #mask2 = tf.Print(mask2, ["n_current_mask2 = ", n_current_mask2, " - weights=", weights.shape])
    
    #mask2 = mask2[values_update_mask2].assign(0)
    #mask2 = tf.Print(mask2, ["mask2 = ", mask2, " - weights=", weights.shape])
    #intersect = math_ops.cast(math_ops.reduce_sum(mask1, mask3))
    #mask2 = tf.Print(mask2, ["intersect = ", intersect, " - weights=", weights.shape])
   
    #mask2 = mask3
    
    
    
    
    
    #print("mask_casted.dtype ", mask_casted.dtype)
    #print("@@@@@@@@@@ update_mask2 ", update_mask2)
    #mask2 = array_ops.scatter_nd(sorted_indices_ex, update_mask2,
    #                             update_mask2.shape)
    

    # delete readded values from mask 2
 
    # add randomly n_readded connections
    # get the indices where connection can be added
    #set_random = array_ops.where(
    #     tf.math.logical_and(tf.math.logical_and(tf.equal(mask3,0), tf.equal(mask_casted_reshaped,0)),
    #                    tf.math.logical_and(tf.equal(mask1,0), tf.equal(mask2,0))),
    #    array_ops.ones_like(sorted_indices, dtype=mask_casted.dtype),
    #    array_ops.zeros_like(sorted_indices, dtype=mask_casted.dtype))
    
    score_random = self._random_uniform(
       weights.shape, seed=hash(weights.name + 'grow'))
    score_random = array_ops.reshape(score_random, [-1])
    #score_random = tf.random.uniform([n_total])
    score_random_lifted = array_ops.where(
        math_ops.equal(mask3, 1),
        array_ops.ones_like(mask3) * (math_ops.reduce_min(score_random) - 1),
        score_random)
        
    score_random_lifted2 = array_ops.where(
        math_ops.equal(mask1, 1),
        array_ops.ones_like(mask1) * (math_ops.reduce_min(score_random_lifted) - 1),
        score_random)    
        
    _, sorted_random= nn_ops.top_k(
        score_random_lifted2, k=n_total)
    sorted_indices_ex = array_ops.expand_dims(sorted_random, 1)
    # We will have zeros after having `n_keep` many ones.
    new_values = array_ops.where(
        math_ops.range(n_total) < n_readded,
        array_ops.ones_like(sorted_random, dtype=mask_casted.dtype),
        array_ops.zeros_like(sorted_random, dtype=mask_casted.dtype))

    mask_random = array_ops.scatter_nd(sorted_indices_ex, new_values,
                                 new_values.shape)
    mask4 = mask3 + mask_random
   
    #mask4 = tf.Print(mask4, [new_values, "mask_random = ", math_ops.reduce_sum(mask_random), " - weights=", weights.shape])
    #intersect = math_ops.reduce_sum(mask1* mask3)
    #mask4 = tf.Print(mask4, ["intersect_3 = ", intersect, " - weights=", weights.shape])
    #intersect = math_ops.reduce_sum(mask1* mask_random)
    #mask4 = tf.Print(mask4, ["intersect_random = ", intersect, " - weights=", weights.shape])
    
    #indices_readded = array_ops.where(mask2[indices_deleted] == 1 )
    
    #mask_deleted = tf.sets.difference(mask_casted, mask1)
    
    #mask_deleted = tf.sets.difference(mask_deleted, mask1)
    
    
    
    
    # Ensure masks are disjoint.
    assert_op = control_flow_ops.Assert(
        math_ops.equal(math_ops.reduce_sum(mask1 * mask4), 0.), [mask1, mask4])

    with ops.control_dependencies([assert_op]):
        # Let's set the weights of the growed connections.
        mask2_reshaped = array_ops.reshape(mask4, mask.shape)
    # Set the values of the new connections.
    grow_tensor = self.get_grow_tensor(weights, self._grow_init)
    if reinit_when_same:
      # If dropped and grown, we re-initialize.
      new_connections = math_ops.equal(mask2_reshaped, 1)
    else:
      new_connections = math_ops.logical_and(
          math_ops.equal(mask2_reshaped, 1), math_ops.equal(mask_casted, 0))
    new_weights = array_ops.where(new_connections, grow_tensor, weights)
    weights_update = state_ops.assign(weights, new_weights)
    # Ensure there is no momentum value for new connections
    reset_op = self.reset_momentum(weights, new_connections)

    with ops.control_dependencies([weights_update, reset_op]):
      mask_combined = array_ops.reshape(mask1 + mask4, mask.shape)
    mask_combined = math_ops.cast(mask_combined, dtype=old_dtype)
    new_mask = state_ops.assign(mask, mask_combined)
    return new_mask

  def reset_momentum(self, weights, new_connections):
    reset_ops = []
    for s_name in self._optimizer.get_slot_names():
      # Momentum variable for example, we reset the aggregated values to zero.
      optim_var = self._optimizer.get_slot(weights, s_name)
      new_values = array_ops.where(new_connections,
                                   array_ops.zeros_like(optim_var), optim_var)
      reset_ops.append(state_ops.assign(optim_var, new_values))
    return control_flow_ops.group(reset_ops)

  def get_grow_tensor(self, weights, method):
    """Different ways to initialize new connections.

    Args:
      weights: tf.Tensor or Variable.
      method: str, available options: 'zeros', 'random_normal', 'random_uniform'
        and 'initial_value'

    Returns:
      tf.Tensor same shape and type as weights.

    Raises:
      ValueError, when the method is not valid.
    """
    if not isinstance(method, six.string_types):
      raise ValueError('Grow-Init: %s is not a string' % method)

    if method == 'zeros':
      grow_tensor = array_ops.zeros_like(weights, dtype=weights.dtype)
    elif method.startswith('initial_dist'):
      original_shape = weights.initial_value.shape
      divisor = extract_number(method)
      grow_tensor = array_ops.reshape(
          random_ops.random_shuffle(
              array_ops.reshape(weights.initial_value, [-1])),
          original_shape) / divisor
    elif method.startswith('random_normal1'):
      stddev = math_ops.reduce_std(weights)
      divisor = extract_number(method)
      grow_tensor = self._random_normal(
          weights.shape,
          mean=0,
          stddev=1, #stddev,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_n')) / 10 #divisor
    elif method.startswith('random_normal2'):
      stddev = math_ops.reduce_std(weights)
      divisor = extract_number(method)
      grow_tensor = self._random_normal(
          weights.shape,
          mean=0,
          stddev=stddev,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_n'))  /2
          
    elif method.startswith('random_uniform'):
      mean = math_ops.reduce_mean(math_ops.abs(weights))
      mean = np.sqrt(6. / weights.get_shape().as_list()[0])
      divisor = extract_number(method)
      grow_tensor = self._random_uniform(
          weights.shape,
          minval=-mean,
          maxval=mean,
          dtype=weights.dtype,
          seed=hash(weights.name + 'grow_init_u')) / divisor
    else:
      raise ValueError('Grow-Init: %s is not a valid option.' % method)
    return grow_tensor

  def _random_uniform(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = math_ops.cast(
          array_ops.stack([c_seed, self._global_step]), dtypes.int32)
      return stateless_random_ops.stateless_random_uniform(*args, **kwargs)
    else:
      return random_ops.random_uniform(*args, **kwargs)

  def _random_normal(self, *args, **kwargs):
    if self._use_stateless:
      c_seed = self._stateless_seed_offset + kwargs['seed']
      kwargs['seed'] = math_ops.cast(
          array_ops.stack([c_seed, self._global_step]), dtypes.int32)
      return stateless_random_ops.stateless_random_normal(*args, **kwargs)
    else:
      return random_ops.random_normal(*args, **kwargs)

