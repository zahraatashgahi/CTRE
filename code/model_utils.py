from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os; import sys; sys.path.append(os.getcwd())
import time


from sklearn.metrics import pairwise_distances

import numpy as np
import sparse_optimizers 
import sparse_utils 
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data

from tqdm import tqdm
import numpy as np
import time
from utils import *



def network_fc(data_X, data_y, reuse=False, params = {}, model_pruning=False, FLAGS= {}, args={}):
  """Define a basic FC network."""
  regularizer = contrib_layers.l2_regularizer(scale=FLAGS.l2_scale)
  if model_pruning:
    y = layers.masked_fully_connected(
        inputs=data_X,
        num_outputs=args.num_hidden,
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        reuse=reuse,
        scope='layer1')
    y1 = layers.masked_fully_connected(
        inputs=y,
        num_outputs=args.num_hidden,
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        reuse=reuse,
        scope='layer2')
    y2 = layers.masked_fully_connected(
        inputs=y1,
        num_outputs=args.num_hidden,
        activation_fn=tf.nn.relu,
        weights_regularizer=regularizer,
        reuse=reuse,
        scope='layer3')
    logits = layers.masked_fully_connected(
        inputs=y2, num_outputs=params["num_classes"], reuse=reuse, activation_fn=None,
        weights_regularizer=regularizer, scope='layer4')
  else:
    y = tf.layers.dense(
        inputs=data_X, 
        units=args.num_hidden,
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        reuse=reuse,
        name='layer1')
    y1 = tf.layers.dense(
        inputs=y,
        units=args.num_hidden,
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        reuse=reuse,
        name='layer2')
    y2 = tf.layers.dense(
        inputs=y1,
        units=args.num_hidden,
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        reuse=reuse,
        name='layer3')
    logits = tf.layers.dense(inputs=y2, units=params["num_classes"], reuse=reuse,
                             kernel_regularizer=regularizer, name='layer4')

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=data_y, logits=logits)

  cross_entropy += tf.losses.get_regularization_loss()

  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(data_y, predictions), tf.float32))
    # Remove extra added ones. Current implementation adds the variables twice
  # to the collection. Improve this hacky thing.
  # TODO test the following with the convnet or any other network.
  print("\n\n\n****************         Pruning method          *************", flush=True)
  use_model_pruning = FLAGS.training_method != 'baseline'
  if use_model_pruning:
    for k in ('masks', 'masked_weights', 'thresholds', 'kernel'):
      # del tf.get_collection_ref(k)[2]
      # del tf.get_collection_ref(k)[2]
      collection = tf.get_collection_ref(k)
      print(collection)
      #del collection[len(collection)//2:]
      print(tf.get_collection_ref(k))
    print("*******------------------------------------********")
    print("collection = ", collection, flush=True)
    print("*******------------------------------------********")
  return cross_entropy, accuracy, [y, y1, y2, logits]



def map_fn(x, y):
    # Do transformations here
    return x, y

def get_compressed_fc(masks):
  """Given the masks of a sparse network returns the compact network."""
  # Dead input pixels.
  inds = np.sum(masks[0], axis=1) != 0
  masks[0] = masks[0][inds]
  compressed_masks = []
  for i in range(len(masks)):
    w = masks[i]
    # Find neurons that doesn't have any incoming edges.
    do_w = np.sum(w, axis=0) != 0
    if i < (len(masks) - 1):
      # Find neurons that doesn't have any outgoing edges.
      di_wnext = np.sum(masks[i+1], axis=1) != 0
      # Kept neurons should have at least one incoming and one outgoing edges.
      do_w = np.logical_and(do_w, di_wnext)
    compressed_w = w[:, do_w]
    compressed_masks.append(compressed_w)
    if i < (len(masks) - 1):
      # Remove incoming edges from removed neurons.
      masks[i+1] = masks[i+1][do_w]
  sparsities = [np.sum(m == 0) / float(np.size(m)) for m in compressed_masks]
  sizes = [compressed_masks[0].shape[0]]
  for m in compressed_masks:
    sizes.append(m.shape[1])
  return sparsities, sizes




def get_sparsities(args, params, FLAGS):
  print("num_classes = ", params["num_classes"])
  print("num_hidden = ", args.num_hidden)
  print("dim = ", params["dim"])
  print("epsilon = ", args.epsilon)
  
  nc = params["num_classes"]
  nh = args.num_hidden
  ndim = params["dim"]
  ep = args.epsilon
        
  dense_connections = ndim*nh+nh*nh+nh*nh+nh*ndim
  sparse_connections= int((ndim*nh)* (ep*(ndim+nh)/(ndim*nh))) +\
                    int((nh*nh)* (ep*(nh+nh)/(nh*nh)))  +\
                    int((nh*nh)* (ep*(nh+nh)/(nh*nh)))  +\
                    int((nh*nc)* (ep*(nh+nc)/(nh*nc))) 

  sparsity = 1 -  sparse_connections/ dense_connections
  FLAGS.end_sparsity = sparsity
  print("sparsity " , sparsity, flush=True)
  s1 = 1 - ep*(ndim+nh)/(ndim*nh)
  s2 = 1 - ep*(nh+nh)/(nh*nh)
  s3 = 1 - ep*(nh+nh)/(nh*nh)
  s4 = 1 - ep*(nh+nc)/(nh*nc)
  if s4 < 0:
    s4 = 0
  print("layer 1 - num connections:" , int((ndim*nh)* (ep*(ndim+nh)/(ndim*nh))), "sparsity:", s1*100, "%", flush=True)
  print("layer 2 - num connections:" , int((nh*nh)* (ep*(nh+nh)/(nh*nh)))      , "sparsity:", s2*100, "%", flush=True)
  print("layer 3 - num connections:" , int((nh*nh)* (ep*(nh+nh)/(nh*nh)))      , "sparsity:", s3*100, "%", flush=True)
  print("layer 4 - num connections:" , int((nh*nc)* (ep*(nh+nc)/(nh*nc)))      , "sparsity:", s4*100, "%", flush=True)
  print("FLAGS.end_sparsity  = ", FLAGS.end_sparsity )
  custom_sparsities = {
        'layer1': s1, #FLAGS.end_sparsity * FLAGS.sparsity_scale,
        'layer2': s2, #FLAGS.end_sparsity * FLAGS.sparsity_scale,
        'layer3': s3, #FLAGS.end_sparsity * FLAGS.sparsity_scale,
        'layer4': s4, #FLAGS.end_sparsity # * 0
  }
  return custom_sparsities, FLAGS


def get_optimizer(FLAGS, num_batches, loss, custom_sparsities ={}):
  global_step = tf.train.get_or_create_global_step()
  use_model_pruning = FLAGS.training_method != 'baseline'
  if FLAGS.optimizer != 'adam':
    if not use_model_pruning:
      boundaries = [int(round(s * num_batches)) for s in [60, 70, 80]]
    else:
      # if pruning
      boundaries = [int(round(s * num_batches)) for s
                    in [FLAGS.lr_drop_epoch, FLAGS.lr_drop_epoch + 20]]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries,
        values=[FLAGS.learning_rate / (3. ** i)
                for i in range(len(boundaries) + 1)])
    print("boundaries = ", boundaries)
  else:
    learning_rate = FLAGS.learning_rate
  print("************* learning_rate = ", learning_rate)
  
  print("\n\n\n****************        optimizer          *************")
  print("************* optimizer = ", FLAGS.optimizer, flush=True)
  if FLAGS.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
  elif FLAGS.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum,
                                     use_nesterov=FLAGS.use_nesterov)
  elif FLAGS.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise RuntimeError(FLAGS.optimizer + ' is unknown optimizer type')
    
    
  if FLAGS.training_method == 'set':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseSETOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)
        
  elif FLAGS.training_method in ['cte','ctre_seq']:
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseCTEOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)      
  elif FLAGS.training_method in ['ctre_sim']:
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseCTREsimOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)      
        
        
  elif FLAGS.training_method == 'static':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseStaticOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal)
  elif FLAGS.training_method == 'momentum':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseMomentumOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, momentum=FLAGS.s_momentum,
        frequency=FLAGS.maskupdate_frequency, drop_fraction=FLAGS.drop_fraction,
        grow_init=FLAGS.grow_init,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal, use_tpu=False)
  elif FLAGS.training_method == 'rigl':
    # We override the train op to also update the mask.
    opt = sparse_optimizers.SparseRigLOptimizer(
        opt, begin_step=FLAGS.maskupdate_begin_step,
        end_step=FLAGS.maskupdate_end_step, grow_init=FLAGS.grow_init,
        frequency=FLAGS.maskupdate_frequency,
        drop_fraction=FLAGS.drop_fraction,
        drop_fraction_anneal=FLAGS.drop_fraction_anneal,
        initial_acc_scale=FLAGS.rigl_acc_scale, use_tpu=False)
  elif FLAGS.training_method == 'snip':
    opt = sparse_optimizers.SparseSnipOptimizer(
        opt,
        mask_init_method=FLAGS.mask_init_method,
        default_sparsity=FLAGS.end_sparsity,
        custom_sparsity_map=custom_sparsities,
        use_tpu=False)
  elif FLAGS.training_method in ('scratch', 'baseline', 'prune'):
    pass
  else:
    raise ValueError('Unsupported pruning method: %s' % FLAGS.training_method)
  print("\n\n\n****************       Get Training optimizer         ***************")
  #print("start ", flush=True)
  #print("loss" , loss, flush=True)
  train_op = opt.minimize(loss, global_step=global_step)
  #print("done", flush=True)

  if FLAGS.training_method == 'prune':
    hparams_string = ('begin_pruning_step={0},sparsity_function_begin_step={0},'
                      'end_pruning_step={1},sparsity_function_end_step={1},'
                      'target_sparsity={2},pruning_frequency={3},'
                      'threshold_decay={4}'.format(
                          FLAGS.prune_begin_step, FLAGS.prune_end_step,
                          FLAGS.end_sparsity, FLAGS.pruning_frequency,
                          FLAGS.threshold_decay))
    print("hparams_string = ", hparams_string, flush=True)
    pruning_hparams = pruning.get_pruning_hparams().parse(hparams_string)
    pruning_hparams.set_hparam('weight_sparsity_map',
                               ['{0}:{1}'.format(k, v) for k, v
                                in custom_sparsities.items()])
    print("pruning_hparams = ", pruning_hparams, flush= True)
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)
    with tf.control_dependencies([train_op]):
      train_op = pruning_obj.conditional_mask_update_op()
      

      
  return opt, train_op, learning_rate















