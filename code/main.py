
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os; import sys; sys.path.append(os.getcwd())
import time


from sklearn.metrics import pairwise_distances

import numpy as np
import sparse_optimizers
import sparse_utils
from  sparse_optimizers import *
from sparse_utils import *
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data

from tqdm import tqdm

import os; import sys; sys.path.append(os.getcwd())
import numpy as np
import time




from utils import *
from model_utils import *



########################################################################################
#### read arguments


def main(unused_args):
  import numpy as np
  args = parse_arguments()
  FLAGS = define_flags(args)
  # set seed
  print("\n\n************  seed = ", FLAGS.seed, " ************")
  tf.set_random_seed(FLAGS.seed)
  tf.get_variable_scope().set_use_resource(True)
  np.random.seed(FLAGS.seed)
  params = {}
  # Load the MNIST data and set up an iterator.
  print("***********************************************************************", flush=True)
  print("****************                 Reading data            *************", flush=True)
  X_train, y_train, X_test, y_test = load_data(args)
  y_test = y_test.astype(np.int32)
  y_train = y_train.astype(np.int32)
  # split X_train into training and validation
  X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42) 
  print("new X_train: ", X_train.shape)
  print("new y_train: ", y_train.shape) 
  print("X_valid: ", X_valid.shape)
  print("X_train: ", y_valid.shape) 
  print("****************         Finished Reading data           *************", flush=True)
  print("***********************************************************************", flush=True)
  
  
  
  
  
  
  print("\n\n\n****************         Parameter Settings          *************", flush=True)
  params["num_classes"] = np.unique(y_train).shape[0]
  num_batches = X_train.shape[0] // FLAGS.batch_size
  num_batches_valid = X_valid.shape[0] // FLAGS.batch_size
  num_batches_test = X_test.shape[0] // FLAGS.batch_size
  params["dim"] = X_train.shape[1]
  FLAGS.maskupdate_end_step= num_batches* FLAGS.num_epochs
  FLAGS.prune_end_step=num_batches* FLAGS.num_epochs
  FLAGS.maskupdate_frequency= num_batches
  FLAGS.pruning_frequency=num_batches
  print("num_classes = ", params["num_classes"])
  print("num_batches = ", num_batches)
  print("num_batches_valid = ", num_batches_valid)
  print("num_batches_test = ", num_batches_test)
  print("dim = ",params["dim"])
  print("num_samples  = ", X_train.shape[0])
  print("maskupdate_frequency = ",FLAGS.maskupdate_frequency)
  print("pruning_frequency = ",FLAGS.pruning_frequency)
  print("maskupdate_end_step = ",FLAGS.maskupdate_end_step)
  
  
  
  print("\n\n\n****************         Training method          *************", flush=True)
  # Set up loss function.
  use_model_pruning = FLAGS.training_method != 'baseline'
  print("Training_method = ",FLAGS.training_method)
  print("use_model_pruning = ", use_model_pruning, flush=True)
  

  #################################################################################
  ##########        data iterators
  placeholder_X = tf.placeholder(tf.float32, shape = [None, params["dim"]])
  placeholder_y = tf.placeholder(tf.int32, shape = [None])
  
  # Create separate Datasets for training, validation and testing
  train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
  train_dataset = train_dataset.batch(FLAGS.batch_size).map(lambda x, y: map_fn(x, y))

  val_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
  val_dataset = val_dataset.batch(FLAGS.batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test_dataset = test_dataset.batch(FLAGS.batch_size)
  
  
  handle = tf.placeholder(tf.string, shape = [])
  iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
  data_X, data_y = iterator.get_next()
  data_y = tf.cast(data_y, tf.int32)
  
  # Create Reinitializable iterator for Train and Validation, one shot iterator for Test
  train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
  train_iterator = train_val_iterator.make_initializer(train_dataset)
  val_iterator = train_val_iterator.make_initializer(val_dataset)
  test_iterator = train_val_iterator.make_initializer(test_dataset) #test_dataset.make_one_shot_iterator()

  #################################################################################
  ##########        create the model
  
  print("\n\n\n****************         Creating the model           *************", flush=True)
  print("******* network_type = ",FLAGS.network_type, flush=True)
  print("******* num_hidden = ",args.num_hidden, flush=True)
  
  if FLAGS.network_type == 'fc':
    loss, accuracy, all_layers= network_fc(
        data_X, data_y, params =params, model_pruning=use_model_pruning, FLAGS = FLAGS, args = args)
  else:
    raise RuntimeError(FLAGS.network + ' is an unknown network type.')
  
   

  

    
 
      
  print("\n\n\n****************        sparsity          ***************", flush=True)
  custom_sparsities, FLAGS = get_sparsities(args, params, FLAGS)
  print("sparsity = ", custom_sparsities, flush=True)
  
  
  print("\n\n\n****************        Get Optimizer          *************", flush=True)
  print("FLAGS.training_method = ", FLAGS.training_method, flush=True)
  opt, train_op, learning_rate = get_optimizer(FLAGS, num_batches, loss, custom_sparsities=custom_sparsities)
  
  weight_sparsity_levels = pruning.get_weight_sparsity()
  print(weight_sparsity_levels, flush=True)
  print("***-----------------***", flush=True)
  global_sparsity = sparse_utils.calculate_sparsity(pruning.get_masks())
  tf.summary.scalar('test_accuracy', accuracy)
  tf.summary.scalar('global_sparsity', global_sparsity)
  print('accuracy', accuracy)
  print('global_sparsity', global_sparsity, flush=True)
  print("***-----------------***", flush=True)
  
  for k, v in zip(pruning.get_masks(), weight_sparsity_levels):
    tf.summary.scalar('sparsity/%s' % k.name, v)
  if FLAGS.training_method in ('prune', 'snip', 'baseline'):
    mask_init_op = tf.no_op()
    tf.logging.info('No mask is set, starting dense.')
  else:
    all_masks = pruning.get_masks()
    print("********* ---->", all_masks, flush=True)
    print("********* ---->", custom_sparsities, flush=True)
    mask_init_op = sparse_utils.get_mask_init_fn(
        all_masks, FLAGS.mask_init_method, FLAGS.end_sparsity,
        custom_sparsities)
  
  
  ## save model
  if FLAGS.save_model:
    saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()
  hyper_params_string = '_'.join([FLAGS.network_type, str(FLAGS.batch_size),
                                  str(FLAGS.learning_rate),
                                  str(FLAGS.momentum),
                                  FLAGS.optimizer,
                                  str(FLAGS.l2_scale),
                                  FLAGS.training_method,
                                  str(FLAGS.prune_begin_step),
                                  str(FLAGS.prune_end_step),
                                  str(FLAGS.end_sparsity),
                                  str(FLAGS.pruning_frequency),
                                  str(FLAGS.seed)])
  # remove old files in directory
  tf.io.gfile.makedirs(FLAGS.save_path)
  import os, shutil
  folder = FLAGS.save_path
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
  filename = os.path.join(FLAGS.save_path, hyper_params_string + '.txt')
  merged_summary_op = tf.summary.merge_all()



  # Run session.
  print("\n\n\n****************        Training starts         ***************", flush=True)
  #from tensorflow.python.client import device_lib 
  #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  #print(device_lib.list_local_devices(), flush=True)
  #config = tf.ConfigProto(log_device_placement=True)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  

  if use_model_pruning:
    ##################################################################################
    #####  if pruning:
    maximum_accuracy_valid=0
    metrics = np.zeros((FLAGS.num_epochs, 6))
    
    
    if FLAGS.training_method in ['cte', 'ctre', 'ctre_sim']:
        # array of all activations
        a0_all = tf.Variable(tf.zeros([num_batches*FLAGS.batch_size, params["dim"] ]), trainable =False, name="a0") 
        a1_all = tf.Variable(tf.zeros([num_batches*FLAGS.batch_size, args.num_hidden ]), trainable =False)      
        a2_all = tf.Variable(tf.zeros([num_batches*FLAGS.batch_size, args.num_hidden]), trainable =False)  
        a3_all = tf.Variable(tf.zeros([num_batches*FLAGS.batch_size, args.num_hidden ]) , trainable =False) 
        a4_all = tf.Variable(tf.zeros([num_batches*FLAGS.batch_size, params["num_classes"]]) , trainable =False) 
        # reset activations
        reset_a_all_0  = a0_all.assign(tf.zeros([num_batches*FLAGS.batch_size, params["dim"] ]))
        reset_a_all_1  = a1_all.assign(tf.zeros([num_batches*FLAGS.batch_size, args.num_hidden ]))
        reset_a_all_2  = a2_all.assign(tf.zeros([num_batches*FLAGS.batch_size, args.num_hidden ]))
        reset_a_all_3  = a3_all.assign(tf.zeros([num_batches*FLAGS.batch_size, args.num_hidden ]))
        reset_a_all_4  = a4_all.assign(tf.zeros([num_batches*FLAGS.batch_size, params["num_classes"] ]))
        # iteration index
        idx = tf.placeholder(tf.int32)
        # activation of a single batch
        a0 = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, params["dim"] ))
        a1 = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, args.num_hidden ))     
        a2 = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, args.num_hidden))
        a3 = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, args.num_hidden ) )
        a4 = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, params["num_classes"] ) )
        # assign activation functions
        assign_activation = [[],[],[],[],[]]
        scatter = tf.scatter_nd(tf.expand_dims(tf.range(idx*100,idx*100+FLAGS.batch_size), axis=1), a0, tf.shape(a0_all))
        assign_activation[0] = a0_all.assign(scatter)
        scatter = tf.scatter_nd(tf.expand_dims(tf.range(idx*100,idx*100+FLAGS.batch_size), axis=1), a1, tf.shape(a1_all))
        assign_activation[1] = a1_all.assign(scatter)
        scatter = tf.scatter_nd(tf.expand_dims(tf.range(idx*100,idx*100+FLAGS.batch_size), axis=1), a2, tf.shape(a2_all))
        assign_activation[2] = a2_all.assign(scatter)
        scatter = tf.scatter_nd(tf.expand_dims(tf.range(idx*100,idx*100+FLAGS.batch_size), axis=1), a3, tf.shape(a1_all))
        assign_activation[3] = a3_all.assign(scatter)
        scatter = tf.scatter_nd(tf.expand_dims(tf.range(idx*100,idx*100+FLAGS.batch_size), axis=1), a4, tf.shape(a4_all))
        assign_activation[4] = a4_all.assign(scatter)



      
    def compute_cosine_distances(a, b):
        # x shape is n_a * dim
        # y shape is n_b * dim
        # results shape is n_a * n_b
        a = tf.transpose(a)
        b = tf.transpose(b)
        
        #similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
        # Only necessary if vectors are not normalized
        #similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
        normalize_a = tf.nn.l2_normalize(a,1)        
        normalize_b = tf.nn.l2_normalize(b,1)
        similarity = tf.abs(tf.matmul(normalize_a, normalize_b, transpose_b=True))
        return similarity
    

        
    if FLAGS.training_method in ['cte', 'ctre', 'ctre_sim']:
        cal_cosine1 = opt._score_cosine[0].assign(compute_cosine_distances(a0_all, a1_all))
        cal_cosine2 = opt._score_cosine[1].assign(compute_cosine_distances(a1_all, a2_all))
        cal_cosine3 = opt._score_cosine[2].assign(compute_cosine_distances(a2_all, a3_all))
        cal_cosine4 = opt._score_cosine[3].assign(compute_cosine_distances(a3_all, tf.nn.softmax(a4_all, axis=1, name="fc_softmax")))

        weights = opt.get_weights()
        rand_cosine1 = opt._score_cosine[0].assign(tf.random.uniform(weights[0].shape))
        rand_cosine2 = opt._score_cosine[1].assign(tf.random.uniform(weights[1].shape))
        rand_cosine3 = opt._score_cosine[2].assign(tf.random.uniform(weights[2].shape))
        rand_cosine4 = opt._score_cosine[3].assign(tf.random.uniform(weights[3].shape))
    
     
    with tf.Session(config=config) as sess:
      train_val_string = sess.run(train_val_iterator.string_handle())
      sess.run(init_op)
      sess.run(mask_init_op)
      tic = time.time()
      mask_records = {}
      with tf.io.gfile.GFile(filename, 'w') as outputfile:
        train_loss, train_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0
        test_loss, test_accuracy = 0,0
        cnt_t = 0
        cnt = 0
        
        # ctre params
        early_stop = 0 
        flag_ctre = False
        
        sess.run(train_iterator, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})
        #mask_vals_old = sess.run(pruning.get_masks())
        
        ####################################################################################################
        ################                          start training                            ################
        ####################################################################################################
        
        
        KL = []
        for i in range(FLAGS.num_epochs * num_batches):

          ##################################################################################
          ###############   validation and test at the end of epoch   #####################
          ##################################################################################
          if (i % (num_batches)) == 0 and i >0:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("EPOCH ", i//num_batches, ". | Training algorithm: ", args.train_alg)
            #print("***------------------------computing accuracy--------------------------***")
            epoch_time = time.time() - tic
            tic_start_accuracy = time.time() 
            # Start validation iterator
            sess.run(val_iterator, feed_dict = {placeholder_X: X_valid, placeholder_y: y_valid})
            cnt = 0
            try:
                while True:
                    l, acc = sess.run([loss, accuracy], feed_dict = {handle: train_val_string})
                    val_loss += l
                    val_accuracy += acc
                    cnt += 1
            except tf.errors.OutOfRangeError:
                pass
            

            #print('Epoch: {}'.format(i//num_batches ))
            print("Training time = ", epoch_time, flush=True)

            ##############   train loss and accuracy   #################
            train_accuracy = train_accuracy / cnt_t
            train_loss  = train_loss / cnt_t 
            metrics[i//num_batches - 1, 0] = train_loss
            metrics[i//num_batches - 1, 1] = train_accuracy
            print('Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy, train_loss ))
            
            ##############   validation loss and accuracy   #################
            val_accuracy = val_accuracy / cnt
            val_loss = val_loss / cnt                                  
            metrics[i//num_batches - 1, 2] = val_loss
            metrics[i//num_batches - 1, 3] = val_accuracy
            print('Val accuracy:   {:.4f}, loss: {:.4f}\n'.format(val_accuracy,   val_loss   ))
            
            ##############   test loss and accuracy   #################
            if val_accuracy > maximum_accuracy_valid:
                early_stop = 0
                maximum_accuracy_valid = val_accuracy
                #if FLAGS.save_model:
                #    saver.save(sess, os.path.join(FLAGS.save_path, 'model.ckpt'))
                if mask_records:
                    np.save(os.path.join(FLAGS.save_path, 'mask_records'), mask_records)
                sess.run(test_iterator, feed_dict = {placeholder_X: X_test, placeholder_y: y_test})
                test_loss, test_accuracy = 0,0 
                cnt = 0
                try:
                    while True:
                        # Feed to feedable iterator the string handle of one shot iterator
                        l, acc = sess.run([loss, accuracy], feed_dict = {handle: train_val_string})
                        test_loss += l
                        test_accuracy += acc
                        #print("batch ", l , acc)
                        cnt += 1
                except tf.errors.OutOfRangeError:
                    pass
                test_accuracy = test_accuracy / cnt
                test_loss  = test_loss / cnt   
            else:
                early_stop += 1
            #print("early_stop = ", early_stop)
            if early_stop == args.early_stop_epoch and FLAGS.training_method in ['ctre']:
                print("\n\n\n\n ###########################################################################\n\n")
                print("\n\nStart Random Addition")
                print("\n\n\n\n ###########################################################################")
                flag_ctre  = True

            metrics[i//num_batches - 1, 4] = test_loss
            metrics[i//num_batches - 1, 5] = test_accuracy      
            
            train_loss, train_accuracy = 0, 0
            val_loss, val_accuracy = 0, 0
            cnt_t = 0
            sess.run(train_iterator, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})
            
            
            
            ##############   summary   #################
            np.savetxt(FLAGS.save_path +"results.txt", metrics)
                
                
            # Write logs at every test iteration.
            #summary_writer.add_summary(summary, i)
            log_str = 'Loss test: %.4f, Accuracy test: %.4f' % ( test_loss, test_accuracy)
            print(log_str, flush=True) 
            log_str =  '\nglobal_sparsity_val: %.4f : Layer sparsities: %.4f, %.4f, %.4f, %.4f ' % (global_sparsity_val, 
                weight_sparsity[0], weight_sparsity[1], weight_sparsity[2], weight_sparsity[3])
            print(log_str, flush=True)
            tic = time.time()
   
            
            
            print("Computing accuracy time",  time.time() - tic_start_accuracy)
          ##################################################################################
          ###############             Compute cosine similarity        #####################
          ##################################################################################
          if FLAGS.training_method in ['cte', 'ctre', 'ctre_sim'] and (i % (num_batches)) == 0 and i >0 : 

            import numpy as np
            tic3 = time.time() 
            
            mask_vals = sess.run(pruning.get_masks())
            for idx_l in range(len(mask_vals)):
                mask = mask_vals[idx_l].reshape(mask_vals[idx_l].shape[0]*mask_vals[idx_l].shape[1])
                mask_idx = np.where(mask != 0)[0]
                w = [v for v in tf.global_variables() if v.name == "layer"+str(idx_l+1)+"/weights:0"][0]
                w = sess.run(w)
                m = w.shape[0]
                n = w.shape[1]
                print("sparsity of layer ", idx_l+1, " (shape = ",w.shape,")= ",1 - mask_idx.shape[0]/ (w.shape[0]*w.shape[1]), flush=True)
                
                w_1d = w.reshape(w.shape[0]*w.shape[1])
                
                path_save_weights = FLAGS.save_path + "weights/layer"+str(idx_l+1)+"/"
                check_path(path_save_weights)
                check_path(path_save_weights+ "distribution_plots/")
                w_new = np.zeros(w_1d.shape[0])
                w_new[mask_idx] = w_1d[mask_idx]
                w_new = w_new.reshape((m,n))
                #print(w_new.shape)
                if int(i // (num_batches))% 25 == 0 or int(i // (num_batches))==1:
                    with open(path_save_weights + "w"+str(idx_l+1)+"_"+str(int(i // (num_batches)))+".npy", 'wb') as f:
                        np.save(f, w_new)
                
                from matplotlib import pyplot as plt
                import numpy as np
                heights,bins = np.histogram(w_1d[mask_idx],bins=100)
                heights = heights/sum(heights)
                plt.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), alpha=0.5)
                plt.savefig(path_save_weights+ "distribution_plots/w_"+str(idx_l+1)+"_" +  str(int(i // (num_batches))) +".png", bbox_inches='tight')
                plt.close() 
        
        
            if FLAGS.training_method in ['cte', 'ctre_sim'] or (FLAGS.training_method in ['ctre'] and not flag_ctre):
                #tic1 = time.time() 
                print("@@@@ computing cosine @@@@")
                sess.run(cal_cosine1)
                sess.run(cal_cosine2)
                sess.run(cal_cosine3)
                sess.run(cal_cosine4)
                c0 = sess.run(opt._score_cosine[0])
                c1 = sess.run(opt._score_cosine[1])
                c2 = sess.run(opt._score_cosine[2])
                c3 = sess.run(opt._score_cosine[3])
            elif  FLAGS.training_method in ['ctre'] and flag_ctre:
                print("random addition")
                sess.run(rand_cosine1)
                sess.run(rand_cosine2)
                sess.run(rand_cosine3)
                sess.run(rand_cosine4)
            # reset activations
            sess.run(reset_a_all_0)
            sess.run(reset_a_all_1)
            sess.run(reset_a_all_2)
            sess.run(reset_a_all_3)
            sess.run(reset_a_all_4)
   
            
          _, l, acc, a0_Val, a1_Val , a2_Val,\
                        a3_Val,a4_Val = sess.run([train_op, loss, accuracy, data_X, all_layers[0],\
                                        all_layers[1],all_layers[2],all_layers[3]],
                                        feed_dict = {handle: train_val_string})
          train_loss += l
          train_accuracy += acc
          cnt_t += 1
    
          ## assign activations   
          if FLAGS.training_method in ['cte', 'ctre', 'ctre_sim'] :
              sess.run(assign_activation[0],feed_dict = {a0:a0_Val, idx: i % (num_batches)})
              sess.run(assign_activation[1],feed_dict = {a1:a1_Val, idx: i % (num_batches)})
              sess.run(assign_activation[2],feed_dict = {a2:a2_Val, idx: i % (num_batches)})
              sess.run(assign_activation[3],feed_dict = {a3:a3_Val, idx: i % (num_batches)})
              sess.run(assign_activation[4],feed_dict = {a4:a4_Val, idx: i % (num_batches)})
              
          weight_sparsity, global_sparsity_val = sess.run([weight_sparsity_levels, global_sparsity])




if __name__ == '__main__':
  tf.app.run()
