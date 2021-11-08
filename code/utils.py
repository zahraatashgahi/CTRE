"""*************************************************************************"""
"""                           IMPORT LIBRARIES                              """
  
import numpy as np   
from sklearn import preprocessing  
import scipy
from scipy.io import loadmat
from sklearn.model_selection import train_test_split 
import urllib.request as urllib2 
import errno
#import tensorflow as tf
import pandas as pd
import argparse
import numpy as np

"""*************************************************************************"""
"""                         Read input                                  """


import  sklearn
import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix

###############################################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

###############################################################
# FUNCTIONS

def metrics(Y_test, y_pred):
    print('---------------------------------------------------------------')
    print('Accuracy: %.2f' % accuracy_score(Y_test,   y_pred) )
    confmat = confusion_matrix(y_true=Y_test, y_pred=y_pred)
    print("confusion matrix")
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
    print('F1-measure: %.3f' % f1_score(y_true=Y_test, y_pred=y_pred))
    print('\n\n\n\n')


def convert_numeric(x):
    d = []
    for y in x:
        if y=='na':
            d.append(np.nan)
        else:
            d.append(float(y))
    return d



# Counting null values in all the columns
def count_null(x):
    count = 0
    for i in x:
        if i=='na':
            count+=1
    return count


# list reverse
def Reverse(lst):
    return [element for element in reversed(lst)]


def save_time(path, epsilon, zeta, time, t_10):
    file1 = open(path+"epsilon_"+str(epsilon)+"_zeta_"+str(zeta)+"_runningtime.txt","a") 
    file1.write('%s\n' % t_10)    
    file1.write('%s\n' % time)
    file1.close()

    
def check_path(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise 

import pickle
def save_obj(name, obj):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


"""*************************************************************************"""
"""                               Parameters                                """
def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data options
    parser.add_argument('--dataset_name', type=str, default='coil20', help='dataset name')

    
    # Model options
    parser.add_argument("--epsilon", type=int, default=13, help="epsilon")
    parser.add_argument("--zeta", type=float, default=0.2, help="zeta")
    parser.add_argument("--lr_drop_epoch", type=int, default=50, help="lr_drop_epoch")
    parser.add_argument('--train_alg', type=str, required=True, help='Training algorithm')
    parser.add_argument('--weight_init', type=str, required=True, help='weight_init')
    parser.add_argument('--alg_extra', type=str, default="", help='extra')
    parser.add_argument("--num_hidden", default=100, help="num_hidden", type=int)
  

    # Train options
    parser.add_argument("--epochs", help="epochs", type=int)
    parser.add_argument('--rounds', nargs="+", type=int)
    parser.add_argument("--seed", default=0, help="seed", type=int)
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples per mini-batch')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--early_stop_epoch", type=int, default=40, help="early_stop_epoch")
    
    

    args = parser.parse_args()
    return args


def define_flags(args):
    from absl import flags
    #flags.DEFINE_string('mnist', '/tmp/data', 'Location of the MNIST ' 'dataset.')

    ## optimizer hyperparameters
    flags.DEFINE_integer('batch_size', args.batch_size, 'The number of samples in each batch')
    flags.DEFINE_float('learning_rate', args.lr, 'Initial learning rate.')
    flags.DEFINE_float('momentum', args.momentum, 'Momentum.')
    flags.DEFINE_boolean('use_nesterov', True, 'Use nesterov momentum.')
    flags.DEFINE_integer('num_epochs', args.epochs, 'Number of epochs to run.')
    flags.DEFINE_integer('lr_drop_epoch', args.lr_drop_epoch, 'The epoch to start dropping lr.')
    flags.DEFINE_string('optimizer', 'momentum',
                        'Optimizer to use. sgd, momentum or adam')
    flags.DEFINE_float('l2_scale', 1e-4, 'l2 loss scale')
    flags.DEFINE_string('network_type', 'fc',
                        'Type of the network.')
    flags.DEFINE_enum(
        'training_method', args.train_alg,
        ('scratch', 'set', 'baseline', 'momentum', 'rigl', 'static', 'snip', 'cte', 'ctre_seq', 'ctre_sim', 'prune'),
        'Method used for training sparse network. `scratch` means initial mask is '
        'kept during training. `set` is for sparse evalutionary training and '
        '`baseline` is for dense baseline.')
    flags.DEFINE_float('drop_fraction', args.zeta,
                       'When changing mask dynamically, this fraction decides how '
                       'much of the ')
    flags.DEFINE_string('drop_fraction_anneal', 'constant',
                        'If not empty the drop fraction is annealed during sparse'
                        ' training. One of the following: `constant`, `cosine` or '
                        '`exponential_(\\d*\\.?\\d*)$`. For example: '
                        '`exponential_3`, `exponential_.3`, `exponential_0.3`. '
                        'The number after `exponential` defines the exponent.')
    if args.train_alg == "rigl":
        flags.DEFINE_string('grow_init', 'zeros',
                            'Passed to the SparseInitializer, one of: zeros, '
                            'initial_value, random_normal, random_uniform.')
    else:
        flags.DEFINE_string('grow_init', args.weight_init,
                            'Passed to the SparseInitializer, one of: zeros, '
                            'initial_value, random_normal1, random_normal2, random_uniform.')
                            
    flags.DEFINE_float('s_momentum', 0.9,
                       'Momentum values for exponential moving average of '
                       'gradients. Used when training_method="momentum".')

    flags.DEFINE_float('sparsity_scale', 0.9, 'Relative sparsity of second layer.')
    flags.DEFINE_float('rigl_acc_scale', 0.,
                       'Used to scale initial accumulated gradients for new '
                       'connections.')
    flags.DEFINE_integer('maskupdate_begin_step', 0, 'Step to begin mask updates.')
    # will be adapted later
    flags.DEFINE_integer('maskupdate_end_step', 50000, 'Step to end mask updates.')
    # will be adapted later
    flags.DEFINE_integer('maskupdate_frequency', 600,
                         'Step interval between mask updates.')
    flags.DEFINE_integer('mask_record_frequency', 0,
                         'Step interval between mask logging.')
    flags.DEFINE_string(
        'mask_init_method',
        default='random',
        help='If not empty string and mask is not loaded from a checkpoint, '
        'indicates the method used for mask initialization. One of the following: '
        '`random`, `erdos_renyi`.')
    flags.DEFINE_integer('prune_begin_step', 0, 'step to begin pruning')
    # will be adapted later
    flags.DEFINE_integer('prune_end_step', 30000, 'step to end pruning')
    # will be adapted later
    flags.DEFINE_float('end_sparsity', .98, 'desired sparsity of final model.')
    # will be adapted later
    flags.DEFINE_integer('pruning_frequency', 600, 'how often to prune.')
    flags.DEFINE_float('threshold_decay', 0, 'threshold_decay for pruning.')
    flags.DEFINE_string('save_path', "./results/results_"+args.dataset_name+"/"+str(args.num_hidden)+"/"+\
                               str(args.epsilon)+"/"+args.train_alg+args.alg_extra+"/run_"+str(args.seed)+"/"\
                            , 'Where to save the model.')
    flags.DEFINE_boolean('save_model', True, 'Whether to save model or not.')
    flags.DEFINE_integer('seed', default=args.seed, help=('Sets the random seed.'))

    FLAGS = flags.FLAGS
    return FLAGS



"""*************************************************************************"""
"""                             Load data                                   """
    
def load_data(args):
    import numpy as np
    name = args.dataset_name
        
    if name=="madelon":
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        X_train = np.loadtxt(urllib2.urlopen(train_data_url))
        y_train = np.loadtxt(urllib2.urlopen(train_resp_url))
        X_test =  np.loadtxt(urllib2.urlopen(val_data_url))
        y_test =  np.loadtxt(urllib2.urlopen(val_resp_url))
        y_train[y_train < 0] = 0
        y_test[y_test < 0] = 0
        
    elif name=="isolet":
        import pandas as pd 
        data= pd.read_csv('./datasets/isolet.csv')
        data = data.values 
        X = data[:,:-1]
        X = X.astype("float")
        y = data[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
        for i in range(len(y_train)):
            if len(y_train[i])==4:
                y_train[i] = int(y_train[i][1])*10 + int(y_train[i][2])
            elif len(y_train[i])==3:
                y_train[i] = int(y_train[i][1])
        for i in range(len(y_test)):
            if len(y_test[i])==4:
                y_test[i] = int(y_test[i][1])*10 + int(y_test[i][2])
            elif len(y_test[i])==3:
                y_test[i] = int(y_test[i][1])

        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')

    elif name == "MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')  
        
    elif name == "Fashion-MNIST":
        import tensorflow as tf
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')    

    elif name == "cifar10":    
        import tensorflow as tf
        # load dataset
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print(X_train.shape)
        print(y_train.shape)
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
        X_test  = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))
        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')
        y_test = y_test.ravel()
        y_train = y_train.ravel()
    

    
    
    import numpy as np
    import pandas as pd
    #
    if name == "madelon":
        scaler = preprocessing.StandardScaler().fit(X_train)
    else:
        scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)   

    
    if name in ["har", "isolet"]:
        y_train = y_train - 1
        y_test = y_test - 1
    print("train labels: ", np.unique(y_train))
    print("test labels: ", np.unique(y_test))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32') 
    y_train = y_train.astype('int')
    y_test  = y_test.astype('int')
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)       
    return X_train, y_train, X_test, y_test
    




