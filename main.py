#Trying to implement a clustering-based semi-supervised CNN with MNIST and pytorch. 2020/01/27
#fix init#
#3/9 change steps to epoches#
from numpy.random import seed
#from tensorflow.random import set_seed 
#set_seed(2)
import argparse 
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import pickle
import numpy as np
import tensorflow as tf
tf.random.set_seed(2)
tf.get_logger().setLevel('ERROR')
import keras
sys.stderr = stderr
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import TensorBoard

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#import clustering
import my_clustering
import utils
from models import *

def parse_args():
    parser = argparse.ArgumentParser(description='Keras Implementation of Semi-DeepCluster')
    #parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--n_cluster', '--k', type=int, default=16,
                        help='number of cluster for k-means (default: 16)')
    parser.add_argument('--alpha', '--a', type=float, default=0.05,
                        help='parameter that controls the role of supervsied CNN (default: 0.05)')
    parser.add_argument('--beta', '--b', type=float, default=1,
                        help='parameter that controls the role of Deep Clustering (default: 1)')                            
    parser.add_argument('--epochs', '--e',type=int, default=20,
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--n_classes', '--c', type=int, default=16,
                        help='number of classes (default: 16)')
    parser.add_argument('--n_sup', type=int, default=16,
                        help='number of supervsied data (default: 16)')
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed (default: 3)')                            
    parser.add_argument('--path2data', type=str, default='C:/Users/acicula/Desktop/sGAN/dataset/experiment1/', help='path to dataset folder')
    parser.add_argument('--scale', default=(0,1), type=tuple,
                        help='scale range of the sample (default: 0 to 1)')    
    parser.add_argument('--verbose', '-v', action='store_true', help='make noise')

    return parser.parse_args()


def main(args):

    ###################### setup TensorBoard ######################
    logdir = 'log/main/Semi-Deep-Clustering-{}'.format(int(time.time()))

    #tensorboard = TensorBoard(log_dir=logdir)
    train_writer = tf.summary.create_file_writer(logdir)
    ###############################################################
    # fix random seeds
    seed(args.seed)
    end = time.time()
    # load the data
    X_train, y_train, X_tst, y_tst = utils.data_preproc(np.asarray(pickle.load(open('dataset/EXP1.pickle','rb'))))
    X_train , X_eva, y_train , y_eva = train_test_split(X_train, y_train, test_size=0.1)

    if args.verbose:
        print('Loading CSI Samples from {} \n Training set size: {}, \n Testing size: {}'.format(args.path2data, X_train.shape, X_tst.shape))

    # setup optimizer
    DC_opt = Adam(lr=1e-5, beta_1=0.5)
    C_opt = Adam(lr=1e-6, beta_1=0.5)

    #create CNN   
    DC_model, C_model = semi_cnn(X_train.shape[1:], args.n_classes, args.n_cluster, args.alpha, args.beta, DC_opt, C_opt)
    if args.verbose:
        DC_model.summary()
        C_model.summary()
    #tensorboard.set_model(C_model)
    #tensorboard.set_model(DC_model)

    #load supervised samples, pre-train the CNN
    X_sup, y_sup = utils.select_supervised_samples((X_train, y_train), args.n_sup, args.n_classes)

    # create one-hot label
    y_sup = keras.utils.to_categorical(y_sup, args.n_classes)
    y_eva = keras.utils.to_categorical(y_eva, args.n_classes)
    y_tst = keras.utils.to_categorical(y_tst, args.n_classes)
    #if args.verbose:
    #    print('=>Train the CNN with supervised sampeles with shape:: {}'.format(X_sup.shape, y_sup.shape))
    #    print(y_sup)

    # clustering algorithm to use
    deepcluster = my_clustering.Kmeans(args.n_cluster)


    # calculate the number of training iterations
    bat_per_epo = int(X_train.shape[0] / args.batch_size)
    n_steps = bat_per_epo * args.epochs
    if args.verbose:
        print('n_epochs=%d, n_batch=%d, b/e=%d, steps=%d' % (args.epochs, args.batch_size, bat_per_epo, n_steps))

    ############################################################    
    ###########training model with semi-DeepCluster#############
    ############################################################
    with train_writer.as_default(): 
        for i in range(n_steps):
            end = time.time()
            if args.verbose:
                print('=>Step.{}/ {}'.format(i+1, n_steps))

            #fit the model with supervised samples
            _loss1, _acc1 = C_model.train_on_batch(X_sup, y_sup)

            if args.verbose:
                print('C_loss:{}'.format(_loss1))        
            #remove DC_models' fc layer 
            features_model = Model(DC_model.input, DC_model.layers[-3].output)
            # get features for the whole dataset
            features = features_model.predict(X_train)
            # cluster the features
            #if args.verbose:
                #print('=>Cluster the features')
            images_lists = deepcluster.cluster(features, verbose=args.verbose)

            # assign pseudo-labels
            #if args.verbose:
                #print('=>Assign pseudo labels')           
            train_dataset = my_clustering.cluster_assign(images_lists, X_train)
            _X_train , _y_train = train_dataset.X, train_dataset.pseudolabels
            _X_train_dc , _X_eva_dc, _y_train_dc , _y_eva_dc = train_test_split(_X_train , _y_train, test_size=0.1)
            _X_train_dc, _y_train_dc = utils.select_supervised_samples([_X_train_dc , _y_train_dc], args.batch_size, args.n_cluster)
            _y_train_dc = keras.utils.to_categorical(_y_train_dc, args.n_cluster)
            _y_eva_dc = keras.utils.to_categorical(_y_eva_dc, args.n_cluster)

            # train network with clusters as pseudo-labels
            _loss2, _acc2 = DC_model.train_on_batch(_X_train_dc, _y_train_dc)
            # evaluate DC every batch
            _eva_acc_dc = utils.evaluate_c_model(DC_model, _X_eva_dc, _y_eva_dc) 
            # evaluate C every batch
            _eva_acc_c = utils.evaluate_c_model(C_model, X_eva, y_eva)
            # test C every batch
            _tst_acc_c = utils.evaluate_c_model(C_model, X_tst, y_tst)
            if args.verbose:
               print('DC_loss:{}'.format(_loss2))
            tf.summary.scalar("C_loss", _loss1, step=i)
            tf.summary.scalar("DC_loss", _loss2, step=i)
              
            # summarize
            if i ==0 or (i+1)%bat_per_epo ==0:
                #writer.add_summary(summary= [_loss1, _loss2], global_step= i)
                #_acc = utils.evaluate_c_model(C_model, X_tst, y_tst)
                tf.summary.scalar("DC Evaluate Accuracy", _eva_acc_dc, step=i)
                tf.summary.scalar("C Evaluate Accuracy", _eva_acc_c, step=i)
                tf.summary.scalar("C Test Accuracy", _tst_acc_c, step=i)
                #acc = C_model.test_on_batch(X_tst, y_tst)
                #C_model.save('semi-cnn.h5')
    #save
    train_writer.flush() 
    train_writer.close()
    #print('Best Test:', max(tst_history))           
    #print('Time: % s\n'%(time.time() - end))

if __name__ == '__main__':
    # tensorboard --logdir=log/ --host localhost --port 8088
    args = parse_args()
    main(args)