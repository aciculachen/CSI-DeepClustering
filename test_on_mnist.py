#Trying to implement a clustering-based semi-supervised CNN with MNIST and pytorch. 2020/01/27
#fix init#
from numpy.random import seed
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
tf.get_logger().setLevel('ERROR')
import keras
sys.stderr = stderr
from tensorflow.keras.models import Model
from tensorflow.keras import datasets 
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

#import clustering
import my_clustering
import utils
import models

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
                        help='mini-batch size (default: 10)')
    parser.add_argument('--n_classes', '--c', type=int, default=10,
                        help='number of classes (default: 10)')
    parser.add_argument('--n_sup', type=int, default=100,
                        help='number of supervsied data (default: 100)')
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed (default: 3)')                            
    parser.add_argument('--path2data', type=str, default='C:/Users/acicula/Desktop/sGAN/dataset/experiment1/', help='path to dataset folder')
    parser.add_argument('--scale', default=(0,1), type=tuple,
                        help='scale range of the sample (default: 0 to 1)')    
    parser.add_argument('--verbose', '-v', action='store_true', help='make noise')

    return parser.parse_args()

def main(args):
    seed(1)
    end = time.time()
    # load the data
    (X_train, y_train), (X_tst, y_tst) = datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255
    X_tst = X_tst.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    X_train = np.expand_dims(X_train, -1)
    X_tst = np.expand_dims(X_tst, -1)
    if args.verbose:
        print('Loading MNIST Samples.\n Training set size: {}, \n Testing size: {}'.format(X_train.shape, X_tst.shape))

    # setup optimizer
    DC_opt = Adam(lr=1e-5, beta_1=0.5)
    C_opt = Adam(lr=1e-6, beta_1=0.5)

    #create CNN   
    DC_model, C_model = models.semi_cnn(X_train.shape[1:], args.n_classes, args.n_cluster, args.alpha, args.beta, DC_opt, C_opt)
    if args.verbose:
        DC_model.summary()
        C_model.summary()
    #load supervised samples, pre-train the CNN
    X_sup, y_sup = utils.select_supervised_samples((X_train, y_train), args.n_sup, args.n_classes)

    # from label to categorical  
    y_sup = keras.utils.to_categorical(y_sup, args.n_classes)
    y_train = keras.utils.to_categorical(y_train, args.n_classes)
    y_tst = keras.utils.to_categorical(y_tst, args.n_classes)
    #if args.verbose:
    #    print('=>Train the CNN with supervised sampeles with shape:: {}'.format(X_sup.shape, y_sup.shape))
    #    print(y_sup)
    #cnn.fit(X_sup, y_sup, epochs = 100, verbose =0) 

    # clustering algorithm to use
    deepcluster = my_clustering.Kmeans(args.n_cluster)

    #Pre-train model in supervised with some label
    #cnn.fit(X_sup, y_sup, _batch_size, _epochs, verbose)
    #score= cnn.evaluate(tstset[0], tstset[1])
    #print('Test accuracy:', score[1])
    #cnn.fit(dataset[0], dataset[1], args.batch, args.epochs, verbose)

    loss1_history = []
    loss2_history = []
    tst_history = [] 

    # calculate the number of training iterations
    bat_per_epo = int(X_train.shape[0] / args.batch_size)
    n_steps = bat_per_epo * args.epochs
    if args.verbose:
        print('n_epochs=%d, n_batch=%d, b/e=%d, steps=%d' % (args.epochs, args.batch_size, bat_per_epo, n_steps))

    ############################################################    
    ###########training model with semi-DeepCluster#############
    ############################################################
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
        if args.verbose:
            print('=>Cluster the features')
            print(features.shape)
        images_lists = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('=>Assign pseudo labels')           
        train_dataset = my_clustering.cluster_assign(images_lists, X_train)
        X, Y = train_dataset.X, train_dataset.pseudolabels
    
        # train network with clusters as pseudo-labels
        _X_dc, _y_dc = utils.select_supervised_samples([X, Y], args.batch_size, args.n_cluster)   
        _y_dc = keras.utils.to_categorical(_y_dc, args.n_cluster)
        _loss2, _acc2 = DC_model.train_on_batch(_X_dc, _y_dc)
        if args.verbose:
            print('DC_loss:{}'.format(_loss2))
        # summarize   
        if (i+1)%bat_per_epo ==0:
            acc = utils.evaluate_c_model(C_model, X_tst, y_tst)
            print('>> {}'.format(acc))
            tst_history.append(acc)
            loss1_history.append(_loss1)
            loss2_history.append(_loss2)

    print('DC loss history:', loss2_history)
    print('Test history:', tst_history)        
    print('Time: % s\n'%(time.time() - end))

if __name__ == '__main__':
    
    args = parse_args()
    main(args)
