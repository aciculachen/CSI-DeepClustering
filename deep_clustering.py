#Trying to implement a clustering-based semi-supervised CNN with MNIST and pytorch. 2020/01/27

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
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorboard.plugins import projector

import my_clustering 
import utils 
from models import *

def parse_args():
    parser = argparse.ArgumentParser(description='Keras Implementation of Semi-DeepCluster')
    #parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--n_cluster', '--k', type=int, default=16,
                        help='number of cluster for k-means (default: 16)')
    parser.add_argument('--alpha', '--a', type=float, default=0.1,
                        help='parameter that controls the role of supervsied CNN (default: 0.1)')  
    parser.add_argument('--epochs', '--e',type=int, default=20,
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('--batch_size','--b', default=16, type=int,
                        help='mini-batch size (default: 16)')
    parser.add_argument('--n_classes', '--c', type=int, default=16,
                        help='number of classes (default: 16)')
    parser.add_argument('--n_sup', type=int, default=16,
                        help='number of supervsied data (default: 16)')
    parser.add_argument('--seed', type=int, default=3,
                        help='random seed (default: 3)')                            
    parser.add_argument('--path2data', type=str, default='dataset/', help='path to dataset folder')
    parser.add_argument('--scale', default=(0,1), type=tuple,
                        help='scale range of the sample (default: 0 to 1)')    
    parser.add_argument('--verbose', '-v', action='store_true', help='make noise')

    return parser.parse_args()

def main(args):
    ###################### setup TensorBoard ######################
    NAME = "Deep-Clustering-{}".format(int(time.time()))
    logdir = 'log/{}'.format(NAME)
    train_writer = tf.summary.create_file_writer(logdir)
    ###############################################################

    # fix random seeds
    seed(args.seed)

    # load the data
    X_tra, y_tra, X_tst, y_tst = utils.data_preproc(np.asarray(pickle.load(open('dataset/EXP1.pickle','rb'))))
    if args.verbose:
        print('Loading CSI Samples from {} \n Training set size: {}, \n Testing size: {}'.format(args.path2data, X_tra.shape, X_tst.shape))  

    # setup optimizer
    opt= Adam(lr=0.0002, beta_1=0.5)    

    # create CNN
    DC_model = cnn(X_tra.shape[1:], args.n_cluster, opt)
    if args.verbose:
        DC_model.summary()

    # clustering algorithm to use
    deepcluster = my_clustering.Kmeans(args.n_cluster)

    # calculate the number of training iterations
    bat_per_epo = int(X_tra.shape[0] / args.batch_size)
    n_steps = bat_per_epo * args.epochs
    if args.verbose:
        print('n_epochs=%d, n_batch=%d, b/e=%d, steps=%d' % (args.epochs, args.batch_size, bat_per_epo, n_steps))

    ############################################################    
    ##################### Start DeepCluster#####################
    ############################################################
    with train_writer.as_default(): 
        for i in range(n_steps):
            end = time.time()
            if args.verbose:
                print('=>Step.{}/ {}'.format(i+1, n_steps))
         
            #remove DC_models' fc layer 
            features_model = Model(DC_model.input, DC_model.layers[-4].output)
            # get features for the whole dataset
            features = features_model.predict(X_tra)
        
            # cluster the features
            #if args.verbose:
                #print('=>Cluster the features')
            images_lists = deepcluster.cluster(features, verbose=args.verbose)

            # assign pseudo-labels
            #if args.verbose:
                #print('=>Assign pseudo labels')           
            train_dataset = my_clustering.cluster_assign(images_lists, X_tra)
            _X , _Y= train_dataset.X, train_dataset.pseudolabels
            _X_tra , _X_tst, _Y_tra , _Y_tst = train_test_split(_X , _Y, test_size=0.1)
            # train network with clusters as pseudo-labels
            _X_dc, _y_dc = utils.select_supervised_samples([_X_tra , _Y_tra], args.batch_size, args.n_cluster)
            _y_dc = keras.utils.to_categorical(_y_dc, args.n_cluster)
            _Y_tst = keras.utils.to_categorical(_Y_tst, args.n_cluster)
            _loss, _acc = DC_model.train_on_batch(_X_dc, _y_dc)


            if args.verbose:
               print('DC_loss:{}'.format(_loss))
            # summarize   
            if (i+1)%bat_per_epo ==0:
                _acc = utils.evaluate_c_model(DC_model, _X_tst , _Y_tst)
                print('Step.{}/ {} Validation Acc:{} %, loss: {}'.format(i+1, n_steps, _acc, _loss))
                tf.summary.scalar("DC Evaluate Accuracy", _acc, step=i)
                #　save model 
                #DC_model.save('Trained-models/DC_{}.h5'.format(i+1))
                # save crosponding pY
                #with open('pY_{}.pickle'.format(i+1), 'wb') as handle:
                    #pickle.dump(_Y, handle)
    #save
    train_writer.flush() 
    train_writer.close()

    #print('Best Test:', max(tst_history))           
    #print('Time: % s\n'%(time.time() - end))


if __name__ == '__main__':

    args = parse_args()
    main(args)