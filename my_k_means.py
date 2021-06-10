from collections import defaultdict
from random import uniform
import math 
import time 

import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

import utils

def load_real_samples(rootPath, scale_range=(0,1)):
    Train_X = np.asarray(pickle.load(open(rootPath + "Train_X.pickle",'rb')))
    Train_Y = np.asarray(pickle.load(open(rootPath + "Train_Y.pickle",'rb')))

    return [Train_X, Train_Y]

class semi_Kmeans():
    '''
    Dataset: List[X, Y], non-modified raw CSIs with corresponding labels
    X: np array with size: number of class X number of csi per class X CSI smaples (1, 120, 1), e.g. (16, 1, 120, 1) 
    Y: np arary with size: number of label X 
    k: int(k) for kmeans, which equals to class in this setup

    '''
    def __init__(self, k):
        self.k = k
        self.n_per_class = 1
        #self.sup_X, self.sup_y, self.unsup_X = select_supervised_samples(dataset, self.n_per_class, self.k)
        #self.assignments = self.run_kmeans(unsup_X, sup_X)

    def run_kmeans(self, unsup_X, sup_X):
        end = time.time()
        assignments = assign_points(unsup_X, sup_X)
        old_assignments = None
        while assignments != old_assignments:
            new_centroids = update_centroids(unsup_X, assignments)
            old_assignments = assignments
            assignments = assign_points(unsup_X, new_centroids)
        print('K-Means Time: % s\n'%(time.time() - end))

        #X = np.concatenate((np.asarray(unsup_X), sup_X), axis=0)
        #Y = np.concatenate((np.asarray(assignments), sup_y), axis=0)
        return np.asarray(assignments)

    def make_dataset(self):
        X = np.concatenate((unsup_X, sup_X), axis=0)
        y = np.concatenate((self.assignments, sup_y), axis=0)
        X, y = shuffle(X, y, random_state=0)

        return [X, y]



def select_supervised_samples(dataset, n_per_class, n_classes):
    """
    Given `data_set`, which is an np array of np arrays,
    Generate `k` random points from.
    Return an array of the random points within each class.
    """
    X, Y = dataset
    X_list, Y_list = list(), list()
    X_list_c, Y_list_c = list(), list()
    for i in range(n_classes):
        X_with_class = X[Y==i]
        ix = np.random.randint(0, len(X_with_class), n_per_class)
        for j in ix:
         X_list.append(X_with_class[j])
         tst = np.delete(X_with_class, j, axis=0)
         X_list_c.append(tst)
        #[X_list.append(X_with_class[j]) for j in ix]
        [Y_list.append(i) for j in ix]

    return np.asarray(X_list), np.asarray(Y_list), np.asarray(X_list_c).reshape(-1, X.shape[1])

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])
    new_centroid = []

    for dimension in range(dimensions): #dimension 120
        dim_sum = 0  # dimension sum
        for p in points: # for all point withn the same cluster
            dim_sum += p[dimension]

        # average of each dimension
        new_centroid.append(dim_sum / float(len(points)))
    return new_centroid


def update_centroids(data_set, assignments):
    """
    """
    print('update_centroids')
    new_means = defaultdict(list)
    centroids = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        centroids.append(point_avg(points))

    return centroids


def assign_points(data_set, centroids):
    print('assign_points')
    assignments = []
    for point in data_set:
        shortest = math.inf  # positive infinity
        shortest_index = 0
        for i in range(len(centroids)):
            val = distance(point, centroids[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Input:
    a: np array (flatten)
    b: np array (flatten)
    Outpit:
    Euclidean distance between a and b 
    """
    dist = np.sqrt(np.sum((a - b) ** 2))

    return dist

