import math
import time 

import faiss
import numpy as np
from matplotlib import pyplot as plt

import utils
import main
import tensorflow as tf

class ReassignedDataset():
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform
        self.pseudolabels = np.asarray(pseudolabels)
        self.X = np.asarray(self.get_X(image_indexes, dataset))

    #def make_dataset(self, image_indexes, pseudolabels, dataset):
        #label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        #images = []
        #for j, idx in enumerate(image_indexes):
            #path = dataset[idx]
            #pseudolabel = label_to_idx[pseudolabels[j]]
            #images.append((path ,pseudolabel))
        #return images

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path ,pseudolabel))
        return images

    def get_X(self, image_indexes, dataset):
        """
        2021/3/3
        """
        X = []
        for idx in image_indexes:
            X.append(dataset[idx])
        return X

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = path
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # x PCA-reducing, whitening and L2-normalization
        xb = data

        # cluster the data
        I = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        #if verbose:
            #print('k-means time: {0:.0f} s'.format(time.time() - end))

        return self.images_lists     


def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     #std=[0.229, 0.224, 0.225])
    #t = transforms.Compose([transforms.RandomResizedCrop(224),
                            #transforms.RandomHorizontalFlip(),
                            #transforms.ToTensor(),
                            #normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset)

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1)
    clus.niter = 20
    clus.max_points_per_centroid = 256
    #res = faiss.StandardGpuResources()
    #flat_config = faiss.GpuIndexFlatConfig()
    #flat_config.useFloat16 = False
    #flat_config.device = 0
    #index = faiss.GpuIndexFlatL2( d)
    index = faiss.IndexFlatL2( d)
    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    #losses = faiss.vector_to_array(clus.obj)
    #if verbose:
    #    print('Kmeans finished')

    return [int(n[0]) for n in I]

