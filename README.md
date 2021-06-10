# Semi-Supervised Learning with Deep Clustering (DC) algorithm for Device-Free Fingerprinting Indoor Localization
######  Last update: 6/10/2021
## Introduction:
Implementation of Deep Clustering for Device Free Wi-Fi Fingerprinting Indoor Localization. 

The Deep Clustering code is inherited and modified from [here](https://github.com/facebookresearch/deepcluster).

## Concept:
<img src="https://github.com/aciculachen/CSI-DeepClustering/blob/master/overview.png" width="600">

## Features:

- **main.py**: train the semi-supervised DC model under the pre-defined indoor localization scenarios.
- **deep_clustering.py**: Implementation of unsupervised DC algorithm with keras according to [here](https://github.com/facebookresearch/deepcluster).
- **my_clustering.py**: Implementation of k-means according to [here](https://github.com/facebookresearch/deepcluster).
- **my_k_means.py**: Implementation of a semi-supervised k-means
- **test_on_mnist.py**: semi-supervised DC with MNIST
- **models.py**: definded models
- **dataset**: pre-collected CSI samples save as pickle in the form of (X_train, y_train, X_tst, y_tst)
## Dependencies:
- tensorflow 2.0
- python 3.6.4
- keras 2.15
