import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def evaluate_c_model(c_model, X, Y):
    '''
    Input: c_model (compiled keras model), X (np array), Y (np array)

    output: Testing accuracy (%)
    '''
    _, acc = c_model.evaluate(X, Y, verbose=0)

    return acc * 100 

def select_supervised_samples(dataset, n_samples, n_classes):
    X, Y = dataset
    X_list, Y_list = list(), list()
    n_per_class = int(n_samples/n_classes)
    for i in range(n_classes):
        X_with_class = X[Y==i]
        ix = np.random.randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [Y_list.append(i) for j in ix]

    return np.asarray(X_list), np.asarray(Y_list)

def single_minmaxscale(data, scale_range):
    def minmaxscale(data, scale_range):
        scaler = MinMaxScaler(scale_range)
        scaler.fit(data)
        normalized = scaler.transform(data)
        return normalized

    X = []
    for i in data:
        X.append(minmaxscale(i.reshape(-1,1), scale_range))
    return np.asarray(X)     


def data_preproc(dataset, scale_range = (-1, 1)):
    X_tra, y_tra, X_tst, y_tst = dataset
    X_tra = single_minmaxscale(X_tra, scale_range)
    X_tst = single_minmaxscale(X_tst, scale_range)

    X_tra = X_tra.astype('float32')
    X_tra = X_tra.reshape(-1,1,120,1)
    X_tst = X_tst.astype('float32')
    X_tst = X_tst.reshape(-1,1,120,1)
    print('Finished preprocessing.')
    return (X_tra, y_tra, X_tst, y_tst)






