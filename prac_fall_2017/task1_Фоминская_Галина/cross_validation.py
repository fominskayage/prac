import numpy as np
from nearest_neighbors import KNNClassifier


def ith_fold(n, n_folds, i, indexes):
    if (i + 1 != n_folds):
        first_part_of_train_fold = indexes[np.arange(0, i * (n // n_folds), dtype=int)]
        second_part_of_train_fold = indexes[np.arange((i + 1) * (n // n_folds), n, dtype=int)]
        test_fold = indexes[np.arange(i * (n // n_folds), (i + 1) * (n // n_folds), dtype=int)]
        return (np.hstack((first_part_of_train_fold, second_part_of_train_fold)), test_fold)
    train_fold = indexes[np.arange(0, i * (n // n_folds), dtype=int)]
    test_fold = indexes[np.arange(i * (n // n_folds), n, dtype=int)]
    return (train_fold, test_fold)


def kfold(n, n_folds):
    inds = np.arange(n, dtype=int)
    np.random.shuffle(inds)
    res = []
    for i in np.arange(n_folds, dtype=int):
        fold_0 = np.sort(ith_fold(n, n_folds, i, inds)[0])
        fold_1 = np.sort(ith_fold(n, n_folds, i, inds)[1])
        res += [(fold_0, fold_1)]
    return res


def online_predict(weights, y, k, neighbors, classes, fold):
    res = np.empty(neighbors.shape[0], dtype=int)
    for i in range(neighbors.shape[0]):
        res[i] = np.argmax(np.bincount(y[neighbors[i]], weights[i]))
    return res


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 3)
    ans = {}
    for k in k_list:
        ans[k] = np.empty(len(cv))
    knn = KNNClassifier(k=k_list[-1], **kwargs)
    for index, fold in enumerate(cv):
        knn.fit(X[fold[0], :], y[fold[0]])
        curr_dist, curr_neighbors = knn.find_kneighbors(X=X[fold[1]], return_distance=True)
        if knn.weights:
            weights = (curr_dist + (10 ** (-5))) ** (-1)
        else:
            weights = np.ones(curr_neighbors.shape)
        classes = np.unique(y[fold[0]])
        for k in k_list[-1::-1]:
            curr_weights = weights[:, :k]
            curr_neighbors = curr_neighbors[:, :k]
            res = online_predict(curr_weights, y[fold[0]], k, curr_neighbors, classes, fold)
            ans[k][index] = np.ones(res.shape[0], dtype=int)[res == y[fold[1]]].sum() \
                / res.shape[0]
    return ans
