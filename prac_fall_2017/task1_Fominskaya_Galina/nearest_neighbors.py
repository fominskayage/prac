import numpy as np


def pairwise_dist_euclidean(x, y):
    x_sq = ((x ** 2).sum(1))[:, np.newaxis]
    y_sq = ((y ** 2).sum(1))[np.newaxis, :]
    xy = x.dot(y.transpose())
    return (x_sq + y_sq - 2 * xy) ** 0.5


def pairwise_dist_cosine(x, y):
    x_sq = (((x ** 2).sum(1))[:, np.newaxis]) ** 0.5
    y_sq = (((y ** 2).sum(1))[np.newaxis, :]) ** 0.5
    xy = x.dot(y.transpose())
    x_sq_y_sq = x_sq * y_sq
    return 1 - xy / (x_sq_y_sq)


class KNNClassifier:
    """docstring for ClassName"""
    def __init__(self, k=3, strategy='brute', metric='cosine', weights=True, test_block_size=128):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.array(list(set(self.y)))
        if self.strategy != 'my_own':
            from sklearn.neighbors import NearestNeighbors as NN
            self.nn = NN(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric)
            self.nn.fit(X, y)

    def find_kneighbors(self, X, return_distance=False):
        if self.strategy != 'my_own':
            return self.nn.kneighbors(X, return_distance=return_distance)
        
        dist = 0
        ind = 0
        
        res_dist = np.empty(0)
        res_ind = np.empty(0)

        while ind < X.shape[0] - self.test_block_size:
            X_tmp = X[ind: ind + self.test_block_size, :]
            dist = 0
            if (self.metric == 'euclidean'):
                dist = pairwise_dist_euclidean(X_tmp, self.X)
            else:
                dist = pairwise_dist_cosine(X_tmp, self.X)
            
            dist_tmp = np.empty((dist.shape[0], self.k))
            ind_tmp = np.empty((dist.shape[0], self.k), dtype=int)
            for i in range(dist.shape[0]):
                dist_tmp[i] = np.sort(np.partition(dist[i], self.k - 1)[:self.k])
                tmp = np.argpartition(dist[i], self.k - 1)
                ind_tmp[i] = tmp[:self.k][np.argsort(np.partition(dist[i], self.k - 1)[:self.k])]
            if res_dist.shape[0] == 0 or res_dist.shape[1] == 0:
                res_dist = res_dist_tmp.copy()
                res_ind = res_ind_tmp.copy()
            
            else:
                res_dist = np.vstack((res_dist, dist_tmp))
                res_ind = np.vstack((res_ind, ind_tmp))
            ind = ind + self.test_block_size
        
        X_tmp = X[ind:, :]
        dist = 0
        if (self.metric == 'euclidean'):
            dist = pairwise_dist_euclidean(X_tmp, self.X)
        else:
            dist = pairwise_dist_cosine(X_tmp, self.X)
        
        dist_tmp = np.empty((dist.shape[0], self.k))
        ind_tmp = np.empty((dist.shape[0], self.k), dtype=int)
        for i in range(dist.shape[0]):
            dist_tmp[i] = np.sort(np.partition(dist[i], self.k - 1)[:self.k])
            tmp = np.argpartition(dist[i], self.k - 1)
            ind_tmp[i] = tmp[:self.k][np.argsort(np.partition(dist[i], self.k - 1)[:self.k])]
        if res_dist.shape[0] == 0 or res_dist.shape[1] == 0:
            res_dist = dist_tmp.copy()
            res_ind = ind_tmp.copy()
        
        else:
            res_dist = np.vstack((res_dist, dist_tmp))
            res_ind = np.vstack((res_ind, ind_tmp))
        
        if return_distance:
            return (res_dist, res_ind)
        return res_ind

    def predict(self, X):
        if self.weights:
            dist, neighbors = self.find_kneighbors(X=X, return_distance=True)
            weights = (dist + (10 ** (-5))) ** (-1)
        
        else:
            neighbors = self.find_kneighbors(X=X, return_distance=False)
            weights = np.ones(neighbors.shape)
        res = np.empty(weights.shape[0])
        for i in range(X.shape[0]):
            tmp = np.empty_like(self.classes)
            for ind in range(self.classes.shape[0]):
                tmp[ind] = weights[i, :][self.y[neighbors[i, :]] == self.classes[ind]].sum()
            res[i] = self.classes[np.argmax(tmp)]
        return res
