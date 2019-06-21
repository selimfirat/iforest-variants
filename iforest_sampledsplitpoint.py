import numpy as np
from base_algorithm import BaseAlgorithm

class ExternalNode:

    def __init__(self, size):

        self.size = size

class InternalNode:

    def __init__(self, left, right, split_attr, split_val):
        
        self.split_val = split_val
        self.split_attr = split_attr
        self.right = right
        self.left = left

class IForestSampledSplitPoint(BaseAlgorithm):

    name = "iForest_SampledSplitPoint"

    def __init__(self, t=100, psi=256):

        self.t = t
        self.psi = psi
        self.forest = None


    def fit(self, X):

        self.forest = self.iForest(X, self.t, self.psi)


    def anomaly_score(self, x):

        if self.forest is None:
            raise RuntimeError("IForest is not trained yet!")


        total = 0.0
        for tree_idx in range(self.t):

            iTree = self.forest[tree_idx]

            total += self.path_length(x, iTree, 0)

        avg = total / self.t # divide by number of trees

        s = 2 ** (-avg/self.c(self.psi))

        return s

    def predict(self, X):

        anomaly_scores = np.apply_along_axis(self.anomaly_score, axis=1, arr=X)

        return anomaly_scores


    def sample(self, X, psi):
        self.psi = np.minimum(psi, X.shape[0])

        idxs = np.random.choice(range(X.shape[0]), size=self.psi, replace=False)

        return X[idxs]


    def iForest(self, X, t, psi):
        # X: input data (size, attrs)
        # t: number of trees
        # psi: subsampling size

        forest = []

        l = np.ceil(np.log2(psi))

        for i in range(1, t+1):

            X_bar = self.sample(X, psi)

            new_tree = self.iTree(X_bar, 0, l)

            forest.append(new_tree)

        return forest


    def iTree(self, X, e, l):
        # X: input data
        # e: current tree height
        # l: height limit

        if len(X.shape) == 2:
            size = X.shape[0]
        elif X.shape[0] == 0:
            size = 0
        else:
            size = 1

        if e >= l or X.shape[0] <= 1:
            return ExternalNode(size)

        num_attrs = X.shape[1]

        cur_attr = np.random.randint(0, num_attrs)


        # No need to determine minimum/maximum beforehand.

        # split_point = np.random.uniform(min, max) # only this line is different from iForest
        # min = np.min(X[:, cur_attr])
        # max = np.max(X[:, cur_attr])
        cur_sample = np.random.randint(0, size)
        split_point = X[cur_sample, cur_attr]

        mask = X[:, cur_attr] < split_point

        X_l = X[mask, :]
        X_r = X[~mask, :]

        left = self.iTree(X_l, e+1, l)
        right = self.iTree(X_r, e+1, l)

        return InternalNode(left, right, cur_attr, split_point)

    def c(self, size):
        euler_const = 0.5772156649

        r = np.log(size - 1) if size > 1 else 0
        l = 2*(size - 1)/size if size > 1 else 0
        return r + euler_const - l

    def path_length(self, x, T, e):
        # x: an instance
        # T: an iTree
        # e: current path length

        if type(T) is ExternalNode:
            return e + self.c(T.size)

        next_iTree = T.left if x[T.split_attr] < T.split_val else T.right

        return self.path_length(x, next_iTree, e + 1)