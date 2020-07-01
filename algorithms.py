from scipy.spatial.distance import pdist, squareform
import scipy
import numpy as np


class SVGD(object):
    def __init__(self, start_h, start_iter, dlnp_fun):
        self.dlnp_fun = dlnp_fun
        self.adagrad = AdaGrad(start_h, start_iter)

    def update(self, x: np.array, n_iters, step_size):
        """
        :param x: N x D array where N is number of points and D is its dimension.
        :return: updated N x D array.
        """
        for _ in range(n_iters):
            x = self.adagrad.update(x, -self.phi(x), step_size)
        return x

    def phi(self, x: np.array):
        """
        :param x: N x D array where N is number of points and D is its dimension.
        :return: N x D array whose ith row is phi(x_i).
        """
        assert len(x.shape) == 2
        k, h = self._rbf_kernel(x)
        dlnp = self.dlnp_fun(x)
        res = np.zeros(x.shape)
        n = x.shape[0]
        for i in range(n):
            xi = np.tile(x[i], (n, 1))
            res[i, :] = np.matmul(k[:, i], 2 / h * (xi - x) + dlnp) / n
        return res

    def _rbf_kernel(self, x: np.array):
        """
        :param x: N x D array where N is number of points and D is its dimension.
        :return: N x N array whose (i, j)-entry is k(x_i, x_j), h
        """
        assert x.shape[0] > 1
        pairwise_dists = squareform(pdist(x))
        med = np.median(pairwise_dists)
        h = med ** 2 / np.log(x.shape[0])
        k = scipy.exp(-pairwise_dists ** 2 / h)
        return k, h


class AdaGrad(object):
    def __init__(self, start_h, start_iter, auto_corr=0.95, eps=1e-7):
        self.h = start_h
        self.iter = start_iter
        self.auto_corr = auto_corr
        self.eps = eps

    def update(self, w, grad, step_size):
        if self.iter == 0:
            self.h = grad**2
        else:
            self.h = self.auto_corr*self.h + (1-self.auto_corr)*grad**2
        self.iter += 1
        adjusted_grad = grad / (np.sqrt(self.h) + self.eps)
        return w - adjusted_grad*step_size
