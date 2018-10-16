import numpy as np
import scipy


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

        
class BinaryHinge(BaseSmoothOracle):
    """
    Оракул для задачи двухклассового линейного SVM.
    """
    
    def __init__(self, C=1.0):
        """
        Задание параметров оракула.
        """
        self.C = C
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        weight_part = 0.5 * w[1:].dot(w[1:])
        sum_part = (np.vstack([np.zeros_like(y), 1 - y * X.dot(w)]).max(axis=0))
        return weight_part + (self.C / X.shape[0]) * sum_part.sum()

    def grad(self, X, y, w):
        """
        Вычислить субградиент функционала в точке w на выборке X с ответами y.
        Субгрдиент в точке 0 необходимо зафиксировать равным 0.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        l = X.shape[0]
        rand_part = np.zeros_like(X)
        mask = (1 > y * X.dot(w))
        rand_part[mask] = (- y[:, np.newaxis] * X)[mask]
        return np.hstack((0, w[1:])) + (self.C / l) * rand_part.sum(axis=0)
