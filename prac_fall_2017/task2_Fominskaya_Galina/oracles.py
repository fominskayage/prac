import numpy as np
import scipy
from scipy import sparse
from scipy.special import expit
from scipy.special import logsumexp


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

        
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """
    
    def __init__(self, l2_coef=1e-5):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        
        block_size = 256

        loss = np.logaddexp(0, - y * X.dot(w)).sum()
        reg = self.l2_coef * 0.5 * (w ** 2).sum()
        return (1 / X.shape[0]) * loss + reg

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        tmp = X.dot(w)
        tmp *= -y
        tmp = expit(tmp)
        tmp *= -y
        return X.transpose().dot(tmp).transpose() / y.shape[0] + self.l2_coef * w
    
    
class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    
    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """
    
    def __init__(self, class_number=None, l2_coef=1):
        """
        Задание параметров оракула.
        
        class_number - количество классов в задаче
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.class_number = class_number
        self.l2_coef = l2_coef
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        if self.class_number is None:
            self.class_number = w.shape[0]

        first_for_loss = - X.dot(w.transpose())[:, y].diagonal().sum()

        sum_for_loss = logsumexp(X.dot(w.transpose()).transpose(), axis=0).sum()
        
        loss = (1 / X.shape[0]) * (sum_for_loss + first_for_loss)

        reg = self.l2_coef * 0.5 * (w ** 2).sum()

        return loss + reg
        
    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        if self.class_number is None:
            self.class_number = w.shape[0]
        mask = y[np.newaxis, :] == np.arange(self.class_number)[:, np.newaxis]
            
        first_sum = (X.transpose().dot(mask.transpose())).transpose()
        a_max = X.dot(w.transpose()).max(axis=1)[:, np.newaxis]

        tmp_h = np.exp(X.dot(w.transpose()) - a_max)

        tmp_l = (np.exp(X.dot(w.transpose()) - a_max)).sum(axis=1)

        tmp_next = X.transpose().dot(tmp_h / tmp_l[:, np.newaxis]).transpose()
        return - (first_sum - tmp_next) / X.shape[0] + self.l2_coef * w
