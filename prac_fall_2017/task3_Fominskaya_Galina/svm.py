from cvxopt import matrix
from cvxopt import solvers
import numpy as np


class SVMSolver:
    """
    Класс с реализацией SVM через метод внутренней точки.
    """
    def __init__(self, C, method, kernel='linear', gamma=None, degree=None):
        """
        C - float, коэффициент регуляризации
        
        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        gamma - ширина rbf ядра, только если используется rbf-ядро
        d - степень полиномиального ядра, только если используется полиномиальное ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """
        self.C = C
        self.method = method
        if method == 'dual':
            self.kernel = kernel
            if self.kernel == 'rbf':
                self.gamma = gamma
            elif self.kernel == 'polynomial':
                self.degree = degree
        else:
            self.kernel = None
    
    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        if self.method is 'dual':
            raise Exception('Not able to compute primal objective for dual.')
        if self.w is None:
            raise Exception('Not trained yet.')
        margins = y.T * (X.dot(self.w.T) + self.w_0)
        ksi = np.max(np.vstack((1 - margins, np.zeros_like(margins))), axis=0)
        return 0.5 * self.w.T.dot(self.w) + (self.C / X.shape[0]) * ksi.sum()
        
    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """ 
        if self.method is 'primal':
            raise Exception('Not able to compute dual objective for primal.')
        if self.kernel == 'linear':
            P = y[:, np.newaxis] * y[np.newaxis, :] * (X.dot(X.T))
        if self.kernel == 'polynomial':
            P = y[:, np.newaxis] * y[np.newaxis, :] * ((X.dot(X.T) + 1) ** self.degree)
        if self.kernel == 'rbf':
            y_tmp = y[:, np.newaxis] * y[np.newaxis, :]
            tmp = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
            P = y_tmp * (np.exp(-self.gamma * (tmp ** 2).sum(axis=2)))
        return 0.5 * (self.lambdas * (P.dot(self.lambdas))).sum() - self.lambdas.sum()
        
    def fit(self, X, y, tolerance=10e-7, max_iter=100):
        """
        Метод для обучения svm согласно выбранной в method задаче
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе
        """
        d = X.shape[1]
        l = X.shape[0]
        solvers.options['maxiters'] = max_iter
        solvers.options['reltol'] = tolerance
        
        if self.method == 'primal':
            P = np.zeros((d + l + 1, d + l + 1))
            P[np.arange(1, d + 1), np.arange(1, d + 1)] = 1
            q = np.zeros((d + l + 1))
            q[d + 1:] = self.C / l
            G = np.zeros((2 * l, d + l + 1))
            G[: l, : d + 1] = - y[:, np.newaxis] * np.hstack((np.ones((l, 1), dtype=float), X))
            G[np.arange(l), np.arange(d + 1, d + l + 1)] = -1
            G[np.arange(l, 2 * l), np.arange(d + 1, d + l + 1)] = -1
            
            h = - np.zeros(2 * l)
            h[:l] = -1

            tmp = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

            self.w_0 = np.array(tmp['x'][0])
            self.w = np.array(tmp['x'][1: d + 1]).reshape(d)

        elif self.method == 'dual':
            if self.kernel == 'linear':
                P = y[:, np.newaxis] * y[np.newaxis, :] * (X.dot(X.T))

            if self.kernel == 'polynomial':
                P = y[:, np.newaxis] * y[np.newaxis] * ((X.dot(X.T) + 1) ** self.degree)
            if self.kernel == 'rbf':
                first = y[:, np.newaxis] * y[np.newaxis]
                second = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                P = first * (np.exp(-self.gamma * (second ** 2).sum(axis=2)))
            
            q = - np.ones(l)
            
            G = np.eye(2 * l, l)
            G[np.arange(l, 2 * l), np.arange(l)] = -1
            
            h = np.zeros((2 * l))
            h[:l] = self.C / l
            
            A = y[np.newaxis, :].astype(float)
            b = np.zeros((1, 1))

            tmp = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))

            self.lambdas = np.array(tmp['x']).ravel()
            self.support_vectors = X[self.lambdas > 10e-7]
            self.support_answers = y[self.lambdas > 10e-7]
            self.support_lambdas = self.lambdas[self.lambdas > 10e-7]

            if self.kernel == 'linear':
                self.w = (self.support_lambdas * self.support_answers).dot(self.support_vectors)
                self.w_0 = - self.support_vectors[0].dot(self.w) + self.support_answers[0]   

            if(self.kernel == 'rbf'):
                first = self.support_answers * self.support_lambdas
                mask = (self.support_lambdas < self.C / (X.shape[0]))
                second = self.support_vectors[mask][0][np.newaxis, :]
                tmp = ((self.support_vectors[:, np.newaxis] - second) ** 2).sum(axis=2)
                third = self.support_answers[mask][0]
                self.w_0 = -1 * first.dot((np.exp(-self.gamma * (tmp)))) + third

            if(self.kernel == 'polynomial'):
                first = self.support_answers * self.support_lambdas
                tmp = (self.support_vectors.dot(self.support_vectors[0]) + 1) ** self.degree
                self.w_0 = - first.dot(tmp) + self.support_answers[0]

    def predict(self, X):
        """
        Метод для получения предсказаний на данных
        
        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """
        if self.method == 'primal' or self.kernel == 'linear':
            return np.sign(X.dot(self.get_w()) + self.get_w0())

        if self.kernel == 'rbf':
            first = self.support_answers * self.support_lambdas
            tmp = ((self.support_vectors[:, np.newaxis] - X[np.newaxis, :]) ** 2).sum(axis=2)
            return np.sign(first.dot((np.exp(-self.gamma * tmp))) + self.w_0)
 
        if(self.kernel == 'polynomial'):
            first = self.support_answers * self.support_lambdas
            tmp = first.dot((self.support_vectors.dot(X.T) + 1) ** self.degree)
            return np.sign(tmp + self.w_0)
  
    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: одномерный numpy array
        """
        if self.w is None:
            raise Exception('Not trained yet.')
        return self.w
        
    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига
        
        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 
        
        return: float
        """
        if self.w_0 is None:
            raise Exception('Not trained yet.')
        return self.w_0
        
    def get_dual(self):
        """
        Получить двойственные переменные
        
        return: одномерный numpy array
        """
        if self.method == 'primal':
            raise Exception('Not able to compute dual variables for dual.')
        elif self.method == 'dual':
            if self.lambdas is None:
                raise Exception('Not trained yet.')
            return self.lambdas
