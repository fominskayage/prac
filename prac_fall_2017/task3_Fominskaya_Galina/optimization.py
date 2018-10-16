import oracles
import timeit
import numpy as np
from random import randint, sample
from scipy.special import expit
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from scipy import sparse


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=10000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        self.loss_function = loss_function
        if loss_function == 'binary_logistic':
            self.oracle = oracles.BinaryLogistic(**kwargs)
        elif loss_function == 'binary_hinge':
            self.oracle = oracles.BinaryHinge(**kwargs)
        else:
            self.oracle = oracles.MulticlassLogistic(**kwargs)

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        
    def fit(self, X, y, w_0=None, trace=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        history = {'time': [], 'func': [0], 'cum_time': [], 'accuracy': []}

        self.w = w_0

        if self.w is None:
            if self.loss_function == 'binary_logistic' or self.loss_function == 'binary_hinge':
                self.w = np.zeros(X.shape[1], dtype=float)
            else:
                self.w = np.zeros((np.unique(y).shape[0], X.shape[1]), dtype=float)
        count = 1
        
        grad = self.get_gradient(X, y)

        eta = self.step_alpha * (count ** -self.step_beta)

        w_next = self.w - eta * grad
        
        tol = self.tolerance

        history['func'].append(self.get_objective(X, y))

        while (abs(history['func'][-2] - history['func'][-1]) >= tol) and (count <= self.max_iter):
            start_time = timeit.default_timer()
            
            self.w = w_next.copy()
            if not (X_test is None):
                y_pred = self.predict(X_test)
                history['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_pred))

            grad = self.get_gradient(X, y)
            
            count += 1

            eta = self.step_alpha * (count ** -self.step_beta)

            w_next -= eta * grad
            
            history['func'].append(self.get_objective(X, y))

            history['time'].append(timeit.default_timer() - start_time)

        if trace:
            history['cum_time'] = np.cumsum(history['time'])
            history['func'] = history['func'][1:]
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        if self.w is None:
            raise Exception('Not trained yet')
        if self.loss_function == 'binary_logistic' or self.loss_function == 'binary_hinge':
            res = self.predict_proba(X).argmax(axis=1)
            res[res == 0] = -1
            return res
        else:
            return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        if self.w is None:
            raise Exception('Not trained yet')
        if self.loss_function == 'binary_logistic' or self.loss_function == 'binary_hinge':
            res = expit(X.dot(self.w))
            return np.vstack((1 - res, res)).transpose()
        else:
            a_max = X.dot(self.w.transpose()).max(axis=1)[:, np.newaxis]
            tmp_h = np.exp(X.dot(self.w.transpose()) - a_max)
            tmp_l = (np.exp(X.dot(self.w.transpose()) - a_max)).sum(axis=1)
            
            return tmp_h / tmp_l[:, np.newaxis]

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        if self.w is None:
            raise Exception('Not trained yet')
        return self.oracle.func(X, y, self.w)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        if self.w is None:
            raise Exception('Not trained yet')
        
        return self.oracle.grad(X, y, self.w)
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """
        if self.w is None:
            raise Exception('Not trained yet')
        return self.w 

    
class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, batch_size=1, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=10000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        
        max_iter - максимальное число итераций
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        lf = loss_function
        super().__init__(lf, step_alpha, step_beta, tolerance, max_iter, **kwargs)

        self.batch_size = batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями 
        приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет 
        записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между 
        двумя соседними замерами
        history['func']: list of floats, содержит значения функции после 
        текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы 
        разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)

        history = {'epoch_num': [], 'time': [], 'func': [0], 'cum_time': [], 'accuracy': []}

        self.w = w_0
        
        if self.w is None:
            if self.loss_function == 'binary_logistic' or self.loss_function == 'binary_hinge':
                self.w = np.zeros(X.shape[1], dtype=float)
            else:
                self.w = np.zeros((np.unique(y).shape[0], X.shape[1]), dtype=float)

        count = 1

        indexes = np.random.permutation(X.shape[0])[:self.batch_size]

        grad = self.get_gradient(X[indexes], y[indexes])
        
        eta = self.step_alpha * (count ** -self.step_beta)

        w_next = self.w - eta * grad
        
        tol = self.tolerance
        
        history['epoch_num'].append(count * self.batch_size // X.shape[0])
        prev_func = 0
        curr_func = self.get_objective(X, y)
        
        history['func'].append(curr_func)

        set_timer = True

        while (abs(prev_func - curr_func) >= tol) and (count <= self.max_iter):
            if set_timer:
                start_time = timeit.default_timer()
                set_timer = False

            self.w = w_next.copy()
            indexes = np.random.permutation(X.shape[0])[:self.batch_size]

            grad = self.get_gradient(X[indexes], y[indexes])

            count += 1

            eta = self.step_alpha * (count ** -self.step_beta)

            w_next -= eta * grad
            
            if(count * self.batch_size > X.shape[0] * (log_freq + history['epoch_num'][-1]) + 1):
                if not (X_test is None):
                    y_pred = self.predict(X_test)
                    history['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_pred))

                history['epoch_num'].append((count * self.batch_size + 1) // X.shape[0])
                history['func'].append(self.get_objective(X, y))
                history['time'].append(timeit.default_timer() - start_time)
                set_timer = True
            prev_func = curr_func
            curr_func = self.get_objective(X, y)
        print(count)
        if trace:
            history['cum_time'] = np.cumsum(history['time'])
            history['func'] = history['func'][1:]
            return history


class PEGASOSMethod:
    """
    Реализация метода Pegasos для решения задачи svm.
    """
    def __init__(self, step_lambda, batch_size, num_iter, **kwargs):
        """
        step_lambda - величина шага, соответствует 
        
        batch_size - размер батча
        
        num_iter - число итераций метода, предлагается делать константное
        число итераций 
        """
        self.step_lambda = step_lambda
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.oracle = oracles.BinaryHinge(**kwargs)
        
    def fit(self, X, y, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        if(type(X) == sparse.csr_matrix):
            X = sparse.hstack((np.ones((X.shape[0], 1)), X), format='csr')
        else:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        curr_w = np.zeros(X.shape[1])
        history = {'time': [0], 'func': [self.get_objective(X, y, curr_w)]}
        best_w = curr_w.copy()
        curr_iter = 1
        F_best = self.get_objective(X, y, curr_w)

        while curr_iter < self.num_iter:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            indices = np.array_split(indices, X.shape[0] / self.batch_size)

            for inds in indices:
                start_time = timeit.default_timer()

                X_tmp = X[inds]
                y_tmp = y[inds]
                
                tmp = 1 / (curr_iter * self.step_lambda)
                mask = y_tmp * curr_w.dot(X_tmp.T) < 1
                first = (1 - tmp * self.step_lambda) * curr_w 
                new_w = first + (tmp / inds.shape[0]) * (y_tmp[mask].dot(X_tmp[mask, :]))
                new_w = min(1, (((new_w ** 2).sum() ** -0.5) * self.step_lambda ** -0.5)) * new_w
                
                if F_best > self.get_objective(X, y, new_w):
                    best_w = new_w
                    F_best = self.get_objective(X, y, new_w)
                
                curr_w = new_w
                curr_iter += 1
                
                history['func'].append(self.get_objective(X, y, new_w))
                history['time'].append(timeit.default_timer() - start_time)
                
        self.w = best_w
        if trace:
            return history

    def predict(self, X):
        """
        Получить предсказания по выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        """
        if self.w is None:
            raise Exception('Not trained yet.')

        if(type(X) == sparse.csr_matrix):
            X = sparse.hstack((np.ones((X.shape[0], 1)), X), format='csr')
        else:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

        res = X.dot(self.w.T)
        res[res >= 0] = 1
        res[res < 0] = -1
        return res

    def func(self, X, y):
        if self.w is None:
            raise Exception('Not trained yet.')
        if(type(X) == sparse.csr_matrix):
            X = sparse.hstack((np.ones((X.shape[0], 1)), X), format='csr')
        else:
            X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

        return self.oracle.func(X, y, self.w)

    def get_objective(self, X, y, w):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.oracle.func(X, y, w)
