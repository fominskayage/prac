import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    result = []
    sh = w.shape
    w = w.ravel()
    for e_i in np.eye(w.shape[0]):
        result.append(((function(w + eps * e_i) - function(w)) / eps))
    return np.array(result).reshape((sh[0], w.shape[0] + 1 // sh[0]))
