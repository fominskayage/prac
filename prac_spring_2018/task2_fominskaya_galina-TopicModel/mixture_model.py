import numpy as np
from numpy.linalg import slogdet, det, solve
from math import pi
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal
import timeit

eps = np.finfo(float).eps

class MixtureModel:
    def __init__(self, n_components, diag=False):
        """
        Parametrs:
        ---------------
        n_components: int
        The number of components in mixture model

        diag: bool
            If diag is True, covariance matrix is diagonal
        """
        self.n_components = n_components
        # bonus part
        self.diag = diag
        
    def _E_step(self, data):
        """
        E-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """    
        # set self.q_z
        N, d = data.shape
        
        tmp_N_i_j = np.empty((data.shape[0], self.n_components))
        for j in range(self.n_components):
            tmp = multivariate_normal.logpdf(data, self.Mean[j], self.Sigma[j])
            tmp_N_i_j[:, j] = np.exp(tmp - tmp.max())
        mixture = self.w[np.newaxis, :] * tmp_N_i_j + eps
        self.q_z = (mixture / (mixture.sum(axis=1)[:, np.newaxis]))
        if self.diag:
            # bonus part
            pass
        else:
            pass
                
    def _M_step(self, data):
        """
        M-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """
        N, d = data.shape
        

        tmp_N_m = self.q_z.sum(axis=0)
        self.w = tmp_N_m / N
        
        for m in range(self.n_components):
            self.Mean[m] = (self.q_z.T[m].dot(data)) / tmp_N_m[m]
            x_mu = data - self.Mean[m]
            self.Sigma[m] = np.dot(self.q_z[:, m] * x_mu.T, x_mu) / tmp_N_m[m]
            
            self.Sigma[m].flat[::d + 1] += 25.0
            
        if self.diag:
            # bonus part
            pass
        else:
            pass
    
    def EM_fit(self, data, max_iter=10, tol=1e-3,
               w_init=None, m_init=None, s_init=None, trace=False):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        w_init: numpy array shape(n_components)
        Array of the each mixture component initial weight

        Mean_init: numpy array shape(n_components, n_features)
        Array of the each mixture component initial mean

        Sigma_init: numpy array shape(n_components, n_features, n_features)
        Array of the each mixture component initial covariance matrix
        
        trace: bool
        If True then return list of likelihoods
        """
        # parametrs initialization
        N, d = data.shape
        self.q_z = np.zeros((N, self.n_components))
        self.tol = tol
        
        # other initialization
        if w_init is None:
            #pass
            self.w = np.ones(shape=(self.n_components)) / self.n_components
        else:
            self.w = w_init

        if m_init is None:
            self.Mean = np.random.uniform(low=data.min(), high=data.max(), size=(self.n_components, d))
            #pass
        else:
            self.Mean = m_init

        if s_init is None:
            self.Sigma = np.tile(np.diag(np.random.uniform(low=25, high=np.abs(data).max() + 25, size=d)), [self.n_components, 1, 1])
            #pass
        else:
            self.Sigma = s_init
        log_likelihood_list = []
        ll_prev = -np.inf
        
        # algo
        #start_time = timeit.default_timer()
        
        for i in range(max_iter):
            # Perform E-step 
            #pass
            self._E_step(data)
            # Compute loglikelihood
            #pass
            log_likelihood_list.append(self.compute_log_likelihood(data))
            
            # Perform M-step
            #pass
            self._M_step(data)

            if np.abs(log_likelihood_list[-1] - ll_prev) < self.tol:
                break
            ll_prev = log_likelihood_list[-1]
        #print('one iteration:', timeit.default_timer() - start_time)
            
        # Perform E-step
        self._E_step(data)
        # Compute loglikelihood
        log_likelihood_list.append(self.compute_log_likelihood(data))
        
        if trace:
            return self.w, self.Mean, self.Sigma, log_likelihood_list
        else:
            return self.w, self.Mean, self.Sigma
    
    def EM_with_different_initials(self, data, n_starts, max_iter=10, tol=1e-3):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        n_starts: int
        The number of algorithm running with different initials

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        Returns:
        --------
        Best values for w, Mean, Sigma parameters
        """
        best_w, best_Mean, best_Sigma, max_log_likelihood = None, None, None, -np.inf
        for i in range(n_starts):
            curr_w, curr_Mean, curr_Sigma, curr_log_likelihood_list = self.EM_fit(data=data, max_iter=max_iter, tol=tol,
               w_init=None, m_init=None, s_init=None, trace=True)
            if curr_log_likelihood_list[-1] > max_log_likelihood:
                best_w, best_Mean, best_Sigma, max_log_likelihood = curr_w, curr_Mean, curr_Sigma, curr_log_likelihood_list[-1]
        
        self.w = best_w
        self.Mean = best_Mean
        self.Sigma = best_Sigma
        
        return self.w, self.Mean, self.Sigma
    
    def compute_log_likelihood(self, data):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.
        """
        tmp_N_i_j = np.empty((self.n_components, data.shape[0]))
        for j in range(self.n_components):
            res = multivariate_normal.pdf(data, self.Mean[j], self.Sigma[j])
            tmp_N_i_j[j] = res
        mixture = (self.w[np.newaxis, :] * tmp_N_i_j.T).sum(axis=1) + eps
        log_likelihood = np.log(mixture).sum()
        return log_likelihood

    def predict(self, data):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.
        """
        tmp_N_i_j = np.empty((self.n_components, data.shape[0]))
        for j in range(self.n_components):
            res = multivariate_normal.pdf(data, self.Mean[j], self.Sigma[j])
            tmp_N_i_j[j] = res
        mixture = (self.w[np.newaxis, :] * tmp_N_i_j.T)
        res = np.argmax(mixture, axis=1)
        return res
