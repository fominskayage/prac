from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
import numpy as np
import timeit
from IPython.display import clear_output

class LFM:    
    def __init__(self, n_components, lamb=1e-2, mu=1e-2, max_iter=10, tol=1e-4, verbose=False):
        """
        Parameters:
        -----------
            n_components : float, number of components in Latent Factor Model
            
            lamb : float, l2-regularization coef for users profiles
            
            mu : float, l2-regularization coef for items profiles
            
            max_iter: int, maximum number of iterations
            
            tol: float, tolerance of the algorithm
            (if \sum_u \sum_d p_{ud}^2 + \sum_i \sum_d q_{id}^2 < tol then break)
            
            verbose: bool, if true then print additional information during the optimization
        """
        self.n_components = n_components
        self.lamb = lamb
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
    def fit(self, X, P_init=None, Q_init=None, validation_triplets=None, trace=False):
        """
        Fitting of Latent Factor Model using ALS method
        
        Parameters:
        -----------
            X : sparse matrix, users-items matrix
        """
        if P_init:
            P = P_init
        else:
            P = np.random.uniform(high=(10.5 / self.n_components) ** 0.5, size=(self.n_components, X.shape[0]))#init
            
        if Q_init:
            Q = Q_init
        else:
            Q = np.random.uniform(high=(10.5 / self.n_components) ** 0.5, size=(self.n_components, X.shape[1]))#init

        # need for faster optimization        
        XT = csr_matrix(X.T)

        norm_tracker = []
        #time_tracker = []
        #P_tracker = []
        #Q_tracker = []

        for iteration in range(self.max_iter):
            #print(iteration)
            #start_time_1 = timeit.default_timer()
            old_P = P.copy()
            old_Q = Q.copy()
            #P_tracker.append(P.copy())
            #Q_tracker.append(Q.copy())

            norm_p = 0
            norm_q = 0
            
            # fix Q, recalculate P
            
            inds = (np.array(X.sum(axis=1)) != 0).ravel()
            #start_time = timeit.default_timer()
            for u in np.arange(P.shape[1])[inds]:
                #if u % 500 == 0:
                    #clear_output()
                    #print(u)
                Q_tmp = Q[:, X[u].indices]
                ch = cho_factor(Q_tmp.dot(Q_tmp.T) + self.lamb * np.eye(Q_tmp.shape[0]))
                P[:, u] = cho_solve(ch, X[u].dot(Q.T).T).ravel()
            #print(timeit.default_timer() - start_time_1)
            # fix P, recalculate Q
            
            #start_time = timeit.default_timer()

            inds = (np.array(X.sum(axis=0)) != 0).ravel()
            for i in np.arange(Q.shape[1])[inds]:
                #if i % 500 == 0:
                    #clear_output()
                    #print(i)
                P_tmp = P[:, XT[i].indices]
                ch = cho_factor(P_tmp.dot(P_tmp.T) + self.mu * np.eye(P.shape[0]), lower=True)
                
                Q[:, i] = cho_solve(ch, XT[i].dot(P.T).T).ravel()
                
            norm_p = ((P - old_P) ** 2).sum() ** 0.5
            norm_q = ((Q - old_Q) ** 2).sum() ** 0.5

            norm_tracker.append(norm_p + norm_q)
            #time_tracker.append(timeit.default_timer() - start_time)
            if norm_p + norm_q <= self.tol:
                break
            #print(timeit.default_timer() - start_time_1)
            

        self.Q = Q
        self.P = P
        if (trace):
            return norm_tracker#, time_tracker, P_tracker, Q_tracker
    
    def predict_for_pair(self, user, item):
        """
        Get the prediction
        
        Parameters:
        -----------
            user : non-negative int, user index
            
            item : non-negative int, item index
        """
        Q = self.Q
        P = self.P

        res = P[:, user].T.dot(Q[:, item])
        return res
        #pass