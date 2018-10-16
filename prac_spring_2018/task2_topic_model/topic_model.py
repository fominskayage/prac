import numpy as np
import scipy.sparse as sparse
import timeit

eps = np.finfo(float).eps

class TopicModel:
    def __init__(self,
                 num_topics,
                 max_iter=30,
                 batch_size=100,
                 regularizers=tuple(),
                 modalities_coefs=(1.0, )):
        """
        Parameters:
        ---------------
        num_topics : int
            The number of topics in the algorithm
        
        max_iter: int
            Maximum number of EM iterations

        batch_size : int
            The number of objects in one batch
        
        regularizers : tuple of BaseRegularizer subclasses
            The tuple of model regularizers
                
        modalities_coefs : tuple of float
            The tuple of modalities coefs. Each coef corresponds to an element of list of data
        """
        self.num_topics = num_topics
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.regularizers = list(regularizers)
        self.modalities_coefs = modalities_coefs
    
    def _EM_step_for_batch(self, data_batch, Theta_part, batch_start_border, batch_end_border):
        """
        Iteration of the algorithm for one batch.
        It should include implementation of the E-step and M-step for the Theta matrix.
        
        Parametrs:
        ----------
        data: sparse array shape (n_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
        
        Theta_part : numpy array (n_topics, batch_size)
            Part of Theta matrix
        
        Returns:
        --------
        Theta_part : numpy array (n_topics, batch_size)
            Part of Theta matrix (after M-step)
        
        n_wt : numpy array (n_words, n_topics)
            n_wt estimates
        """
        num_documents, num_words = data_batch.shape
                
        # your code is here
        # count n_wt for batch
           
        norm_phi_theta = self.Phi[:, :, np.newaxis] * Theta_part[np.newaxis, :, :]
        
        
        norm_phi_theta[norm_phi_theta < 0] = 0
        
        sums = norm_phi_theta.sum(axis=1)[:, np.newaxis, :]
        sums[sums == 0] = 1
        norm_phi_theta /= sums
        
        
        n_wtd = data_batch.toarray().T[:, np.newaxis, :] * norm_phi_theta
        
        n_wt = n_wtd.sum(axis=2)
        n_td = n_wtd.sum(axis=0)
        self._ntd[:, batch_start_border:batch_end_border] += n_td
        # set Theta_part
        

        dR_all_dTheta = np.zeros_like(Theta_part, dtype=float)
        for regularizer in self.regularizers:
            _, dTheta = regularizer.grad(self.Phi, Theta_part, self._nwt, n_td, batch_start_border, batch_end_border)
            dR_all_dTheta += dTheta
        Theta_part = n_td + Theta_part * dR_all_dTheta
        Theta_part[Theta_part < 0] = 0
        Theta_sums = Theta_part.sum(axis=0)
        Theta_sums[Theta_sums == 0] = 1
        Theta_part /= Theta_sums
        

        return Theta_part, n_wt
            
    def _EM_iteration(self, data):
        """
        E-step of the algorithm. It should include 
        implementation of the E-step and M-step for the Theta matrix.
        
        Don't store ndwt in the memory simultaneously!
        
        Parametrs:
        ---------------
        data: sparse array shape (n_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
        """ 
        # your code
        # set self._nwt shape of (num_words, num_topics)
        self._nwt = np.zeros((data.shape[1], self.num_topics))
        # set self._ntd shape of (num_topics, num_documents)
        self._ntd = np.zeros((self.num_topics, data.shape[0]))#####################????????
        
        num_documents, num_words = data.shape
        num_batches = int(np.ceil(num_documents / self.batch_size))
                
        for batch_number in range(num_batches):
            #start_time = timeit.default_timer()
            
            batch_start_border = batch_number * self.batch_size
            batch_end_border = (1 + batch_number) * self.batch_size
            
            Theta_part = self.Theta[:, batch_start_border:batch_end_border]#None # your code is here
            
            Theta_part, n_wt_parts = self._EM_step_for_batch(data[batch_start_border:batch_end_border],
                                                            Theta_part, batch_start_border, batch_end_border)            
            # your code
            # Theta estimates
            self.Theta[:, batch_start_border:batch_end_border] = Theta_part
            # n_wt accumulation
            self._nwt += n_wt_parts
            #print(batch_number, 'batch, time:', timeit.default_timer() - start_time)
        
        # your code
        # Phi estimates
        dR_all_dPhi = np.zeros(self.Phi.shape)
        
        for regularizer in self.regularizers:
            dPhi, _ = regularizer.grad(self.Phi, self.Theta, self._nwt, self._ntd)
            dR_all_dPhi += dPhi
        
        self.Phi = self._nwt + self.Phi * dR_all_dPhi
        self.Phi[self.Phi < 0] = 0
        Phi_sums = self.Phi.sum(axis=0)
        Phi_sums[Phi_sums == 0] = 1
        self.Phi /= Phi_sums

        #print(self.get_top_words(10))
        
    def EM_fit(self, data, phi_init=None, theta_init=None,
               trace=False, vocab=None, document_names=None):
        """
        Parameters:
        -----------
        data: sparse array shape (n_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
        
        phi_init : numpy array (n_words, n_topics)
            Init values for phi matrix
            
        theta_init : numpy array (n_topics, n_documents)
            Init values for theta matrix
        
        trace: bool
            If True then return list of likelihoods
            
        vocab: list of words or list of list of words in modalities case
            vocab[i] - word that corresponds to an i column of data 
        
        document_names : list of str
            document_names[i] - name of the i-th document
        """
        num_documents, num_words = data.shape
        
        self.vocab = vocab
        self.document_names = document_names

        if phi_init:
            self.Phi = phi_init
        else:
            # use normalized random uniform dictribution
            self.Phi = np.random.uniform(size=(num_words, self.num_topics))
            self.Phi /= self.Phi.sum(axis=0)
            # in bonus task for modalities Phi must be a list of numpy arrays
        
        if theta_init:
            self.Theta = theta_init
        else:
            # use the same number for all Theta values, 1 / num_topics
            self.Theta = (1 / self.num_topics) * np.ones((self.num_topics, num_documents), dtype=float)
            
        log_likelihood_list = []
        
        for i in range(self.max_iter):
            #start_time = timeit.default_timer()
            # your code is here
            self._EM_iteration(data)

            if trace:
                log_likelihood_list += [self.compute_log_likelihood(data)]

            #print(i, 'time:', timeit.default_timer() - start_time)
        
        if trace:
            return self.Phi, self.Theta, log_likelihood_list
        else:
            return self.Phi, self.Theta
    
    def get_Theta_for_new_documents(self, data, num_inner_iter=10):
        """
        Parameters:
        -----------
        data: sparse array shape (n_new_documents, n_words) or
            list of sparse array shape (n_documents, n_modality_words) in modalities case
            Array of data points. Each row corresponds to a single document.
            
        num_inner_iter : int
            Number of e-step implementation
        """
        try:
            old_Theta = self.Theta
            old_ntd = self._ntd.copy()
            
            # your code
            #self.Theta = None
            self.Theta = (1 / self.num_topics) * np.ones((self.num_topics, data.shape[0]), dtype=float)
            self._ntd = np.zeros((self.num_topics, data.shape[0]))
            
            num_documents, num_words = data.shape
            num_batches = int(np.ceil(num_documents / self.batch_size))
         

            # your code
            for i in range(num_inner_iter):
                for batch_number in range(num_batches):
                #start_time = timeit.default_timer()
                
                    batch_start_border = batch_number * self.batch_size
                    batch_end_border = (1 + batch_number) * self.batch_size
                    
                    Theta_part = self.Theta[:, batch_start_border:batch_end_border]#None # your code is here
                    
                    Theta_part, _ = self._EM_step_for_batch(data[batch_start_border:batch_end_border],
                                                                    Theta_part, batch_start_border, batch_end_border)            
                    self.Theta[:, batch_start_border:batch_end_border] = Theta_part
                    
                
        finally:
            new_Theta = self.Theta
            self.Theta = old_Theta
            self._ntd = old_ntd
        
        return new_Theta


    def compute_log_likelihood(self, data):
        """
        Parametrs:
        ---------------
        data: sparse array shape (n_documents, n_words)
            Array of data points. Each row corresponds to a single document.
        """

        log_likelihood = (data.T.multiply(np.log((self.Phi + eps).dot(self.Theta + eps)))).sum()
        return log_likelihood#None
    
    def get_top_words(self, k):
        """
        Get list of k top words for each topic
        """
        # use argmax for Phi

        top_words = []
        for i in range(self.num_topics):
            #indices = np.argpartition(10 - self.Phi[:, i], k)
            #indices_sorted = np.argsort(10 - self.Phi[indices[:k], i])
            #final_inds = indices[indices_sorted]
            final_inds = np.argsort(10. - self.Phi[:, i])
            tmp = []
            for ind in final_inds[:k]:
                tmp.append(self.vocab[ind])
            top_words.append(tmp)

        return top_words
        #pass
        
    def get_top_docs(self, k):
        """
        Get list of k top documents for each topic
        """
        n_d = self._ntd.sum(axis=0)
        n_t = self._ntd.sum(axis=1)
        p_dt = (self.Theta * n_d[np.newaxis]) / n_t[:, np.newaxis]

        top_docs = []
        for i in range(self.num_topics):
            #indices = np.argpartition(10 - p_dt[:, i], k)
            #indices_sorted = np.argsort(10 - p_dt[indices[:k], i])
            #final_inds = indices[indices_sorted]
            final_inds = np.argsort(10. - p_dt[i])
            tmp = []
            for ind in final_inds[:k]:
                tmp.append(self.document_names[ind])
            top_docs.append(tmp)

        return top_docs
        # use argmax for p_dt
        #pass


class BaseRegularizer:
    def __init__(self, tau=1.):
        """
        Parameters:
        ----------
        tau : float
            Regularization coef
        """
        self.tau = tau
        
    def grad(self, Phi, Theta, nwt, ntd):
        """
        Gradients for Phi and for Theta
        """
        raise NotImplementedError('must be implemented in subclass')

class KLWordPairsRegularizer(BaseRegularizer):
    def __init__(self, tau, word_pairs):
        """
        Parameters:
        ----------
        tau : float
            Regularization coef
            
        word_pairs : dict (str, list_of_str) or (int, list_of_ints)
            Dict of words and their translations. Implementation depends on you. 
        """
        super().__init__(tau)
        self.word_pairs = word_pairs
    
    def grad(self, Phi, Theta, nwt, ntd, batch_start_border=None, batch_end_border=None):
        """
        Gradients for Phi and for Theta
        """

        dR_dPhi = np.zeros(Phi.shape)
        tr = np.zeros(Phi.shape)
        n_ut_normed = nwt / (nwt.sum(axis=1) + eps * (nwt.sum(axis=1) == 0))[:, np.newaxis]
        for w, u_s in self.word_pairs.items():
            tr[w] += n_ut_normed[u_s].sum(axis=0)
        
        dR_dPhi = tr
        dR_dPhi[Phi != 0] /= Phi[Phi != 0]

        dR_dTheta = np.zeros(Theta.shape)
        
        return dR_dPhi * self.tau, dR_dTheta

class KLDocumentPairsRegularizer(BaseRegularizer):
    def __init__(self, tau, document_pairs):
        """
        Parameters:
        ----------
        tau : float
            Regularization coef
            
        document_pairs : dict (int, list of ints)
            Dict of documents and their parallel variant
        """
        super().__init__(tau)
        self.document_pairs = document_pairs
        self.docs = np.array(list(self.document_pairs.keys()))
        self.parallel = np.array(list(self.document_pairs.values())).ravel()
    
    def grad(self, Phi, Theta, nwt, ntd, batch_start_border=None, batch_end_border=None):
        """
        Gradients for Phi and for Theta
        """    
        dR_dPhi = np.zeros(Phi.shape)
            
        if not (batch_start_border is None):
        
            tr = np.zeros_like(Theta, dtype=float)
            n_ts_normed = ntd / (ntd.sum(axis=0) + eps * (ntd.sum(axis=0) == 0))[np.newaxis]
            
            mask_docs = np.bitwise_and(self.docs >= batch_start_border, self.docs < batch_end_border)
            mask_parallel = np.bitwise_and(self.parallel >= batch_start_border, self.parallel < batch_end_border)
            mask = np.bitwise_and(mask_docs, mask_parallel)

            tr[:, self.docs[mask]] = n_ts_normed[:, self.parallel[mask]]
            dR_dTheta = ((Theta != 0) / (Theta + eps * (Theta == 0))) * tr

        else:

            tr = np.zeros_like(Theta, dtype=float)
            n_ts_normed = ntd / (ntd.sum(axis=0) + eps * (ntd.sum(axis=0) == 0))[np.newaxis]
            
            tr[:, self.docs] = n_ts_normed[:, self.parallel]
            dR_dTheta = ((Theta != 0) / (Theta + eps * (Theta == 0))) * tr
        
        return dR_dPhi, dR_dTheta * self.tau
