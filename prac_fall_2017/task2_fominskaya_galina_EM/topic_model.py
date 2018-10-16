import numpy as np
import scipy.sparse as sparse

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
        zeros = np.zeros((num_words, self.num_topics, num_documents), dtype=float)
        norm_phi_theta = np.maximum((self.Phi[:, :, np.newaxis] * Theta_part[np.newaxis, :, :]), zeros)
        norm_phi_theta /= norm_phi_theta.sum(axis=1)[:, np.newaxis, :]
        #n_wtd = data_batch.T[:, np.newaxis, :] * norm_phi_theta
        n_wt = (data_batch.T[:, np.newaxis, :] * norm_phi_theta).sum(axis=2)
        n_td = (data_batch.T[:, np.newaxis, :] * norm_phi_theta).sum(axis=0)
        self._ntd[:, batch_start_border:batch_end_border] += n_td
        # set Theta_part
        dR_all_dTheta = np.zeros_like(Theta_part, dtype=float)
        for regularizer in self.regularizers:
            _, dTheta = regularizer.grad(self.Phi, Theta_part, n_wt, n_td)
            dR_all_dTheta += dTheta
        zeros = np.zeros_like(Theta_part, dtype=float)
        Theta_part = np.maximum(n_td + Theta_part * dR_all_dTheta, zeros)
        Theta_part /= Theta_part.sum(axis=0)

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
        
        # your code
        # Phi estimates
        dR_all_dPhi = np.zeros_like(self.Phi, dtype=float)
        
        for regularizer in self.regularizers:
            dPhi, _ = regularizer.grad(self.Phi, self.Theta, self._nwt, self._ntd)
            dR_all_dPhi += dPhi
        
        zeros = np.zeros_like(self.Phi, dtype=float)
        self.Phi = np.maximum(self._nwt + self.Phi * dR_all_dPhi, zeros)
        self.Phi /= self.Phi.sum(axis=0)
        
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
            # your code is here
            self._EM_iteration(data)

            if trace:
                log_likelihood_list += [self.compute_log_likelihood(data)]
        
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
            
            # your code
            self.Theta = None
            
            # your code
            for i in range(num_inner_iter):
                pass
                
                
        finally:
            new_Theta = self.Theta
            self.Theta = old_Theta
        
        return new_Theta
    
    
    def compute_log_likelihood(self, data):
        """
        Parametrs:
        ---------------
        data: sparse array shape (n_documents, n_words)
            Array of data points. Each row corresponds to a single document.
        """
        log_likelihood = (data.T * np.log(self.Phi.dot(self.Theta))).sum()
        return log_likelihood#None
    
    def get_top_words(self, k):
        """
        Get list of k top words for each topic
        """
        # use argmax for Phi
        pass
        
    def get_top_docs(self, k):
        """
        Get list of k top documents for each topic
        """
        n_d = self._ntd.sum(axis=0)
        n_t = self._ntd.sum(axis=1)
        p_dt = None
        # use argmax for p_dt
        pass


class BaseRegularizer:
    def __init__(self, tau=1.0):
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
        super().__init__(self, tau)
        self.word_pairs = word_pairs
        self.pis = 
    
    def grad(self, Phi, Theta, nwt, ntd):
        """
        Gradients for Phi and for Theta
        """

        dR_dPhi = (pis * n_ut).sum()#None
        pis = np.maximum(pis * n_ut, np.zeros_like(pis * n_ut))
        pis /= pis.sum()
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
        super().__init__(self, tau)
        self.document_pairs = document_pairs
    
    def grad(self, Phi, Theta, nwt, ntd):
        """
        Gradients for Phi and for Theta
        """
        
        dR_dPhi = np.zeros(Phi.shape)
        dR_dTheta = None
        
        return dR_dPhi, dR_dTheta * self.tau
