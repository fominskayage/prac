import numpy as np


class MulticlassStrategy:   
    def __init__(self, classifier, mode, **kwargs):
        """
        Инициализация мультиклассового классификатора
        
        classifier - базовый бинарный классификатор
        
        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'
        
        **kwargs - параметры классификатор
        """
        self.classifier = classifier
        self.mode = mode
        self.kwargs = kwargs
        self.classifiers = None
        
    def fit(self, X, y):
        """
        Обучение классификатора
        """
        self.classes = np.unique(y)
        if self.mode == 'one_vs_all':
            self.classifiers = []
            for cl in self.classes:
                classifier = self.classifier(**self.kwargs)
                y_cl = y.copy()
                y_cl[y != cl] = -1
                y_cl[y_cl != -1] = 1
                classifier.fit(X, y_cl)
                self.classifiers.append(classifier)
        else:
            self.classifiers = []
            for s in range(self.classes.shape[0]):
                for j in range(s + 1, self.classes.shape[0]):
                    mask = np.logical_or(y == self.classes[s], y == self.classes[j])
                    y_cl = y[mask].copy()
                    y_cl[y_cl == classes[s]] = 1
                    y_cl[y_cl == classes[j]] = -1
                    classifier = self.classifier(**self.kwargs)
                    classifier.fit(X[mask], y_cl)
                    self.classifiers.append(classifier)
        
    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        if self.classifiers is None:
            raise Exception('Not trained yet')

        if self.mode == 'one_vs_all':
            res = []
            for cl in self.classifiers:
                res.append(cl.predict_proba(X)[:, 1].transpose())
            res = np.array(res).transpose()
            return res.max(axis=1)

        else:
            res = np.zeros(X.shape[0], self.classes.shape[0])
            for s in range(self.classes.shape[0]):
                for j in range(s + 1: self.classes.shape[0]):
                    curr_pred = self.classifiers[s + j].predict(X)
                    res[:, s] += curr_pred[curr_pred == 1].sum(1)
                    res[:, s] -= curr_pred[curr_pred == -1].sum(1)
            return res.max(axis=1)
