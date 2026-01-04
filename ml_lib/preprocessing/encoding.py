import numpy as np
import pandas as pd

class LabelEncoder:
    def __init__(self):
        self.classes_= None
        self.class_to_index_=None

    def fit(self,y):
        self.classes_ = np.unique(y)
        self.class_to_index_ ={

            cls: idx for idx, cls in enumerate(self.classes_)
        }
        return self
    
    def transform(self,y):
        y=np.asarray(y)
        return np.array ([self.class_to_index_[val] for val in y])
    
    def fit_transform(self,y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self,y):
        y=np.asarray(y)
        return np.array([self.classes_[idx] for idx in y])
    
class OneHotEncoder:
    def __init__(self, sparse=False):
        self.sparse = sparse
        self.categories_ = None
        self.category_to_index_ = None
        self.feature_sizes_ = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_features = X.shape[1]
        self.categories_ = []
        self.category_to_index_ = []
        self.feature_sizes_ = []

        for i in range(n_features):
            cats = np.unique(X[:, i])
            self.categories_.append(cats)
            self.category_to_index_.append(
                {cat: idx for idx, cat in enumerate(cats)}
            )
            self.feature_sizes_.append(len(cats))

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples, n_features = X.shape
        total_dims = sum(self.feature_sizes_)
        onehot = np.zeros((n_samples, total_dims))

        col_start = 0
        for i in range(n_features):
            mapping = self.category_to_index_[i]
            for row in range(n_samples):
                cat = X[row, i]
                idx = mapping[cat]
                onehot[row, col_start + idx] = 1
            col_start += self.feature_sizes_[i]

        return onehot

    def fit_transform(self, X):
        return self.fit(X).transform(X)



