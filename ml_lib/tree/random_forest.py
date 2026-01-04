import numpy as np
from ml_lib.tree.decision_tree  import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(
            self, 
            n_estimators=10,
            max_depth=10,
            min_samples_split=2,
            n_features=None,
            ):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X,y):
        self.trees = []

        for _ in range (self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                n_features = self.n_features
            
            )

            X_sample, y_sample = self._bootstrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_sample(self,X,y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size= n_samples, replace=True)
        return X[indices], y[indices]
    
    def predict(self, X):
        if len(self.trees) == 0:
            raise ValueError("RandomForest has no trees. Did you call fit()?")

        all_tree_preds = []

        for tree in self.trees:
            preds = tree.predict(X)
            all_tree_preds.append(preds)
        all_tree_preds = np.array(all_tree_preds)  # (n_trees, n_samples)

        return np.array([
        np.bincount(all_tree_preds[:, i]).argmax()
        for i in range(X.shape[0])
        ])
