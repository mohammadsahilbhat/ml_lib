import numpy as np

class Node:
    """A node in the decision tree."""
    def __init__(self,feature =None,threshold =None, left =None, right=None,*, value =None):
        self.feature = feature
        self.threshold = threshold  
        self.left = left
        self.right = right
        self.value = value
class DecisionTreeClassifier:
    def __init__(self,max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    def fit(self,X,y):
        self.n_features =X.shape[1] if self.n_features is None else self.n_features
        self.root = self._grow_tree(X,y)
    def _grow_tree(self, X,y,depth=0):
        n_samples, n_features =X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth 
            or n_labels == 1
            or n_samples < self.min_samples_split

        ):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        feature_idxs = np.random.choice(n_features, self.n_features,replace=False)
       
        best_feature, best_thresh = self._best_split(X,y,feature_idxs)
        if best_feature is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs =self._split(X[:,best_feature],best_thresh)
        left = self._grow_tree(X[left_idxs],y[left_idxs],depth=depth+1)
        right = self._grow_tree(X[right_idxs],y[right_idxs],depth=depth+1)

        return Node(best_feature,best_thresh,left,right)
    
    def _best_split(self,X,y,feature_idxs):
        best_gain = -1
        split_idx, split_thresh = None , None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds =np.unique(X_column)

            for thresh in thresholds:
                gain = self. _information_gain(y,X_column, thresh)
                if gain > best_gain:
                    best_gain = gain 
                    split_idx = feature_idx
                    split_thresh = thresh   
        return split_idx, split_thresh
    def _information_gain(self,y,X_column, split_thresh):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len (right_idxs) == 0:
            return 0
        
        n= len(y)
        n_l , n_r = len(left_idxs), len(right_idxs)

        child_entropy = (
            (n_l/n)*self._entropy(y[left_idxs]) +
            (n_r/n)*self._entropy(y[right_idxs])

        )

        return parent_entropy - child_entropy
    

    def _entropy(self,y):
        counts = np.bincount(y)
        probs = counts/len(y)

        return -np.sum([p*np.log2(p) for p in probs if p>0])
    
    def _split(self, X_column, split_threshold):
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs
    
    def _most_common_label(self,y):
        return np.bincount(y).argmax()
    
    def predict (self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)




























        