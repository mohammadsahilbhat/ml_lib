import numpy as np

class Pipeline:
    def __init__(self, steps):
        """
        steps: list of (name, transformer)
        Example: [('imputer', SimpleImputer()), ('scaler', StandardScaler())]
        """
        self.steps = steps

    def fit(self, X):
        """
        Fit and transform X through each step except the last one 
        (model must be fit separately)
        """
        X_transformed = X
        
        for name, step in self.steps:
            # If step has fit_transform → use it
            if hasattr(step, 'fit_transform'):
                X_transformed = step.fit_transform(X_transformed)
            # If step has fit + transform → call both
            elif hasattr(step, 'fit') and hasattr(step, 'transform'):
                step.fit(X_transformed)
                X_transformed = step.transform(X_transformed)
            else:
                raise ValueError(f"Step '{name}' does not support transform")
        
        return X_transformed

    def transform(self, X):
        """
        Only transform X (used for test data)
        """
        X_transformed = X
        
        for name, step in self.steps:
            if hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
            else:
                raise ValueError(f"Step '{name}' does not support transform")

        return X_transformed

