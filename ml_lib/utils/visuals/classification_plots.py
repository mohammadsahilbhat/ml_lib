import matplotlib.pyplot as plt
import numpy as np

def plot_classification_results(X, y_true, y_pred):
    # Correct or wrong predictions
    correct = y_true == y_pred
    
    plt.figure(figsize=(7, 6))
    
    # Plot correct predictions
    plt.scatter(
        X[correct, 0], X[correct, 1],
        c=y_true[correct], cmap="coolwarm",
        edgecolors="k", label="Correct"
    )
    
    # Plot wrong predictions
    plt.scatter(
        X[~correct, 0], X[~correct, 1],
        c=y_true[~correct], cmap="coolwarm",
        edgecolors="red", linewidth=1.8, label="Incorrect"
    )
    
    plt.title("Classification Results (True vs Predicted)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.show()
def plot_prediction_comparison(y_true, y_pred):
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.title("Actual vs Predicted Labels")
    plt.xlabel("Samples")
    plt.ylabel("Class")
    plt.legend()
    plt.grid(True)
    plt.show()