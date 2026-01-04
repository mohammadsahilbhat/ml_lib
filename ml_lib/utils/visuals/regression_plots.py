import numpy as np 
import matplotlib.pyplot as plt

def plot_regression_line(X,y,model,Xlabel="X",ylabel="y",title="REGRESSION LINE"):

    y_pred = model.predict(X)

    plt.scatter(X,y ,label = "Actual")
    plt.plot(X,y_pred,color='red',label="Predicted")
    plt.title(title)
    plt.xlabel(Xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def regression_dashboard(X,y,model,Xlabel="X",ylabel="y",title="REGRESSION DASHBOARD"):

    y_pred = model.predict(X)

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.scatter(X,y,label="Actual")
    plt.plot(X,y_pred,color='red',label="Predicted")
    plt.title("Regression Line")
    plt.xlabel(Xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.scatter(y,y_pred)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)

    plt.subplot(1,3,3)
    residuals = y - y_pred
    plt.scatter(y_pred,residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def loss_cuve(losses,title="LOSS CURVE"):

    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


