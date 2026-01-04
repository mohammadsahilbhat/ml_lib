import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X,y,model, resolution=0.02,title = 'DECISION BOUNDARY'):
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx,yy = np.meshgrid(
        np.arange(x_min,x_max,resolution),
        np.arange(y_min,y_max,resolution)

    )

    grid = np.c_[xx.ravel(),yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    
    cmap_light =ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    plt.contourf(xx,yy,Z,cmap=cmap_light,alpha=0.5)

    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,edgecolor='k',s=50)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    




























