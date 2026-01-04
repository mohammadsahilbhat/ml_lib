import numpy as np

def shuffle_data(X, y):
    idxs= np.random.permutation(len(X))
    return X[idxs], y[idxs]

def train_test_cv_split(X,y,test_size=0.2,cv=0.1,shuffle=True):
    
    if shuffle:
        X,y = shuffle_data(X,y)

    total_size = len(X)

    test_count = int(total_size * test_size)
    cv_count = int(total_size * cv)

    x_test = X[:test_count]
    y_test = y[:test_count]

    x_cv = X[test_count:test_count + cv_count]
    y_cv = y[test_count:test_count + cv_count]

    x_train = X[test_count + cv_count:]
    y_train = y[test_count + cv_count:] 

    return x_train, y_train, x_test, y_test, x_cv, y_cv


def train_test_split(X, y, test_size=0.2, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X, y)

    total_size = len(X)
    test_count = int(total_size * test_size)

    x_test = X[:test_count]
    y_test = y[:test_count]

    x_train = X[test_count:]
    y_train = y[test_count:]

    return x_train, y_train, x_test, y_test

def batch_generator(X,y,batch_size = 32,shuffle =True):
   
    if shuffle:
        X,y = shuffle_data(X,y)
    for i in range(0,len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        yield x_batch, y_batch
        