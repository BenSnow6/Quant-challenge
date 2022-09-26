import numpy as np

def newfitBeta(A, X_train_reshape, Y_train):
    predictors = X_train_reshape @ A
    j = Y_train.T
    targets = j.stack()
    a = predictors.T @ predictors
    b = np.linalg.inv(a)
    c = predictors.T @ targets
    beta = b @ c
    return beta

def newmetric_train(A, beta, X_train_reshape, Ytrue):
    a = A @ beta
    b = X_train_reshape @ a
    c = b.unstack()
    Y_pred = c.T
    
    Y_pred = Y_pred.div(np.sqrt((Y_pred**2).sum()), 1)
    meanOverlap = (Ytrue*Y_pred).sum().mean()
    return 1/meanOverlap