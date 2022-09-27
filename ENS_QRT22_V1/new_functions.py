import numpy as np
from old_functions import *

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

def newCalculateGradient(A, beta, X_train_reshape, Y_train, h=0.000001, previousMetric=None):
    print(f"Previous metric: {previousMetric}")
    G = np.zeros((A.shape[0], A.shape[1])) # gradient matrix initialisation
    # G_old = np.zeros((A.shape[0], A.shape[1])) # gradient matrix initialisation
    C = A # copy of A to allow for changes to A without affecting the original matrix, A
    Ytrue = Y_train.div(np.sqrt((Y_train**2).sum()), 1)
    for i in range(A.shape[0]): 
        for j in range(A.shape[1]): ## Loop over all elements of A
            C[i, j] += h # increment a single element of A by h
            C_beta = newfitBeta(C, X_train_reshape, Y_train) # fit the model to the new A
            G[i, j] = (newmetric_train(C, C_beta, X_train_reshape, Ytrue) - newmetric_train(A, beta, X_train_reshape, Ytrue))/h # calculate the gradient of the training error with respect to the element of A
            # G_old[i, j] = (oldmetric_train(C, C_beta, X_train_reshape=X_train_reshape, Y_train=Y_train) - oldmetric_train(A, beta, X_train_reshape=X_train_reshape, Y_train=Y_train))/h # calculate the gradient of the training error with respect to the element of A
            # print(G[i,j]-G_old[i,j])
            C[i, j] -= h # reset the element of A to its original value
    return G # , G_old # return the gradient matrix