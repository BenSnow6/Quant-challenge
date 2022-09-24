import numpy as np
import pandas as pd

def oldrandomA(D=250, F=10):  
    """ Generate a random orthonormal Stiefel manifold of dimensions D x F

    Args:
        D (int, optional): Number of days of time depth used to calculate the factors. Defaults to 250.
        F (int, optional): Number of factors used in the linear factor model. Defaults to 10.

    Returns:
        np.array: D x F matrix of orthonormal D vectors of length F
    """
    M = np.random.randn(D,F)
    randomStiefel = np.linalg.qr(M)[0] # Apply Gram-Schmidt algorithm to the columns of M
    
    return (randomStiefel)

def oldcheckOrthonormality(A): 
    """ Check the orthonormality of a matrix
    
    Args:
        A (np.array): D x F Stiefel manifold of orthonormal D vectors of length F
    
    Returns:
        bool: True if A is orthonormal, False otherwise
    """
    bool = True
    D, F = A.shape   
    Error = pd.DataFrame(A.T @ A - np.eye(F)).abs()
    
    if any(Error.unstack() > 1e-6):
        bool = False
     
    return bool

def oldfitBeta(A, X_train_reshape, Y_train):
    """_fitBeta: Fit the linear factor model to the training data
    
    Args: 
        A (np.array): D x F Stiefel manifold of orthonormal D vectors of length F
        X_train_reshape (np.array): N x D x F array of N training samples of D days of F factors
        Y_train (np.array): N x D array of N training samples of D days of stock returns
        
    Returns:
        beta (np.array): (F,0) vector of model parameters
    """

    predictors = X_train_reshape @ A # the dataframe of the 10 factors created from A with the (date, stock) in index F_{t,l} = R_{t+1-k} @ A_{k,l}
    targets = Y_train.T.stack() ## the dataframe of stock returns that are to be predicted with the 10 factors
    beta = np.linalg.inv(predictors.T @ predictors) @ predictors.T @ targets
    
    return beta.to_numpy()

def oldmetric_train(A, beta, X_train_reshape, Y_train): 
    """Calculate the training error of the linear factor model

    Args:
        A (nd.array): D x F Stiefel manifold of orthonormal D vectors of length F
        beta (nd.array): Model parameters
        X_train_reshape (_type_): Training data
        Y_train (_type_): Training targets

    Returns:
        _type_: _description_
    """
    if not oldcheckOrthonormality(A):
        return -1.0    
    
    Ypred = (X_train_reshape @ A @ beta).unstack().T         # Predicted values, unstacked and transposed
    Ytrue = Y_train
    
    Ytrue = Ytrue.div(np.sqrt((Ytrue**2).sum()), 1)    ## Actual returns normalized by the sum of the squares of the returns
    Ypred = Ypred.div(np.sqrt((Ypred**2).sum()), 1)

    meanOverlap = (Ytrue * Ypred).sum().mean()        ## Mean of the dot product of the normalized returns

    return  meanOverlap


def oldcalculateGradient(A, beta, h=0.000001):
    """ Calculate the gradient of the training error of the linear factor model with respect to the orthonormal matrix A

    Args:
        A (np.array): D x F Stiefel manifold of orthonormal D vectors of length F
        beta (nd.array): Model parameters
        h (float, optional): Gradient step. Defaults to 0.000001.

    Returns:
        np.array: A.shape[0] X A.shape[0] matrix of the gradient of the training error with respect to A. A.shape[0] = D if following the challenge rules.
    """
    G = np.zeros((A.shape[0], A.shape[1])) # gradient matrix initialisation
    C = A # copy of A to allow for changes to A without affecting the original matrix, A
    for i in range(A.shape[0]): 
        for j in range(A.shape[1]): ## Loop over all elements of A
            C[i, j] += h # increment a single element of A by h
            C_beta = oldfitBeta(C) # fit the model to the new A
            G[i, j] = (oldmetric_train(C, C_beta) - oldmetric_train(A, beta))/h # calculate the gradient of the training error with respect to the element of A
            C[i, j] -= h # reset the element of A to its original value
    return G # return the gradient matrix

def oldSkewSymmetric(G, A):
    """ Calculate a skew-symmetric matrix, X, from the gradient G and the matrix A

    Args:
        G (np.array): D X F gradient matrix
        A (np.array): D x F Stiefel manifold of orthonormal D vectors of length F

    Returns:
        X (np.array): D x D dimensional Skew-symmetric matrix
    """
    return G@A.T - A@G.T

def oldCalculateQ(X, alpha):
    """ Calculate the Cayley transform of a matrix X, with step size alpha.

    Args:
        X (np.array): D x D Skew-symmetric matrix
        alpha (float): Step size

    Returns:
        Q (np.array): D X D matrix used to update the Stiefel manifold, A
    """
    Q = np.linalg.inv((np.eye(X.shape[0]) + alpha/2*X)) @ (np.eye(X.shape[0]) - alpha/2*X)
    return Q