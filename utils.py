import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special
import json
import os

# Functions to reshape the data
def vcol(v):
    return(v.reshape((v.size, 1)))

def vrow(v):
    return (v.reshape((1, v.size)))

# Function to load the training dataset
def loadDataset(filename):
    dir_path = os.path.dirname(__file__)
    file_path = dir_path + filename

    features = []
    classes  = []

    with open(file_path, 'r') as file:
        for line in file:
            l_features = vcol(np.array([float(i) for i in line.split(',')[:6]]))
            l_class = int(line.split(',')[-1])

            features.append(l_features)
            classes.append(l_class)

    return  np.hstack(features), np.array(classes, dtype=np.int32)

# Function to perform Principal Component Analysis on a dataset
def PCA(D, m=2, project=True):

    # Calculate mean over columns of dataset
    mu = D.mean(1)

    # Center the data of the dataset
    # Subtract the mean vector reshaped to the dataset
    mu = vcol(mu)
    DC = D - mu

    # Calculate covariance matrix
    C = (DC @ DC.T) / float(D.shape[1])

    # Calculate eigenvector and eigenvalues
    # Using np.linalg.eigh since C is symmetric
    _, U = np.linalg.eigh(C)

    # Retrieve the m leading eigenvectors
    P = U[:, ::-1][:, 0:m]

    return np.dot(P.T, D) if project else P

# Function to perform Linear Discriminant Analysis on a dataset
def LDA(D, L, m=2, project=True):

    # We need to calculate the Sb matrix 
    mu = vcol(D.mean(1))

    distintClasses = set(L)
    
    Sb = 0
    Sw = 0

    for i in distintClasses:
        D_f = D[:, L==i]
        mu_f = vcol(D_f.mean(1))
        Sb += (D_f.shape[1]) * np.dot((mu_f - mu), (mu_f - mu).T)
        DC = D_f - mu_f
        Sw += np.dot(DC, DC.T)
    Sb /= D.shape[1]
    Sw /= D.shape[1]

    # Using Genralized eigenvalue problem
    return GEP(D, Sb, Sw, m, project)

def GEP(D, Sb, Sw, m, project=True):

    # Sb*w = lambda*Sw*w
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    return np.dot(W.T, D) if project else W

# Function to split the dataset into training and validation
def split_db_2to1(D, L, seed=0):

    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)

# Function to calculate threshold
def calculateThresh(DTR, LTR, DVAL, LVAL):

    # Set a threshold for the classification (over 1st dimension)
    threshold = (DTR[0, LTR==0].mean() + DTR[0, LTR==1].mean()) / 2.0

    # Try finding optimal threshold
    bestErrorRate = 100
    bestThreshold = 0.0
    for i in np.arange(-1, 1, 0.001):

        threshold = i

        # Compare to the threshold
        PVAL = np.zeros(shape = LVAL.shape, dtype=np.int32)
        PVAL[DVAL[0] >= threshold] = 1
        PVAL[DVAL[0]  < threshold] = 0

        cntErr = 0
        for i in range(len(LVAL)):
            if LVAL[i] != PVAL[i]:
                cntErr += 1

        errorRate = cntErr * 100 / len(LVAL)
        if errorRate < bestErrorRate:
            bestErrorRate = errorRate
            bestThreshold = threshold

    print(f'Error rate: {bestErrorRate}')
    print(f'Threshold used: {bestThreshold}')

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]
    
    # Inverse and log-determinant of covariance matrix
    C_inv = np.linalg.inv(C)
    _, log_det_C = np.linalg.slogdet(C)
    
    # Adjust for the constant terms (2π)^(-M/2) and other constants
    const_term = -0.5 * M * np.log(2 * np.pi) - 0.5 * log_det_C
    
    # Reshape mu for broadcasting
    mu = mu.reshape(-1, 1)
    
    # Compute the quadratic term for all samples
    # (X - mu) is of shape (M, N), C_inv is (M, M), resulting shape will be (M, N)
    diffs = X - mu
    quadratic_term = np.sum(diffs * (C_inv @ diffs), axis=0)
    
    # Compute log densities
    log_densities = const_term - 0.5 * quadratic_term
    
    return log_densities

def save_scores(scores, filename):
    dir_path = os.path.dirname(__file__)
    scoresJson = [score for score in scores]
    with open(dir_path + filename, 'w') as f:
        json.dump(scoresJson, f)

def load_scores(filename):
    with open(filename, 'r') as f:
        scores = json.load(f)
    return [score for score in scores]