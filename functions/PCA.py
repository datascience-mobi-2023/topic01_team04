import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

def centered(img): #centering ist ein python eigenname, das Python Modul wird hiermit überschrieben. Das kann zu Problemen führen.
    centered_img = img - img.mean(axis=(1),keepdims=True) #mean is calculated along the horizontal axis
    return centered_img

def pca(centered_img, prop_variance): #prop_variance can be used as input for proportion of variance OR number of eigenvalues
    covariance_matrix = np.cov(centered_img, rowvar=False) #covariance matrix, as each column is a variable we need rowvar=False
    eigen_val , eigen_vec = np.linalg.eigh(covariance_matrix) #eigh function has two outputs, so two values have to be defined
    sorted_index = np.argsort(eigen_val)[::-1] #gives indexes to sort array from lowes to highest and inverts this vector
    sorted_eigenvalue = eigen_val[sorted_index] #apply sorting to eigenvalues
    sorted_eigenvectors = eigen_vec[:,sorted_index] #apply sorting to eigenvectors, first coordinate (vertical axis) has to be : to select all rows
    def propvar(prop_var): #adds eigenvalues until desired proportion of variance is reached
        sum = 0
        sum_eigenvalues = np.sum(eigen_val)
        principal_component_number = 0
        for i in sorted_eigenvalue:
            if sum <= prop_var:
                sum += (i/sum_eigenvalues)
                principal_component_number += 1
        sum *= 100
        print("Our eigenvectors explain " + str(sum) +" % of total variance")
        print(str(principal_component_number) + " eigenvectors are used")
        return principal_component_number
    if prop_variance <= 1:
        prop_variance = propvar(prop_variance)
    else:
        yeeet = np.sum(sorted_eigenvalue[0:(int(prop_variance))]) / np.sum(sorted_eigenvalue) * 100
        print("Our eigenvectors explain " + str(yeeet) + " % of total variance")
    eigenvectors_pca = sorted_eigenvectors[:,0:int(prop_variance)] #slicing of first principil_component_number eigenvectors from sorted eigenvector matrix
    transformed_matrix_pca = np.dot(eigenvectors_pca.transpose(),centered_img.transpose()).transpose() # Transforming data with dot product of two arrays 
    return transformed_matrix_pca, eigenvectors_pca #returns eigenvectors to multiply with test data

    