import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn as skl

from functions.KNN import dist, quality, most_common_items, labl
from sklearn.decomposition import PCA

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
testdata_pixel = testdata.drop('label', axis=1).to_numpy()
traindata_pixel = traindata.drop('label', axis=1).to_numpy()
label_train = traindata['label'].to_numpy()
label_test = testdata['label'].to_numpy()

def centered(img):
    """Performs a Z-Transformation on each row of the images

    Args:
        img (numpy array): pictures you want to center and scale as rows

    Returns:
        numpy array: Z-transformed pictures
    """
    centered_img = (img - img.mean(axis=0))/np.std(img, axis=0) 
    return centered_img

def pca(centered_img, prop_variance): #prop_variance can be used as input for proportion of variance OR number of eigenvalues
    """performs principal component analysis with the columns as variables (pixels) and the rows as realisations (intensities)

    Args:
        centered_img (numpy array): Z-transformed image
        prop_variance (int or float): either desired proportion of variance or number of eigenvectors

    Returns:
        numpy array: transformed matrix
        numpy array: sorted and sliced eigenmatrix
    """
    covariance_matrix = np.cov(centered_img.transpose()) #covariance matrix, as each column is a variable we need rowvar=False
    print(covariance_matrix.shape)
    eigen_val , eigen_vec = np.linalg.eig(covariance_matrix) #eigh function has two outputs, so two values have to be defined
    sorted_index = np.flip(np.argsort(np.abs(eigen_val)))#gives indexes to sort array from lowes to highest and inverts this vector
    sorted_eigenvalue = eigen_val[sorted_index] #apply sorting to eigenvalues
    sorted_eigenvectors = eigen_vec[:,sorted_index] #apply sorting to eigenvectors, first coordinate (vertical axis) has to be : to select all rows
    def propvar(prop_var): 
        """Adds eigenvalues until desired proportion of variance is reached

        Args:
            prop_var (float): desired proportion of variance

        Returns:
            int: numper of principal comonents
        """
        sum = 0
        sum_eigenvalues = np.sum(sorted_eigenvalue)
        principal_component_number = 0
        for i in sorted_eigenvalue:
            if sum < prop_var*sum_eigenvalues:
                sum += i
                principal_component_number += 1
        sum /= sum_eigenvalues
        print("Our eigenvectors explain " + str(sum*100) +" percent of total variance")
        print(str(principal_component_number) + " eigenvectors are used")
        return principal_component_number
    if prop_variance <= 1:
        prop_variance = propvar(prop_variance)
    else:
        percent_prop = np.sum(sorted_eigenvalue[0:(int(prop_variance))]) / np.sum(sorted_eigenvalue) * 100
        print("Our eigenvectors explain " + str(percent_prop) + " % of total variance")
    eigenvectors_pca = sorted_eigenvectors[:,:int(prop_variance)] #slicing of first principil_component_number eigenvectors from sorted eigenvector matrix
    transformed_matrix_pca = np.dot(centered_img,eigenvectors_pca) # Transforming data with dot product of two arrays 
    return transformed_matrix_pca, eigenvectors_pca #returns eigenvectors to multiply with test data
