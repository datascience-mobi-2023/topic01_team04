import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
#import main #No module named 'main'

def centering():
    centered_img = img - img.mean(axis=(1,2), keepdims=True) #mean is calculated along 2nd and 3rd axis, meaning the heigth and the width of an image (remember: the first dimension is which picture we are looking at)
    return centered_img

def pca(centered_img): #added input variable centered_img
    covariance_matrix = np.cov(centered_img.reshape(num_img, -1), rowvar=False) #defining variable and creating covariance matrix while reshaping/flattening the images to obtain a 2D array, rowvar = False to compute cov matrix over rows
    eigen_val , eigen_vec = np.linalg.eigh(covariance_matrix) #eigh function has two outputs, so two values have to be defined
    sorted_index = np.argsort(eigen_val)[::-1] #gives indexes to sort array from lowes to highest and inverts this vector
    sorted_eigenvalue = eigen_val[sorted_index] #apply sorting to eigenvalues
    sorted_eigenvectors = eigen_vec[:,sorted_index] #apply sorting to eigenvectors, first coordinate (vertical axis) has to be : to select all rows
    principal_component_number = 4 #yeah science
    eigenvectors_pca = sorted_eigenvectors[:,0:principal_component_number] #slicing of first principil_component_number eigenvectors from sorted eigenvector matrix
    # transform data