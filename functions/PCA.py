import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import main 

def centering():
    centered_img = img - img.mean(axis=(1,2), keepdims=True)
    return centered_img

def pca():
    covariance_matrix = np.cov(centered_img.reshape(num_img, -1), rowvar=False)
    eigen_val , eigen_vec = np.linalg.eigh(covariance_matrix)
    sorted_index = np.argsort(eigen_val)[::-1]
    sorted_eigenvalue = eigen_val[sorted_index]
    sorted_eigenvectors = eigen_vec[:,sorted_index]
    principal_component_number = 4
    eigenvectors_pca = sorted_eigenvectors[:,0:principal_component_number]
    # transform
