import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
 

def centered(img):
    """Performs a Z-Transformation on each row of the images

    Args:
        img (numpy array): pictures you want to center and scale as rows

    Returns:
        numpy array: Z-transformed pictures
    """
    centered_img = (img - img.mean(axis=0))/np.std(img, axis=0) 
    return centered_img


def pca(training_img, test_img, prop_variance): 
    """performs principal component analysis with the columns as variables (pixels) and the rows as realisations (intensities)

    Args:
        centered_img (numpy array): training images
        centered_test (numpy array): test images
        prop_variance (int or float): either desired proportion of variance or number of eigenvectors

    Returns:
        numpy array: transformed training matrix
        numpy array: transformed test matrix
    """
    centered_img = centered(training_img)
    centered_test = centered(test_img)
    covariance_matrix = np.cov(centered_img.transpose()) 
    print(covariance_matrix.shape)
    eigen_val , eigen_vec = np.linalg.eig(covariance_matrix) 
    sorted_index = np.flip(np.argsort(np.abs(eigen_val)))
    sorted_eigenvalue = eigen_val[sorted_index] 
    sorted_eigenvectors = eigen_vec[:,sorted_index] 
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
    eigenvectors_pca = sorted_eigenvectors[:,:int(prop_variance)] 
    transformed_matrix_pca = np.dot(centered_img,eigenvectors_pca) 
    transformed_test = np.dot(centered_test,eigenvectors_pca)
    return transformed_matrix_pca, transformed_test


def custum_imshow(img):
    """Take a look at the first 400 clothing images

    Args:
        img (numpy array): array of all the images you want to look at as rows
    """
    anz_img = len(img)
    rows = int(np.sqrt(anz_img))
    cols = int(np.ceil(anz_img / rows))
    rows = np.minimum(rows, 1)
    cols = np.minimum(cols, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < anz_img:
            ax.imshow(img[i], cmap='gray')
            ax.set_axis_off()
        else:
            ax.set_axis_off()
    plt.show()
    plt.show()
