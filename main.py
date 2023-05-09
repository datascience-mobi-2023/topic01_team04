# Pakete etc. 
# Pandas, Numpy und Matplotlib davor über Miniforge Prompt herunterladen sonst geht hier gar nix
# Panda is pandas dataframe
# numpy is number array (general python tool)

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

from functions.PCA import pca
from functions.PCA import centered

# Import der Dateien 
# ANMERKUNG: Funktioniert auch nur, wenn die Dateien in der SELBEN Directory wie das Skript sind -> Im Git-Ordner
# Problem mit Git da CSV-Dateien zu groß sind um sie zu committen, sonst kann man gar nix mehr pushen 
# Dateien in GitHub Desktop ab in .gitignore, dann verpisst sich das Problem

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
pixel = traindata.drop('label', axis=1).to_numpy()
label = traindata['label'].to_numpy()

    

img = pixel.reshape((-1, 28, 28))
num_img = pixel.shape[0]

#function that returns a centered matrix as preparation for PCA
#def centering(Matrix): 
#    Matrix = Matrix.flatten()
#    Matrix_c = Matrix - Matrix.mean()
#    return Matrix_c 

# subtracting mean of each pixel while keeping the dimensions of the images to center the images in preparation for PCA
test = centered(img)
print(test[0].shape)
pca(test[0], 1)

#defining variable and creating covariance matrix while reshaping/flattening the images to obtain a 2D array, rowvar = False to compute cov matrix over rows
covariance_matrix = np.cov(centered_img.reshape(num_img, -1), rowvar=False)

#%%
# Printing for testing
print(covariance_matrix)
print(covariance_matrix.shape)

# Computing eigenvalues and eigenvectors of covariance matrix & printing for testing purposes
eigen_val , eigen_vec = np.linalg.eigh(covariance_matrix)
#%%
print(eigen_val)
print(eigen_vec)

# Sorting Eigenvalues and Eigenvectors
# get indices of sorted values, we need to add -1 to indicate that we want to sort in descending order
sorted_index = np.argsort(eigen_val)[::-1]
 
sorted_eigenvalue = eigen_val[sorted_index]
sorted_eigenvectors = eigen_vec[:,sorted_index]

# Selecting a certain amount of PCs (OPTIMIZATION NEEDED!)
 
principal_component_number = 4
eigenvectors_pca = sorted_eigenvectors[:,0:principal_component_number]

# transforming data:



"""Wer sich die Bilder mal anschauen will:

anz_img = len(img)
rows = int(np.sqrt(anz_img))
cols = int(np.ceil(anz_img / rows))
rows = np.minimum(rows, 20)
cols = np.minimum(cols, 20)
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < anz_img:
        ax.imshow(img[i], cmap='gray')
        ax.set_axis_off()
    else:
        ax.set_axis_off()

plt.show()"""

