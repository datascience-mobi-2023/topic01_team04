# Pakete etc. 
# Pandas, Numpy und Matplotlib davor über Miniforge Prompt herunterladen sonst geht hier gar nix
# Panda is pandas dataframe
# numpy is number array (general python tool)

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt


# Import der Dateien 
# ANMERKUNG: Funktioniert auch nur, wenn die Dateien in der SELBEN Directory wie das Skript sind -> Im Git-Ordner
# Problem mit Git da CSV-Dateien zu groß sind um sie zu committen, sonst kann man gar nix mehr pushen 
# Dateien in GitHub Desktop ab in .gitignore, dann verpisst sich das Problem

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')

# Extraction of pixels and labels from csv via .to_numpy, reshaping to create a 3D array containing separate 28 x 28 pictures

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
centered_img = img - img.mean(axis=(1,2), keepdims=True)

#defining variable and creating covariance matrix while reshaping/flattening the images to obtain a 2D array, rowvar = False to compute cov matrix over rows
covariance_matrix = np.cov(centered_img.reshape(num_img, -1), rowvar=False)

print(covariance_matrix)
print(covariance_matrix.shape)


#print(np.cov(img[0].flatten(), rowvar=True))
#print(type(img[0].flatten()))


#before continuing note that the np.cov() function has to be given a 1-Dimensional array, otherwise each row/column is considered a variable (and not each pixel as intended)

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


#commit with text


