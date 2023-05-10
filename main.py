# Pakete etc. 
# Pandas, Numpy und Matplotlib davor über Miniforge Prompt herunterladen sonst geht hier gar nix
# Panda is pandas dataframe
# numpy is number array (general python tool)

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt

from functions.PCA import pca
from functions.PCA import centered
from functions.KNN import dist

print('\n')
# Import der Dateien 
# ANMERKUNG: Funktioniert auch nur, wenn die Dateien in der SELBEN Directory wie das Skript sind -> Im Git-Ordner
# Problem mit Git da CSV-Dateien zu groß sind um sie zu committen, sonst kann man gar nix mehr pushen 
# Dateien in GitHub Desktop ab in .gitignore, dann verpisst sich das Problem

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
testdata_pixel = testdata.drop('label', axis=1).to_numpy()
pixel = traindata.drop('label', axis=1).to_numpy()
label = traindata['label'].to_numpy()

img = pixel.reshape((-1,28,28)) #only relevant for visualisation
num_img = pixel.shape[0]

# subtracting mean of each pixel while keeping the dimensions of the images to center the images in preparation for PCA
test = centered(pixel)


i = False
inbud = input('1. test PCA\n2. test KNN: ')
if  inbud == 1:
    print('shape of the training Data: ' + str(test.shape))
    i = False
    while i == False:
        print('\n')
        
        eiovar = input('type either eigenvector number or explained proportion of variance: ')
        print('shape of the training PCA: ' + str(pca(test, float(eiovar))[0].shape))

        if input('again?: ') == 'no':
            i = True
elif inbud == 2:
    while i == False:
        print('\n')

        k = input('whats k?: ')
        print(str(dist(pca(centered(testdata_pixel), 0.9)[0], pca(test, 0.9)[0], k)))

        if input('again?: ') == 'no':
            i = True




traindata_centered = centered(pixel)
testdata_centered = centered(testdata_pixel)
traindata_pca, eigenmatrix = pca(traindata_centered,0.95)
testdata_pca = np.dot(eigenmatrix.transpose(),testdata_centered.transpose()).transpose()





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

"""
TO DO:  -reshape data into (-1,28*28)
        -modify centering function
        -modify cov matrix
        -check cov matrix
        -check if more modifications are needed
"""