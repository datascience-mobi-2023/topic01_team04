# Pakete etc. 
# Pandas, Numpy und Matplotlib davor über Miniforge Prompt herunterladen sonst geht hier gar nix
# Panda is pandas dataframe
# numpy is number array (general python tool)

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn as skl

from functions.PCA import pca
from functions.PCA import centered
from functions.KNN import dist
from functions.KNN import labl
from functions.KNN import most_common_items
from functions.KNN import quality

from sklearn.decomposition import PCA
print('\n')
# Import der Dateien 
# ANMERKUNG: Funktioniert auch nur, wenn die Dateien in der SELBEN Directory wie das Skript sind -> Im Git-Ordner
# Problem mit Git da CSV-Dateien zu groß sind um sie zu committen, sonst kann man gar nix mehr pushen 
# Dateien in GitHub Desktop ab in .gitignore, dann verpisst sich das Problem

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
testdata_pixel = testdata.drop('label', axis=1).to_numpy()
traindata_pixel = traindata.drop('label', axis=1).to_numpy()
label_train = traindata['label'].to_numpy()
label_test = testdata['label'].to_numpy()


img = traindata_pixel.reshape((-1,28,28)) #only relevant for visualisation
num_img = traindata_pixel.shape[0]

# subtracting mean of each pixel while keeping the dimensions of the images to center the images in preparation for PCA
traindata_centered = centered(traindata_pixel)

#gaining input for testing all the funktions:
i = False
inbud = int(input('1. test PCA\n2. test KNN: '))
#match inbud (was passiert hier? syntax hat nicht funktioniert)
if  inbud == 1: #testing out to PCA
    print('shape of the training Data: ' + str(test.shape))
    i = False
    while i == False:
        print('\n')
        
        eiovar = input('type either eigenvector number or explained proportion of variance: ')
        print('shape of the training PCA: ' + str(pca(test, float(eiovar))[0].shape))
        
        if input('again?: ') == 'no':
            i = True
elif inbud == 2: #Testing out the KNN-Method
    while i == False:
        print('\n')

        k = input('whats k?: ')
        item_numbers_of_most_similar_pics = dist(pca(centered(testdata_pixel), 10)[0], pca(test, 10)[0], k)
        
        print(str(most_common_items(labl(item_numbers_of_most_similar_pics,label))))
        print(str(label_test[:10]))
        print(str(quality(label_test[:10], most_common_items(labl(item_numbers_of_most_similar_pics,label)))))

        if input('again?: ') == 'no':
            i = True

def parameter_input():
    choice = int(input('1. test PCA\n2. test KNN: '))
    match choice:
        case 1:
            print('shape of the training Data: ' + str(traindata_centered.shape))
        case 2: