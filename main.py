# Pakete etc. 
# Pandas, Numpy und Matplotlib davor über Miniforge Prompt herunterladen sonst geht hier gar nix
# Panda is pandas dataframe
# numpy is number array (general python tool)

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn as skl

from functions.PCA import pca, custum_imshow, ztransform
from functions.KNN import dist, quality, most_common_items, labl

from sklearn.decomposition import PCA
print('\n')

# Import der Dateien 
testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
testdata_pixel = testdata.drop('label', axis=1).to_numpy()
traindata_pixel = traindata.drop('label', axis=1).to_numpy()
label_train = traindata['label'].to_numpy()
label_test = testdata['label'].to_numpy()


img = traindata_pixel.reshape((-1,28,28)) #only relevant for visualisation
num_img = traindata_pixel.shape[0]


# subtracting mean of each pixel while keeping the dimensions of the images to center the images in preparation for PCA
traindata_ztransf = ztransform(traindata_pixel)

#gaining input for testing all the functions:
def testfunction(traindata_ztransf):
    i = False
    choice = int(input('1. test PCA\n2. test KNN: '))
    #match inbud (was passiert hier? syntax hat nicht funktioniert)
    if  choice == 1: #testing out to PCA
        print('shape of the training Data: ' + str(traindata_ztransf.shape))
        while i == False:
            print('\n')
            eiovar = input('type either eigenvector number or explained proportion of variance: ')
            print('shape of the training PCA: ' + str(pca(traindata_ztransf, float(eiovar))[0].shape))
            if input('again? y/n: ') == 'n':
                i = True
    elif choice == 2: #Testing out the KNN-Method
        while i == False:
            print('\n')
            k = input('whats k?: ')
            item_numbers_of_most_similar_pics = dist(pca(ztransform(testdata_pixel), 10)[0], pca(traindata_ztransf, 10)[0], k) 
            print(str(most_common_items(labl(item_numbers_of_most_similar_pics,label_train))))
            print(str(label_train[:10]))
            print(str(quality(label_train[:10], most_common_items(labl(item_numbers_of_most_similar_pics,label_train)))))
            if input('again? y/n: ') == 'n':
                i = True

testfunction(traindata_ztransf)
    

