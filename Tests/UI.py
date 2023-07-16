



import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn as skl

from functions.PCA import pca, custum_imshow, centered
from functions.KNN import dist, quality, most_common_items, labl

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
testdata_pixel = testdata.drop('label', axis=1).to_numpy()
traindata_pixel = traindata.drop('label', axis=1).to_numpy()
label_train = traindata['label'].to_numpy()
label_test = testdata['label'].to_numpy()


def testfunction():
    #gaining input for testing all the funktions:
    i = False
    choice = int(input('1. test PCA\n2. test KNN: '))

    if  choice == 1: #testing out to PCA
        print('shape of the training Data: ' + str(traindata_centered.shape))
        while i == False:
            print('\n')
            eiovar = input('type either eigenvector number or explained proportion of variance: ')
            print('shape of the training PCA: ' + str(pca(traindata_centered, float(eiovar))[0].shape))
            if input('again? y/n: ') == 'n':
                i = True
    elif choice == 2: #Testing out the KNN-Method
        while i == False:
            print('\n')
            k = input('whats k?: ')
            item_numbers_of_most_similar_pics = dist(pca(centered(testdata_pixel), 10)[0], pca(traindata_pixel, 10)[0], k)
            print(str(most_common_items(labl(item_numbers_of_most_similar_pics,label_train))))
            print(str(label_test[:10]))
            print(str(quality(label_test[:10], most_common_items(labl(item_numbers_of_most_similar_pics,label_train)))))
            if input('again? y/n: ') == 'n':
                i = True


def test_a_value(value):
    try:
        print(type(value))
    except:
        print()

    try:
        print('valueshape:\n' + str(value.shape))
    except:
        print('no shape')
    try:
        print('value:\n' + str(value))
    except:
        print()