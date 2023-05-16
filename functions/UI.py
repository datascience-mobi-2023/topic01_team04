#sys.path
from functions.PCA import pca, custum_imshow, centered
from functions.KNN import dist, quality, most_common_items, labl



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
            item_numbers_of_most_similar_pics = dist(pca(centered(testdata_pixel), 10)[0], pca(test, 10)[0], k)
            print(str(most_common_items(labl(item_numbers_of_most_similar_pics,label))))
            print(str(label_test[:10]))
            print(str(quality(label_test[:10], most_common_items(labl(item_numbers_of_most_similar_pics,label)))))
            if input('again? y/n: ') == 'n':
                i = True