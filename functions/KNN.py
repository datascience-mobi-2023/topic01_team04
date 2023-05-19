import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from collections import Counter

def dist(PCs_test, PCs_train, k):
    """
    Calculates the k-nearest neighbors of the test data based on the training data.
    
    Parameters:
    PCs_test (numpy array): The test data matrix with shape (n_test_samples, n_features).
    PCs_train (numpy array): The training data matrix with shape (n_train_samples, n_features).
    k (int): The number of nearest neighbors to consider.
    
    Returns:
    numpy array: An array of shape (n_test_samples, k) containing the indices of the k-nearest neighbors for each test sample.
    """
    k = int(k)
    distances = np.linalg.norm(PCs_test[:10, None] - PCs_train, axis=2)
    item_numbers_of_most_similar_pics = np.argpartition(distances, kth=k-1, axis=1)[:, :k]

    return item_numbers_of_most_similar_pics


def labl(item_numbers_of_most_similar_pics,label):
    """
    Calculates the labels of the item numbers that were previously described as the closest ones.
    """
    labls = label[item_numbers_of_most_similar_pics]

    return labls

def most_common_items(arr):
    result = []
    for subarr in arr:
        data = Counter(subarr)
        most_common_item = data.most_common(1)[0][0]
        result.append(most_common_item)
    result = np.array(result)
    return result


def quality(orginal, result):
    fal = 0
    for i in range(0,len(orginal)):
        if orginal[i] != result[i]:
            fal += 1
    false_quote = fal/len(orginal)
    return false_quote

def knn_quality(PCs_train, PCs_test, k, label_train, label_test, testsize, random = False):
    """returns the accuray of the KNN for test images

    Args:
        PCs_train (numpy array): transformed training data
        PCs_test (numpy array): transformed testing data
        k (int): number of neighbours 
        label_train (numpy array 1D): labels of training data
        label_test (numpy array 1D): labels of testing data
        testsize (int): number of testing images you want to classify
        random (boolean): when True, looks at random test images, when False, looks at first testsize images

    Returns:
        float: accuracy
    """
    if random == True:
        indices_test = np.random.choice(PCs_test.shape[0],testsize)
    else:
        indices_test = [*range(testsize)]
    distances = np.linalg.norm(PCs_test[indices_test, None] - PCs_train, axis=2)
    neighbour_index = np.argpartition(distances, kth=k-1, axis=1)[:,:k]
    neighbour_label = label_train[neighbour_index]
    result = []
    for subarray in neighbour_label:
        data = Counter(subarray)
        most_common_item = data.most_common(1)[0][0]
        result.append(most_common_item)
    result = np.array(result)
    accuracy = np.sum(result==label_test[indices_test])/testsize
    return accuracy