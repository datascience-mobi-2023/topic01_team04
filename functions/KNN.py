import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from collections import Counter


#schleifen sind langsam wie sau, hier ist definitiv noch raum für improvement

""" Orginal
def dist(PCs_test,PCs_train,k):
    k = int(k)
    final_result = np.zeros((len(PCs_test),k))
    for i in range(0,len(PCs_test)):
        
        result = np.array([])
        for y in range(0,len(PCs_train)):
            result = np.append(result, [np.linalg.norm(PCs_test[i]-PCs_train[y])])
        
        class_k = np.argsort(result)[:k]
        print(result[class_k])
        print(result.shape)
        
        
        
        final_result[i] = class_k

        print(final_result)
        print('\n')
    return class_k
"""



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
    n_test_samples = len(PCs_test)
    n_train_samples = len(PCs_train)
    distances = np.linalg.norm(PCs_test[:10, None] - PCs_train, axis=2)
    item_numbers_of_most_similar_pics = np.argpartition(distances, kth=k-1, axis=1)[:, :k]

    return item_numbers_of_most_similar_pics


def labl(item_numbers_of_most_similar_pics,label):
    """
    Calculates the labels of the item numbers that were previously described as the closest ones.
    """
    item = item_numbers_of_most_similar_pics
    labls = label[item]

    return labls

def most_common_items(arr):
    result = []
    for subarr in arr:
        data = Counter(subarr)
        most_common_item = data.most_common(1)[0][0]
        result.append(most_common_item)
    return result


