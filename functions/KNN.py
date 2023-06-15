import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn
from scipy import spatial
from scipy.spatial import KDTree
from scipy.stats import mode
from collections import Counter
from sklearn.metrics import confusion_matrix

class_names = np.array(["T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])

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
            print("Image " + i + " was classified as " + class_names[result[i]] + " but is actually a(n) " + class_names[original[i]])
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
    for i in range(0,len(result)):
        if label_test[indices_test][i] != result[i]:
            print("Image " + str(i) + " was classified as " + class_names[result[i]] + " but is actually a(n) " + class_names[label_test[indices_test][i]])
    accuracy = np.sum(result==label_test[indices_test])/testsize
    return accuracy

def conf_matrix(y_pred,label_test,percent=True):
    """Prints a confusion matrix

    Args:
        y_pred (1D numpy array): predicted labels
        label_test (1D numpy array): true labels
        percent (bool, optional): display absolute values or percent. Defaults to True.
    """
    class_names = np.array(["T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])
    conf = sklearn.metrics.confusion_matrix(y_pred, label_test)
    conf_df = pnd.DataFrame(conf, index=class_names, columns=class_names)
    if percent==True:
        conf_df = round(conf_df*100 / conf_df.sum(axis=1),2)
    print(conf_df)

    
def knn_kdtree(PCs_train, PCs_test, k, label_train, label_test, testsize):
    """returns the accuray of the KNN for test images

    Args:
        PCs_train (numpy array): transformed training data
        PCs_test (numpy array): transformed testing data
        k (int): number of neighbours 
        label_train (numpy array 1D): labels of training data
        label_test (numpy array 1D): labels of testing data
        testsize (int): number of testing images you want to classify
        
    Returns:
        knn for number of testsize images
    """

    result = np.array([])
    kd_tree = spatial.KDTree(PCs_train,leafsize=10)
    count_count = 0
    for i in range(0, len(label_test), testsize):
        if count_count % 10 == 0:
            print("Current progress is " + str(count_count*100/(len(label_test)/testsize)) + " percent.")
        count_count += 1
        dist, neighbour_index = kd_tree.query(PCs_test[i:i+testsize, None],p=2,k=k, workers = -1)
        neighbour_label = label_train[neighbour_index]
        batch_result = [mode(neighbour_label,axis=2)[0]]
        batch_result = np.array(batch_result)
        batch_result = np.squeeze(batch_result)
        batch_result = batch_result.astype(int)
        result = result.astype(int)
        result = np.concatenate((result, batch_result), axis=0)
    return result


