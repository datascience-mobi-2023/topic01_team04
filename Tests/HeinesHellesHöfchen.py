# Pakete etc. 
# Pandas, Numpy und Matplotlib davor über Miniforge Prompt herunterladen sonst geht hier gar nix
# Panda is pandas dataframe
# numpy is number array (general python tool)

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import sklearn as skl

from functions.PCA import pca, custum_imshow, centered
from functions.KNN import dist, quality, most_common_items, labl, knn_quality
from functions.UI import *

print('\n')
# Import der Dateien 
# ANMERKUNG: Funktioniert auch nur, wenn die Dateien in der SELBEN Directory wie das Skript sind -> Im Git-Ordner
# Problem mit Git da CSV-Dateien zu groß sind um sie zu committen, sonst kann man gar nix mehr pushen 
# Dateien in GitHub Desktop ab in .gitignore, dann verpisst sich das Problem

testdata = pnd.read_csv('fashion-mnist_test.csv')
traindata = pnd.read_csv('fashion-mnist_train.csv')
# 1. Spalte wird gedropped/ gelöscht, weil dort die Kategorie des Kleidungsstücks steht & kein Intensitätswert eines Pixels, danach sind es 28 x 28 = 784 columns
testdata_pixel = testdata.drop('label', axis=1).to_numpy()
traindata_pixel = traindata.drop('label', axis=1).to_numpy()
# numpy array mit Kategorien der Kleidungsstücke wird erstellt
label_train = traindata['label'].to_numpy()
label_test = testdata['label'].to_numpy()
#class_names = np.array("T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

img = traindata_pixel.reshape((-1,28,28)) #only relevant for visualisation
num_img = traindata_pixel.shape[0]
print(num_img)
PCs_train, PCs_test = pca(traindata_pixel,testdata_pixel,0.8)

print(knn_quality(PCs_train, PCs_test, 10, label_train, label_test, 50))
#print(np.corrcoef(pca(traindata_pixel,testdata_pixel,12)[1],rowvar=False)) #look at transformed training or test data: in both cases, columns have no correlation with each other


# early stopping in KNN to make KNN faster
from heapq import heappush, heappop


def euclidean_distance(point1, point2):
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)


def knn_with_early_stopping(query_point, data_points, k, distance_threshold):
    neighbors = []  # Container for nearest neighbors
    stopping_condition = False  # Initialize stopping condition
    index = 0  # Initialize index counter

    while index < len(data_points) and not stopping_condition:
        current_point = data_points[index]
        distance = euclidean_distance(query_point, current_point)

        if len(neighbors) < k:
            # Add current point to neighbors
            heappush(neighbors, (-distance, current_point))
        else:
            # Check if distance is less than the kth neighbor
            if distance < -neighbors[0][0]:
                heappop(neighbors)
                heappush(neighbors, (-distance, current_point))

            # Check if distance exceeds threshold
            if distance >= distance_threshold:
                stopping_condition = True

        index += 1

    return [neighbor[1] for neighbor in neighbors]


# Example usage
query = np.array([0, 0])
data = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
k = 3
threshold = 3.0

nearest_neighbors = knn_with_early_stopping(query, data, k, threshold)
print(nearest_neighbors)