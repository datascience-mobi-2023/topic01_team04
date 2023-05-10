import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt



def dist(PCs_test,PCs_train,k):
    for i in range(0,len(PCs_test)):
        result = []
        for y in range(0,len(PCs_train)):
            result = numpy.append (result, [np.linalg.norm(PCs_test[i]-PCs_train[y])]) 
            print(result)
            classes = np.argsort(result)
            print(classes)
            return classes

