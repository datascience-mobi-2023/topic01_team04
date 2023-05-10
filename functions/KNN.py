import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt



def dist(PCs_test,PCs_train):
    for i in range(0,len(PCs_test)):
        result = []
        for y in range(0,len(PCs_train)):
            result += [np.linalg.norm(PCs_test[i]-PCs_train[y])]
            classes = np.argsort(result)
            return classes

