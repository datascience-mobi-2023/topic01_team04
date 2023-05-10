import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt


#schleifen sind langsam wie sau, hier ist definitiv noch raum für improvement


def dist(PCs_test,PCs_train,k):
    final_result = [[]]
    for i in range(0,len(PCs_test)):
        
        result = []
        for y in range(0,len(PCs_train)):
            result = np.append(result, [np.linalg.norm(PCs_test[i]-PCs_train[y])])

        #print(result)
        classes = np.argsort(result)
        class_k = classes[:int(k)]
        
        final_result = np.append(final_result, [[class_k]])

        print(final_result)

    return class_k

