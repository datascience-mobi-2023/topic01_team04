import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt


#schleifen sind langsam wie sau, hier ist definitiv noch raum für improvement


def dist(PCs_test,PCs_train,k):
    k = int(k)
    final_result = np.zeros((len(PCs_test),k),dtype=np.int8)
    for i in range(0,len(PCs_test)):
        
        result = np.array([],dtype=np.int8)
        for y in range(0,len(PCs_train)):
            result = np.append(result, [np.linalg.norm(PCs_test[i]-PCs_train[y])])

        #print(result)
        class_k = np.argsort(result)[:k]
        
        
        final_result[i] = class_k

        print(final_result)
        print('\n')
    return class_k

