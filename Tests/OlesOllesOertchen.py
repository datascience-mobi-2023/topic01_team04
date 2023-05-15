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