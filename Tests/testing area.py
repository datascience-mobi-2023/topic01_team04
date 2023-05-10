import numpy as np


test = [[[1,1.2],[2,2.2]],[[3,3.2],[4,4.2]]]
for a in range(0,2):
    for b in range(0,2):
        for c in range(0,2):
            print(test[a][b][c])


test2 = [[[100,1.2],[2,2.2]],[[3,3.2],[4,4.2]]] 


# np.linalg.norm(img1[i]-img2[y])

def distance(img1,img2):
    img1 = np.ravel(img1)
    img2 = np.ravel(img2)
    dis = []
    for i in range(0,len(img1)):
        for y in range(0,len(img2)):
            dis += [np.linalg.norm(img1[i]-img2[y])] #calculates L2 norm of distance vector, which is equal to the euclidean distance (Betrag des Vektors zum Quadrat)
    
    
    return dis


print(distance(test,test2))

# rotations Matrix ist universell anwendbar, also auch auf testdatensatz

