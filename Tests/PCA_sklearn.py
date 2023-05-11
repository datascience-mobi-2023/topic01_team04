import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_sk (centered_img, prop_variance):

    X = images.reshape(images.shape[0], -1)
    pcask = PCA(n_components=2) # wieso components = 2??? 
    pcask.fit(X)
