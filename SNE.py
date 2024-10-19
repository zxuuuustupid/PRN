# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:38:49 2019

@author: lgq-yun
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn import manifold, datasets



def sen_huatu(X):
    tsne=manifold.TSNE(n_components=2, init='pca',n_iter=1000)
    X_tsne= tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm
'''t-SNE'''
