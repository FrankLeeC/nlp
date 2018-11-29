# -*- coding:utf-8 -*-

import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1]])
X_embedded = TSNE(n_components=2, init="pca", n_iter=200).fit_transform(X)
print(X_embedded.shape)
print(X_embedded)