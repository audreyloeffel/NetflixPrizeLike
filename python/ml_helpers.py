# -*- coding: utf-8 -*-
#!/bin/python3.5

from sklearn.decomposition import NMF

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    nmf_model = NMF(n_components=num_features)
    W = nmf_model.fit_transform(train, num_features)
    Z = nmf_model.components_
    return Z.T, W
