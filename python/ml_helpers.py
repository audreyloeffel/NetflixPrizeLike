# -*- coding: utf-8 -*-
#!/bin/python3.5

"""
Some helpers needed related to machine learning.
"""

from sklearn.decomposition import NMF

"""
Factorizes a matrix following the Non-Negative-Factorization
algorithm.
"""
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    nmf_model = NMF(n_components=num_features)
    W = nmf_model.fit_transform(train, num_features)
    Z = nmf_model.components_
    return Z.T, W
