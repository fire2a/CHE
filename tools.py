#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:51:28 2023

@author: rodrigo
"""
from pyod.models.mcd import MCD
from scipy.stats.distributions import chi2
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.special import gamma
from numpy.typing import ArrayLike 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from env_variables import MODELS

def binary_columns(df: pd.DataFrame) -> list[str]:
    bool_cols = [col for col in df 
             if np.isin(df[col].dropna().unique(), [0, 1]).all()]
    return bool_cols

def get_environment_variables(che: object, variable_list: list[str] = None) -> tuple[list[str], ArrayLike]: 
    species = che.species
    data = che.data.copy()
    
    if variable_list == None:

        columns_to_drop = species + MODELS
        all_columns = data.columns
        environment_variables = [variable not in columns_to_drop for variable in all_columns ]
        data = data.loc[:, environment_variables]
        points = data.to_numpy()

    if isinstance(variable_list, list): 
        data = che.data[variable_list]
        environment_variables = variable_list
        points = data.to_numpy()

    return environment_variables, points

def elliptic_envelope(X: ArrayLike, model: str, p: float = 0.95):
    if model == 'COV': 
        d = X.shape[1]
        r = chi2.ppf(p, d)
        
        cov1 = np.cov(X.T)
        c = np.mean(X, axis=0)
        
        A = np.linalg.inv(cov1)/r

    elif model == 'MCD': 
        mod = MCD(contamination = 0.25).fit(X)
        d = X.shape[1]
        r = chi2.ppf(p, d)

        A = np.linalg.inv(mod.covariance_) / r
        c = mod.location_

    return A, c 

def vol_ellipsoid(A): 
    d = A.shape[1]
    volume = (np.pi ** (d / 2)) / (gamma(d / 2 + 1) * np.sqrt(np.linalg.det(A)))
    return volume

class EllipticEnvelope:
    def __init__(self, X: ArrayLike, model: str,p: float = 0.95):
        self.X = X 
        self.model = model
        self.A, self.c = elliptic_envelope(X, model, p)
        self.volume = vol_ellipsoid(self.A)
    
    def contains(self, points: ArrayLike) -> list[bool]: 
        A = self.A 
        c = self.c
        
        # The following product is the Malanobis' distance between x and c, 
        # when A is the inv(cov) matrix.
        whithin = np.array([
            (((x - c).T @ A @ (x - c)) <= 1).all() for x in points
            ]) 
        
        return whithin.astype(int)

def pca(X: ArrayLike, n_components: int = 3) -> ArrayLike: 
    
    return X 

if __name__ == '__main__': 
    print(__name__)
