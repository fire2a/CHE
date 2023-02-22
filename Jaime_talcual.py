#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:00:42 2023

@author: rodrigo
"""
from scipy.stats.distributions import chi2
import polytope as pc 
from sklearn.covariance import EllipticEnvelope, MinCovDet
from scipy.spatial import ConvexHull
from scipy.special import gamma
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike 
from typing import Tuple
from cvxpy import Variable, Problem, Minimize, quad_form


def vol_ellipsoid(A): 
    d = A.shape[1]
    volume = (np.pi ** (d/2)) / (gamma(d / 2 + 1) * np.sqrt(np.linalg.det(A)))
    return volume
    

def cov_approach(X, p):
    # Define the problem data
    d = X.shape[1]
    r = chi2.ppf(p, d) # corroborar con matlab
    
    cov1 = np.cov(X.T)
    c = np.mean(X, axis=0)/r
    
    A = np.linalg.inv(cov1)/r
    
    return A, c

def contains(X, p, points): 
    A, c, _ = cov_approach(X, p)
    
    # The following product is the Malanobis' distance between x and c, 
    # when A is the inv(cov) matrix.
    whithin = np.array([
        (((x - c).T @ A @ (x - c)) <= 1).all() for x in points
        ]) 
    
    return whithin.astype(int)
        
def mcd_aproache(X): 
    mod = MinCovDet() 
    mod.fit(X)
    
    A = mod.covariance_
    c = mod.location_
    
    return A, c 

    
    
    


    
    