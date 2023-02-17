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

def cov_approach(X, q):
    # Define the problem data
    d = X.shape[1]
    x2t = chi2.ppf(q, d)
    
    y = Variable(d)
    cov1 = np.cov(X.T)
    m = np.mean(X, axis=0)
    
    A = np.linalg.inv(cov1)
    
    # Define the constraint
    constraint = [quad_form(y - m, A) <= x2t]
    
    # Create the problem and solve it
    problem = Problem(Minimize(0), constraint)
    problem.solve()
    volE = np.sqrt(np.linalg.det(A) * x2t ** d * np.pi ** d / gamma(d/2 + 1))

    return y.value, volE
