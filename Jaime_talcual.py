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
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike 
from typing import Tuple


def COV(data : pd.DataFrame[int | float] | ArrayLike, 
        p: float = 0.95) -> Tuple[EllipticEnvelope, float]:
    d = data.shape[1] # get the number of columns or dimensions
    x2t = chi2.ppf(p, df = d)
    
    y = np.zeros(d,1)
    cov1 = data.cov(min_periods = None, ddof = 1)
    
    A = np.linalg.pinv
    return E, volE
    