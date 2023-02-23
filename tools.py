#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:51:28 2023

@author: rodrigo
"""
from sklearn.covariance import MinCovDet
from scipy.stats.distributions import chi2
import numpy as np
import pandas as pd
from scipy.special import gamma
from numpy.typing import ArrayLike 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def cov(X: ArrayLike, p: float = 0.95):
    d = X.shape[1]
    r = chi2.ppf(p, d)
    
    cov1 = np.cov(X.T)
    c = np.mean(X, axis=0)
    
    A = np.linalg.inv(cov1)/r
    
    return A, c 

def vol_ellipsoid(A): 
    d = A.shape[1]
    volume = (np.pi ** (d/2)) / (gamma(d / 2 + 1) * np.sqrt(np.linalg.det(A)))
    return volume

def pre_plot_depuration(che,  
model : str = None, 
list_var: list[str] = None) -> tuple[pd.DataFrame, ArrayLike]: 

        pred_col = f'{model}_pred'
        if pred_col in che.data:
            cols = list_var.copy()
            cols.append(pred_col)
            points_df = che.data if list_var == None else che.data[cols]
            points = (points_df.
                      query(f'{pred_col} == 1').
                      to_numpy())
            
        elif model == None: 
            points_df = che.data if list_var == None else che.data[list_var]
            points = points_df.to_numpy()
            
        else: 
            raise ValueError('Molde provided is either not been fit or does not exist.\n\
            Models supported:\n\
               1. \'COV\': Melanobis distance model\n\
               2. \'MVEE\': Minimum-Volume Enclosing Ellipsoid\n\
               3. \'MVE\': Minimum-Volume Ellipsoid\n\
               4. \'MCD\': Minimum Covariance Determinant')
        
        dimensions = points_df.loc[:, points_df.columns != pred_col].shape[1]
        return points_df, points, dimensions

def plot3D_multy_convexhulls(che_list: list[object], list_var: list[str] = None): 
    fig = plt.figure() 
    ax = Axes3D(fig)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i, _ in enumerate(che_list, 0)]
        
    for i, che in enumerate(che_list, start = 0): 
        _, points, dimensions = pre_plot_depuration(che, list_var = list_var)
        for simplex in che.convex_hull.simplices:
            ax.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], 's-', 
            color = color[i])
        plt.show()

def plot3D_convexhull(che,  model : str = None, 
             list_var: list[str] = None, 
             color = '#3ca3a3'):

    points_df, points, dimensions = pre_plot_depuration(che, model, list_var)
    if dimensions == 3: 
        fig = plt.figure(figsize = (20,10),facecolor="w") 
        ax = plt.axes(projection="3d") 
        for simplex in che.convex_hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 's-', 
            color = color)

def plot2D_convexhull(che,  model : str = None, 
             list_var: list[str] = None, 
             color = '#3ca3a3'):

        points_df, points, dimensions = pre_plot_depuration(che, model, list_var)
        if dimensions == 2: 
            if model != None: 
                if list_var == None: 
                    warn('The plotting variables were not defined. \
    The first two columns were taken. The first two columns were taken.', 
    stacklevel = 2)
                    x = che.data.iloc[:,0].to_numpy()
                    y = che.data.iloc[:,1].to_numpy()
                
                else: 
                    loc = []
                    for var in list_var: 
                        col_i = points_df.columns.get_loc(var)
                        loc.append(col_i)
                    x = che.data.iloc[:,loc[0]].to_numpy()
                    y = che.data.iloc[:,loc[1]].to_numpy()
                    
                plt.scatter(x, y, c = 'k', s = 8)

            for simplex in che.convex_hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k')
            plt.fill(points[che.convex_hull.vertices,0], 
                    points[che.convex_hull.vertices,1], 
                    color, 
                    alpha = 0.3)
            plt.title(f"CHE-{model} approach")

        else: 
            pass

class COV:
    def __init__(self, X: ArrayLike, p: float = 0.95):
        self.X = X 
        self.A, self.c = cov(X, p)
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
    
def mcd(X: ArrayLike, p: float = 0.95): 
    mod = MinCovDet(support_fraction = 0.25).fit(X)
    d = X.shape[1]
    r = chi2.ppf(p, d)
    
    A = np.linalg.inv(mod.covariance_)/r
    c = mod.location_
    
    return A, c 

class MCD:
    def __init__(self, X: ArrayLike, p: float = 0.95):
        self.X = X 
        self.A, self.c = mcd(X, p)
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

if __name__ == '__main__': 
    print(__name__)
