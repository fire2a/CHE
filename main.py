#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:10:54 2023

@author: rodrigo
"""

from pypoman.duality import compute_polytope_vertices
from tools import COV, MCD
from plot import plot_ndimensional_convexhull
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike 
from warnings import warn
import os
from pprint import pprint

class CHE:
    def __init__(self, 
                 data : pd.DataFrame | ArrayLike = None, 
                 convex_hull: ConvexHull = None, 
                 model: str = None):
        
        self.convex_hull = convex_hull
        self.data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.model = model
        if convex_hull: 
            d = convex_hull.vertices.shape[1]
            self.A = convex_hull.equations[:,:d]
            self.b = -self.convex_hull.equations[:,d]
            self.volume =  convex_hull.volume
            self.vertices = convex_hull.vertices

    def ch(self, list_var : list[str] = None): 
        points = self.data if list_var == None else self.data[list_var]
        
        points = points.to_numpy()
        d = points.shape[1]
        
        self.convex_hull = ConvexHull(points)
        
        # H-representation: Ax <= b
        self.A = self.convex_hull.equations[:,:d]
        self.b = -self.convex_hull.equations[:,d]
        
        # V-representation
        self.vertices = points[self.convex_hull.vertices]
        
        # get Volume        
        self.volume = self.convex_hull.volume
        

    def cov(self, 
            p = 0.95, 
            list_var : list[str] = None) -> None:
        
        # fit model
        points = self.data if list_var == None else self.data[list_var]
        points = points.to_numpy()
        d = points.shape[1]
        
        ellip = COV(points, p)
        
        pred = ellip.contains(points)
        pred[pred == -1] = 0 
        
        new_points = points[pred.astype(bool)]
        
        self.model = 'COV'
        self.data['COV'] = pred

        self.convex_hull = ConvexHull(new_points)
        
        # H-representation: Ax <= b
        self.A = self.convex_hull.equations[:,:d]
        self.b = -self.convex_hull.equations[:,d]
        
        # V-representation
        self.vertices = new_points[self.convex_hull.vertices]
        
        # get Volume        
        self.volume = self.convex_hull.volume

    def mcd(self, 
            p = 0.95, 
            list_var : list[str] = None) -> None:
        
        
        # fit model
        points = self.data if list_var == None else self.data[list_var]
        points = points.to_numpy()
        d = points.shape[1]
        
        ellip = MCD(points, p)
        
        pred = ellip.contains(points)
        pred[pred == -1] = 0 
        
        new_points = points[pred.astype(bool)]
        
        self.model = 'MCD'
        self.data['MCD'] = pred
        
        self.convex_hull = ConvexHull(new_points)
        
        # H-representation: Ax <= b
        self.A = self.convex_hull.equations[:,:d]
        self.b = -self.convex_hull.equations[:,d]
        
        # V-representation
        self.vertices = new_points[self.convex_hull.vertices]
        
        # get Volume        
        self.volume = self.convex_hull.volume
 
    
    def contains(self, points: ArrayLike) -> list[int]: 
        
        if isinstance(points,(np.ndarray, np.generic)):
            pass
        else: 
            points = points.to_numpy()
            
        bool_list = np.array([
            (np.sum(point*self.A, axis = 1) <= self.b).all() for point in points
            ])
            
        return bool_list.astype(int)
        
    
    def plot(self): 
        plot_ndimensional_convexhull(self)


def intersection(ches: list[object]) -> CHE:
    for i, che in enumerate(ches): 
        if i == 0: 
            A = che.A
            b = che.b
        else: 
            A = np.vstack((A, che.A))
            b = np.hstack((b, che.b))
        
    vertices = compute_polytope_vertices(A, b)  
    vertices = np.array(vertices)
    intersect = CHE(vertices)
    intersect.ch()
    
    return intersect

if __name__ == '__main__': 
    IN_FOLDER = 'data'
    d_fn = os.path.join(IN_FOLDER, 'D.csv')
    b_bar = pd.read_csv(d_fn, usecols = ['Bbar'])
    
    pcs_fn = os.path.join(IN_FOLDER, 'TablaPCs.csv')
    # Visual Python: Data Analysis > File
    pcs = pd.read_csv(pcs_fn, sep = ' ', 
                      usecols = ['PC1', 'PC2', 'PC3', 'PC4'])

    # convert dataframe to arrays
    data = pd.concat([b_bar, pcs], axis=1)
    data = data.query('Bbar == 1')
    
    
    points1 = data[['PC1', 'PC2', 
    'PC3'
    ]].sample(100)
    points2 = data[['PC1', 'PC2', 
    'PC3'
    ]].sample(100)
    
    
    hull_1 = CHE(points1)
    hull_1.mcd(list_var = ['PC1', 'PC2', 
    'PC3'
    ])
    print(f'1 con 1 -> {hull_1.contains(points1).sum()}/{len(points1)}')
    
    hull_2 = CHE(points2)
    hull_2.cov(list_var = ['PC1', 'PC2', 
    'PC3'
    ])

    print(f'1 con 2 -> {hull_2.contains(points1).sum()}/{len(points1)}')
    print(f'2 con 1 -> {hull_1.contains(points2).sum()}/{len(points2)}')
    print(f'2 con 2 -> {hull_2.contains(points2).sum()}/{len(points2)}')
    
    print(hull_1.volume != hull_2.volume)
    
    ches = [hull_1, hull_2]
    
    che3 = intersection(ches)
    
    print(che3.volume <= hull_1.volume)
    print(che3.volume <= hull_2.volume)
    print(che3.volume <= hull_2.volume)
    
    print(f'3 con 1 -> {che3.contains(points1).sum()}/{len(points1)}')
    # # print(f'1 con 3 -> {che1.contains(points2).sum()}/{len(points2)}')
    print(f'3 con 2 -> {che3.contains(points2).sum()}/{len(points2)}')
    
    # plt.figure(1)
    # hull_1.plot(model = 'COV', list_var = ['PC1', 'PC2', 'PC3'])
    # # hull_2.plot(model = 'COV', list_var = ['PC1', 'PC2']) 
    # # che3.plot(color = 'r', list_var = [0,1])
    # plt.show()

    points = data.loc[:, data.columns != 'Bbar']
    che = CHE(points)
    che.cov()
    che.plot()
