#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:10:54 2023

@author: rodrigo
"""

import polytope as pc 
from sklearn.covariance import EllipticEnvelope, MinCovDet
from scipy.spatial import ConvexHull
from pypoman.duality import compute_polytope_vertices
import pandas as pd
import cvxpy as cp
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike 
from typing import Tuple, List
import cdd as pcdd
import os

class CHE:
    def __init__(self, convex_hull = None):
        self.convex_hull = convex_hull
        if convex_hull: 
            self.A = convex_hull.equations 
            self.volume =  convex_hull.volume
            self.vertices = convex_hull.vertices
            
    def ch(self, vertices): 
        n = vertices.shape[1]
        self.convex_hull = ConvexHull(vertices)
        
        #H-representation: Ax <= b
        self.A = self.convex_hull.equations[:,:n]
        self.b = -self.convex_hull.equations[:,n]
        
        # V-representation
        self.vertices = vertices
        
        self.volume = self.convex_hull.volume
        self.data = vertices

    def cov(self, data : pd.DataFrame | ArrayLike, 
            p = 0.95, cols = ['all']) -> None:
        
        model1 = EllipticEnvelope(contamination = 1-p) 
        # fit model
        points = data[['PC1', 'PC2']].to_numpy()
        model1.fit(points)
        
        pred1 = model1.predict(points)
        
        data = data.copy()
        n = data.shape[1]
        data['cov_pred'] = pred1
        
        new_points = (
            data.query('cov_pred == 1')[['PC1', 'PC2']]
            .to_numpy()
        )

        self.convex_hull = ConvexHull(new_points)
        
        #H-representation: Ax <= b
        self.A = self.convex_hull.equations[:,:n]
        self.b = -self.convex_hull.equations[:,n]
        
        # V-representation
        self.vertices = new_points[self.convex_hull.vertices]
        
        
        self.data = data
        self.volume = self.convex_hull.volume
        
    
    def contains(self, points: ArrayLike) -> List[bool]: 
        
        if isinstance(points,(np.ndarray, np.generic)):
            pass
        else: 
            points = points.to_numpy()
            
        print('dejar como binario')
        bool_list = [(np.sum(point*self.A, axis = 1) <= self.b).all() for point in points]
            
        return bool_list
        
    
    def plot(self, color = '#3ca3a3'): 
        hull = self.convex_hull
        points = (
            self.
            data.
            query('cov_pred == 1')[['PC1', 'PC2']]
            .to_numpy()
        )
        plt.scatter(self.data.PC1.to_numpy(), self.data.PC2.to_numpy(), c = 'k', s = 8)
        
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k')
        plt.fill(points[hull.vertices,0], 
                 points[hull.vertices,1], 
                 color, 
                 alpha = 0.3)
        plt.title("CHE-COV approach")

def intersection(ches: List[CHE]) -> CHE:
    for i, che in enumerate(ches): 
        if i == 0: 
            A = che.A
            b = che.b
        else: 
            A = np.vstack((A, che.A))
            b = np.hstack((b, che.b))
        
    vertices = compute_polytope_vertices(A, b)  
    vertices = np.array(vertices)
    intersect = CHE()
    intersect.ch(vertices)
    
    return intersect
    
        
    
    # n_cols = intersection.shape[1] 
    # solution = np.linalg.solve(intersection.T[:n_cols - 1].T, intersection.T[n_cols - 1].T)
    
    # # make the V-representation of the intersection
    # mat = pcdd.Matrix(intersection)
    # mat.rep_type = pcdd.RepType.INEQUALITY
    # polyintersection = pcdd.Polyhedron(mat)

    # # get the vertices; they are given in a matrix prepended by a column of ones
    # vintersection = polyintersection.get_generators()

    # # get rid of the column of ones
    # ptsintersection = np.array(vintersection).T[1:].T

    # # these are the vertices of the intersection; it remains to take
    # # the convex hull
    # ch = ConvexHull(ptsintersection)
    # hull = CHE(ch)
    # hull.vertices = ptsintersection
    
    # return hull


def test(): 
    IN_FOLDER = 'data'
    d_fn = os.path.join(IN_FOLDER, 'D.csv')
    b_bar = pd.read_csv(d_fn, usecols = ['Bbar'])
    
    pcs_fn = os.path.join(IN_FOLDER, 'TablaPCs.csv')
    # Visual Python: Data Analysis > File
    pcs = pd.read_csv(pcs_fn, sep = ' ', 
                      usecols = ['PC1', 'PC2'])

    # convert dataframe to arrays
    data = pd.concat([b_bar, pcs], axis=1)
    data = data.query('Bbar == 1')
    
    
    points1 = data[['PC1', 'PC2']].sample(100)
    points2 = data[['PC1', 'PC2']].sample(100)
    
    
    hull_1 = CHE()
    hull_1.cov(points1, 0.8)
    print(f'1 con 1 -> {hull_1.contains(points1).count(True)}/{len(points1)}')
    
    hull_2 = CHE()
    hull_2.cov(points2, 0.8)

    print(f'1 con 2 -> {hull_2.contains(points1).count(True)}/{len(points1)}')
    print(f'2 con 1 -> {hull_1.contains(points2).count(True)}/{len(points2)}')
    print(f'2 con 2 -> {hull_2.contains(points2).count(True)}/{len(points2)}')
    
    print(hull_1.volume != hull_2.volume)
    
    ches = [hull_1, hull_2]
    
    che3 = intersection(ches)
    
    print(che3.volume <= hull_1.volume)
    print(che3.volume <= hull_2.volume)
    print(che3.volume <= hull_2.volume)
    
    print(f'3 con 1 -> {che3.contains(points1).count(True)}/{len(points1)}')
    # # print(f'1 con 3 -> {che1.contains(points2).sum()}/{len(points2)}')
    print(f'3 con 2 -> {che3.contains(points2).count(True)}/{len(points2)}')
    
    plt.figure(1)
    hull_1.plot()
    hull_2.plot()
    che3.plot(color = 'r')
    # plt.scatter(data.PC1.to_numpy(), data.PC2.to_numpy(), c = 'black', s = 2, marker = '.')
    plt.show()

if __name__ == '__main__': 
    test()