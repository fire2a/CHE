#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:10:54 2023

@author: rodrigo
"""

from pypoman.duality import compute_polytope_vertices
from tools import EllipticEnvelope, binary_columns, get_environment_variables
from plot import plot_ndimensional_convexhull, plot3D_convexhulls
from env_variables import MODELS
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike 
from functools import reduce
import os

# from collections import namedtuple
# # Declaring namedtuple()
# Student = namedtuple('Student', ['name', 'age', 'DOB'])
 
# # Adding values
# S = Student('Nandini', '19', '2541997')

class CHE:
    def __init__(self, 
                 data : pd.DataFrame = None, 
                 convex_hull: ConvexHull = None):
        """
        

        Parameters
        ----------
        data : pd.DataFrame, optional
            DESCRIPTION. The default is None.
        convex_hull : ConvexHull, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        self.convex_hull = convex_hull
        self.data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.model = None
        self.species = binary_columns(data)
        env_var_bools, self.array = get_environment_variables(self)
        self.environment_variables = data.columns[env_var_bools]

        if convex_hull: 
            d = convex_hull.vertices.shape[1]
            self.A = convex_hull.equations[:,:d]
            self.b = -self.convex_hull.equations[:,d]
            self.volume =  convex_hull.volume
            self.vertices = convex_hull.vertices
            self.points = convex_hull.vertices

    def ch(self, list_var : list[str] = None): 
        """
        

        Parameters
        ----------
        list_var : list[str], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        variables_selected = np.intersect1d(self.environment_variables, list_var) if list_var != None else self.environment_variables
        _, points = get_environment_variables(self, list_var)
        
        d = points.shape[1]
        
        self.convex_hull = ConvexHull(points)
        
        # H-representation: Ax <= b
        self.A = self.convex_hull.equations[:,:d]
        self.b = -self.convex_hull.equations[:,d]
        
        # V-representation
        self.vertices = points[self.convex_hull.vertices]
        self.points = points
        
        # get Volume        
        self.volume = self.convex_hull.volume
        

    def cov(self, 
            p = 0.95, 
            list_var : list[str] = None) -> None:
        """
        

        Parameters
        ----------
        p : TYPE, optional
            DESCRIPTION. The default is 0.95.
        list_var : list[str], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None
            DESCRIPTION.

        """
        

        # fit model
        _, points = get_environment_variables(self, list_var)
        
        d = points.shape[1]
        
        ellip = EllipticEnvelope(points, 'COV', p)
        
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
        self.points = new_points
        
        # get Volume        
        self.volume = self.convex_hull.volume

    def mcd(self, 
            p = 0.95, 
            list_var : list[str] = None) -> None:
        """
        

        Parameters
        ----------
        p : TYPE, optional
            DESCRIPTION. The default is 0.95.
        list_var : list[str], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        
        # fit model
        _, points = get_environment_variables(self, list_var)
        d = points.shape[1]
        
        ellip = EllipticEnvelope(points, 'COV', p)
        
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
        self.points = new_points
        
        # get Volume        
        self.volume = self.convex_hull.volume
 
    
    def contains(self, points: ArrayLike) -> list[int]: 
        """
        

        Parameters
        ----------
        points : ArrayLike
            DESCRIPTION.

        Returns
        -------
        list[int]
            DESCRIPTION.

        """          
            
        bool_list = np.array([
            (np.sum(point*self.A, axis = 1) <= self.b).all() for point in points
            ])
            
        return bool_list.astype(int)
        
    
    def plot(self, d3: bool = False, ) -> None:
        """
        

        Parameters
        ----------
        d3 : bool, optional
            DESCRIPTION. The default is False.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        if d3: 
            plot3D_convexhulls([self])
        else: 
            plot_ndimensional_convexhull(self)



def intersection(ches: list[object]) -> CHE:
    """
    

    Parameters
    ----------
    ches : list[object]
        DESCRIPTION.

    Returns
    -------
    CHE
        DESCRIPTION.

    """

    variables = [che.data.columns for che in ches]
    variables = reduce(np.intersect1d, variables)
    bool_map = [(var not in MODELS) for var in variables]
    variables = variables[bool_map]

    bool_map = [(var not in MODELS) for var in variables]
    variables = variables[bool_map]
    species = []

    for i, che in enumerate(ches): 
        species.append(che.species)
        if i == 0: 
            A = che.A
            b = che.b
            data = che.data.copy()
        else: 
            data = pd.concat([data, che.data.copy()], axis = 0, ignore_index = True)
            A = np.vstack((A, che.A))
            b = np.hstack((b, che.b))
        
    vertices = compute_polytope_vertices(A, b)  
    vertices = np.array(vertices)
    vertices_data = pd.DataFrame(vertices, columns = variables, )
    intersect = CHE(vertices_data)
    intersect.ch()

    data = data.fillna(0)
    intersect.data = data

    species = [val for sublist in species for val in sublist]
    intersect.species = species
    return intersect

if __name__ == '__main__': 
    IN_FOLDER = 'data'
    d_fn = os.path.join(IN_FOLDER, 'D.csv')
    b_bar = pd.read_csv(d_fn, usecols = ['Bbar', 'Eisa'])
    
    pcs_fn = os.path.join(IN_FOLDER, 'TablaPCs.csv')
    # Visual Python: Data Analysis > File
    pcs = pd.read_csv(pcs_fn, sep = ' ', 
                      usecols = ['PC1', 'PC2', 'PC3', 'PC4'])

    # convert dataframe to arrays
    data = pd.concat([b_bar, pcs], axis=1)
    
    Bbar = CHE(data.query('Bbar == 1')[['Bbar', 'PC1', 'PC2', 'PC3']])
    Bbar.cov()

    Eisa = CHE(data.query('Eisa == 1')[['Eisa', 'PC1', 'PC2', 'PC3']])

    Eisa.cov()

    ches = [Bbar, Eisa]

    Inte = intersection(ches)

    print(f'environment variables: {Bbar.environment_variables}')
    plot3D_convexhulls([Bbar, Eisa], normal_convexhull=True)
    Inte.plot(d3=True)