#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:10:14 2023

@author: rodrigo


EE
    - COV: Melanobis distance model
    - MVEE: Minimum-Volume Enclosing Ellipsoid
    - MVE: Minimum-Volume Ellipsoid
    - MCD: Minimum Covariance Determinant
    
    

"""

from sklearn.covariance import EllipticEnvelope, MinCovDet
from scipy.spatial import ConvexHull
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
from typing import List
#import torch.linalg as la

pi = np.pi
sin = np.sin
cos = np.cos


# http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
# http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440
# https://minillinim.github.io/GroopM/dev_docs/groopm.ellipsoid-pysrc.html


def mvee(points, tol = 0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
        
    c = u*points
    c = np.squeeze(np.asarray(c))
    
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d 
    A = np.asarray(A)
    return A, c

#some random points
points = np.array([[ 0.53135758, -0.25818091, -0.32382715], 
                   [ 0.58368177, -0.3286576,  -0.23854156,], 
                   [ 0.18741533,  0.03066228, -0.94294771], 
                   [ 0.65685862, -0.09220681, -0.60347573],
                   [ 0.63137604, -0.22978685, -0.27479238],
                   [ 0.59683195, -0.15111101, -0.40536606],
                   [ 0.68646128,  0.0046802,  -0.68407367],
                   [ 0.62311759,  0.0101013,  -0.75863324]])



A, centroid = mvee(points)    
U, D, V = la.svd(A)    
rx, ry, rz = 1./np.sqrt(D)
u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

def ellipse(u,v):
    x = rx*cos(u)*cos(v)
    y = ry*sin(u)*cos(v)
    z = rz*sin(v)
    return x,y,z

E = np.dstack(ellipse(u,v))
E = np.dot(E,V) + centroid
x, y, z = np.rollaxis(E, axis = -1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.05)
ax.scatter(points[:,0],points[:,1],points[:,2])

plt.show()

class EllipsoidTool:
    """Some stuff for playing with ellipsoids"""
    def __init__(self, points = None, tolerance = 0.01):
        self.points = points 
        self.tolerance = tolerance
        self.center, self.radii, self.rotation = self.getMinVolEllipse(points, tolerance)
        self.volume = self.getEllipsoidVolume(self.radii)
        

    def getMinVolEllipse(self, P, tolerance):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(la.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = la.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) - 
                       np.array([[a * b for b in center] for a in center])
                       ) / d
                       
        # Get the values we'd like to return
        U, s, rotation = la.svd(A)
        radii = 1.0/np.sqrt(s)
        
        return (center, radii, rotation)

    def getEllipsoidVolume(self, radii):
        """Calculate the volume of the blob"""
        return 4./3.*np.pi*radii[0]*radii[1]*radii[2]

    def plotEllipsoid(self, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
        """Plot an ellipsoid"""
        center = self.center
        radii = self.radii
        rotation = self.rotation
        points = self.points
        
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        
        # cartesian coordinates that correspond to the spherical angles:
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        # rotate accordingly
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    
        if plotAxes:
            # make some purdy axes
            axes = np.array([[radii[0],0.0,0.0],
                             [0.0,radii[1],0.0],
                             [0.0,0.0,radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)
    
    
            # plot axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color=cageColor)
    
        # plot ellipsoid
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
        ax.scatter(points[:,0],points[:,1],points[:,2], color = 'k', s = 5)
        
        if make_ax:
            plt.show()
            plt.close(fig)
            del fig


class CHE:
    def __init__(self, 
                 datafile : str = "environment_data.csv"):
       self.df = pd.read_csv(datafile)
       
    
    def selected_model(self, 
                       model_name): 
           
        if model_name == 'COV':
           self.model = EllipticEnvelope()
        elif model_name == 'MCD':
           self.model = MinCovDet()
        
        elif model_name == 'MVE': 
           self.model = MinCovDet()
    
        elif model_name == 'MVEE': 
           self.model = mvee()
           
        self.model_name = model_name

        
    
    def split(self, 
              x : List[str], 
              y : str, 
              test_size: float = 0.2, 
              sead: int = None) -> None:
        
        X = np.array(self.df[x])
        y = np.array(self.df[y])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, 
            y, 
            test_size = test_size, 
            random_state = None
            )
    
    def fit(self):
        self.model = self.model.fit(self.X_train, self.y_train)
    
    def predict(self, input_value):
        if input_value == None:
            result = self.model.fit(self.X_test)
        else: 
            result = self.model.fit(np.array([input_value]))
        return result

