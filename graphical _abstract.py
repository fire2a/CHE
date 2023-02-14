#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:04:59 2023

@author: rodrigo
"""

from sklearn.covariance import EllipticEnvelope
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots

from scipy.spatial import ConvexHull
import os

plt.style.use(['science',
               # 'ieee'
               ])
# mpl.rcParams['pdf.fonttype'] = 42 
mpl.rcParams['font.size'] = 20
# Set the axes title font size
plt.rc('axes', titlesize=20)
# Set the axes labels font size
plt.rc('axes', labelsize=20)

# csfont = {'fontname':'Times New Roman'}
# plt.rcParams["font.family"] = "Times New Roman"

actual_wd = os.getcwd()
os.chdir(actual_wd)

# Visual Python: Data Analysis > File
b_bar = pd.read_csv('D.csv', usecols = ['Bbar'])

# Visual Python: Data Analysis > File
pcs = pd.read_csv('TablaPCs.csv', sep = ' ', 
                  usecols = ['PC1', 'PC2'])

# convert dataframe to arrays
data = pd.concat([b_bar, pcs], axis=1)
data = data.query('Bbar == 1')
# data.plot(kind = 'scatter', 
#          x = 'PC1', 
#          y = 'PC2')

# instantiate model
model1 = EllipticEnvelope(contamination = 0.05) 
# fit model
points = data[['PC1', 'PC2']].to_numpy()
model1.fit(points)
hull = ConvexHull(points)

# predict on new data 
pred1 = model1.predict(points)

data['EE_pred'] = pred1

new_points = (
    data.query('EE_pred == 1')[['PC1', 'PC2']]
    .to_numpy()
)

new_hull = ConvexHull(new_points)

custm_map = mpl.colors.ListedColormap(['white', 'white', '#20639b'])

xx1, yy1 = np.meshgrid(np.linspace(-6, 3, 500), np.linspace(-2.5, 4, 500))


fig = plt.figure(1, figsize = (9,18))
gs = gridspec.GridSpec(3, 1)
gs.update(hspace = 0.2)

xtr_subplot = fig.add_subplot(gs[0])

Z1 = model1.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
Z1 = Z1.reshape(xx1.shape)
plt.contourf(xx1, yy1, Z1, 1, cmap = custm_map,  alpha = 0.3)
plt.contour(
        xx1, yy1, Z1, levels=[0], linewidths=2, colors='k'
)

plt.scatter(data.PC1.to_numpy(), data.PC2.to_numpy(), c = 'black', s = 16)

plt.title("Elliptic Envelope (EE)")
plt.tick_params(labelbottom = False)
bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate(
    "outlying points",
    xy=(2.2, -0.7),
    xycoords="data",
    textcoords="data",
    xytext=(0.8, -1.8),
    bbox=bbox_args,
    arrowprops=arrow_args,
)


plt.tick_params(labelbottom = False)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
# plt.ylabel("PC2")
# plt.xlabel("PC1")
# plt.show()


xtr_subplot = fig.add_subplot(gs[1])
#plt.plot(new_points[:,0], new_points[:,1], 'o')
for simplex in new_hull.simplices:
    plt.plot(new_points[simplex, 0], new_points[simplex, 1], 'k')
plt.fill(new_points[new_hull.vertices,0], 
         new_points[new_hull.vertices,1], 
         '#3ca3a3', 
         alpha = 0.3)

plt.scatter(data.PC1.to_numpy(), data.PC2.to_numpy(), c = 'k', s = 16)

plt.title("Combination of EE and CH (CHE-approach)")

bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="->")
plt.annotate(
    "outlying points",
    xy=(2.2, -0.7),
    xycoords="data",
    textcoords="data",
    xytext=(0.8, -1.8),
    bbox=bbox_args,
    arrowprops=arrow_args,
)

plt.tick_params(labelbottom = False)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.ylabel("PC2")



#plt.plot(points[:,0], points[:,1], 'o')
xtr_subplot = fig.add_subplot(gs[2])
for simplex in hull.simplices:
    plt.plot(
        points[simplex, 0], 
        points[simplex, 1], 
        'k',
    )
plt.fill(
    points[hull.vertices,0], 
    points[hull.vertices,1], 
    '#f6d55c',
    alpha = 0.3
)
plt.scatter(data.PC1.to_numpy(), data.PC2.to_numpy(), c = 'black', s = 16)

plt.title("Convex Hull (CH)")


plt.tick_params(labelbottom = True, direction = 'in')

plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
# plt.ylabel("PC2")
plt.xlabel("PC1")
plt.savefig('all_together.png', dpi=300)





