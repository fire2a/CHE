#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:11:19 2023

@author: rodrigo
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import seaborn as sns
from itertools import combinations
import scienceplots
from functools import reduce

MODELS = ['COV', 'MCD', 'MVE', 'MVEE']

def add_headers(
    fig,
    columns,
    *,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()
    col_headers = columns[1:]
    row_headers = columns[:-1]

    for ax in axes:
        sbs = ax.get_subplotspec()
        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and (sbs.colspan.start == sbs.rowspan.start):
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * -90,
                **text_kwargs,
            )

def plot_ndimensional_convexhull(che: object): 
    vertices = che.vertices
    data = che.data.loc[:, che.data.columns not in MODELS]
    dimensions = len(data.columns)
    color = sns.color_palette('rocket', 1)

    plt.style.use(['science'])
    fig = plt.figure(1, figsize = (dimensions * 3, dimensions * 3))
    gs = gridspec.GridSpec(dimensions - 1, dimensions - 1)
    gs.update(hspace = 0.2, wspace = 0.2)
    
    for variable_1, variable_2 in combinations(data.columns, 2):
        print(variable_1, variable_2) 
        i = data.columns.get_loc(variable_1)
        j = data.columns.get_loc(variable_2)
        if i < len(data.columns): 
            subplot = fig.add_subplot(gs[i, j-1])
            x = data[variable_1]
            y = data[variable_2]
            plt.scatter(x, y, c = color, s = 8)
            
            points = vertices.T[[i,j]].T

            convex_hull = ConvexHull(points)

            for simplex in convex_hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k')
            plt.fill(points[convex_hull.vertices,0], 
                    points[convex_hull.vertices,1], 
                    color, 
                    alpha = 0.3)

    add_headers(fig, data.columns) 
    plt.show()


def plot3D_convexhulls(che_list: list[object]) -> None: 
    colors = sns.color_palette('rocket', len(che_list))

    fig = plt.figure()
    ax = fig.add_subplot( projection="3d")

    variables = [che.data.columns for che in che_list]
    variables = reduce(np.intersect1d, variables)
    variables = variables[variables not in MODELS]

    if len(variables) >= 3: 
        pass

    all_points = pd.DataFrame(columns = variables)
    for hull in che_list:
        vertices = hull.vertices
        all_points = pd.concat(all_points, hull.data.loc[:, variables])

        # Plot defining corner points
        ax.plot(vertices, "ko")

        # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

        # Make axis label
        for i in ["x", "y", "z"]:
            eval("ax.set_{:s}label('{:s}')".format(i, i))

    
    plt.show()




