#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:11:19 2023

@author: rodrigo
"""
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull
import numpy as np
import seaborn as sns
from itertools import combinations
import scienceplots
from functools import reduce
import pandas as pd

plt.rcParams['font.family'] = ['serif']


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
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

def plot_ndimensional_convexhull(che: object): 
    vertices = che.vertices
    variables = che.data.columns
    bool_map = [(var not in MODELS) for var in variables]

    variables = variables[bool_map]
    data = che.data.loc[:, variables]

    dimensions = len(variables)
    color = sns.color_palette('rocket', 1)

    plt.style.use(['science'])
    fig = plt.figure(1, figsize = (dimensions * 3, dimensions * 3))
    gs = gridspec.GridSpec(dimensions - 1, dimensions - 1)
    gs.update(hspace = 0.2, wspace = 0.2)
    
    for variable_1, variable_2 in combinations(data.columns, 2):
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


def plot3D_convexhulls(che_list: list[object], variable_list: list[str] = None) -> None: 
    
    all_species = [che.species for che in che_list]
    all_species = [val for sublist in all_species for val in sublist]

    colors = sns.color_palette('rocket', len(all_species))


    # reducir ancho de lineas
    fig = plt.figure(figsize = (6, 6)) # For plotting
    ax = plt.axes(projection="3d") 
    ls = LightSource(azdeg=0,altdeg=65)

    variables = [che.data.columns for che in che_list]
    variables = reduce(np.intersect1d, variables)

    remove_models = [(var not in MODELS) for var in variables]
    variables = variables[remove_models]

    remove_species = [(var not in all_species) for var in variables]
    variables = variables[remove_species]  



    if len(variables) > 3: 
        if variable_list != None: 
            variables = np.intersect1d(variable_list, variables)
        else: 
            raise AttributeError('A list of three common variables must be provided (variable_list).')
    
    

    for i, che in enumerate(che_list):
        points = che.points
        data = che.data

        #reducir tamaño de las aristas
        #aggregar fuente de luz: https://stackoverflow.com/questions/56864378/how-to-light-and-shade-a-poly3dcollection
        for simplex in che.convex_hull.simplices:
            tri = Poly3DCollection([points[simplex]])
            tri.set_color(colors[i])
            tri.set_alpha(0.3)
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
            ax.set_label(che.species)
            # ax.imshow(rgb)
    
        for j, sp in enumerate(che.species, start = i): 
            columns = np.append(variables.copy(), sp)
            query = f'{sp} == 1'
            sp_data = data.loc[:, columns]
            sp_data = data.query(query)
            x, y, z = sp_data.loc[:, variables].to_numpy().T
            ax.scatter3D(x, y, z, s = 2, c = colors[j])

    # agregar nombres de los ejes
    x, y, z = variables
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    # Set grid's linewidth
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.3

    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0

    # Set background color
    ax.w_xaxis.set_pane_color('w')
    ax.w_yaxis.set_pane_color('w')
    ax.w_zaxis.set_pane_color('w')

    print(ax.zaxis._axinfo)


    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    plt.show()


def plot_Ellips(ellips): 
    return None

