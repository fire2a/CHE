#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:38:50 2023

@author: rodrigo
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from typing import Tuple, List
import os
from warnings import warn

def obs2polygon(st_grilla, sf_species, output_fn):
    especies = sf_species['species'].unique()
    for i, s in enumerate(especies):
        print(f"{i + 1}/{len(especies)}  process completed    working on species: {s}")
        obs = sf_species[sf_species['species'] == s]
        mat = np.column_stack((obs['geometry'].x, obs['geometry'].y))
        mat = MultiPoint(mat)
        obs = mat.intersection(st_grilla)
        st_grilla = st_grilla.difference(obs)
    precenses = st_grilla.drop(columns=['geometry'])
    precenses = precenses.astype(np.float32)
    precenses.to_csv(output_fn, sep=' ')


def geobox(corners : Tuple[float, float, float, float]) -> gpd.GeoSeries: 
    xmin, ymin, xmax, ymax = corners
    
    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    
    pol = Polygon(box)
    
    return gpd.GeoSeries(pol) 

def contains(grid: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> pd.DataFrame: 
    bool_df = (points.
               geometry.
               apply(grid.contains).
               any().
               apply(int))
    return bool_df
    
def obs_to_grid(grid: gpd.GeoDataFrame, obs: gpd.GeoDataFrame, min_obs : int = 30,
                write: bool = False, out_fn: str = 'grid_container.parquet') -> pd.DataFrame: 
    species = obs['species'].unique()
    for s in tqdm(species, desc = 'Species\' observation grided', unit = '', ncols = 100):        
        # Select observations of a specific species 
        aux = obs.query('species == @s')
        grid[s] = contains(grid, aux)
    
    warn(f'Species with insuficient cells ocupied (<{min_obs}) were deleted.')
    id = grid['id']
    grid = grid.loc[:, grid.columns.intersection(species)] 
    grid = grid.loc[:, (grid.sum(axis=0) >= min_obs)]
    grid = pd.concat([id, grid], axis = 1 )
    
    if write:
        grid.to_parquet(out_fn)
    return grid

def main(): 
    
    
if __name__ == '__main__': 
    main()
        