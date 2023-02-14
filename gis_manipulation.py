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
from shapely.geometry import Polygon
from typing import Tuple
import os
from warnings import warn


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
    species = (obs.groupby('species').count() > min_obs).index
    species = list(species)
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
        grid.to_parquet(IN_FOLDER, out_fn)
    return grid

def main(): 
    IN_FOLDER = os.path.join('data', 'Valpo')
    
    mammals_fn = os.path.join(IN_FOLDER, 'mammals.shp')
    mammals = gpd.read_file(mammals_fn)[['spp', 'geometry']]
    
    birds_fn = os.path.join(IN_FOLDER, 'birds.shp')
    birds = gpd.read_file(birds_fn)[['spp', 'geometry']]
    
    obs = pd.concat([mammals, birds], ignore_index=True)
    obs = obs.rename(columns = {'spp': 'species'})
    del birds, mammals, birds_fn, mammals_fn
    
    grid_fn =  os.path.join(IN_FOLDER, 'grid.shp')      
    grid = gpd.read_file(grid_fn)[['id','geometry']].head(10000)
    
    grid = obs_to_grid(grid, obs, write = True)
    
    
if __name__ == '__main__': 
    main()
        