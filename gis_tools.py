#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:38:50 2023

@author: rodrigo
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rst
from tqdm import tqdm
from shapely.geometry import Polygon
from numpy.typing import ArrayLike
from typing import Tuple
import os
from warnings import warn
import matplotlib.pyplot as plt


def geobox(corners : Tuple[float, float, float, float]) -> gpd.GeoSeries: 
    xmin, ymin, xmax, ymax = corners
    
    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    
    pol = Polygon(box)
    
    return gpd.GeoSeries(pol) 

def contains(grid: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> pd.DataFrame: 
    bool_df = (points.
               geometry.
               apply(grid.contains).
               any().apply(int))
    return bool_df

def obs_to_grid(grid: gpd.GeoDataFrame, obs: gpd.GeoDataFrame, min_obs : int = 30,
                write: bool = False, out_fn: str = 'grid_container.parquet') -> pd.DataFrame: 
    
    species = (obs.groupby('species').size() > min_obs).index.tolist()
    for s in tqdm(species, desc = 'Species\' observation grided', unit = '', ncols = 100):        
        # Select observations of a specific species 
        aux = obs.query('species == @s')
        grid[s] = contains(grid, aux)
    
    num_species = len(species)
    id = grid['id']
    grid = grid.loc[:, grid.columns.intersection(species)] 
    grid = grid.loc[:, (grid.sum(axis=0) >= min_obs)]
    grid = pd.concat([id, grid], axis = 1 )
    
    deleted_species = num_species - len(grid.columns)
    if deleted_species:
        warn(f'{deleted_species} species with insufficient cells occupied (<{min_obs}) were deleted.')

    if write:
        grid.to_parquet(out_fn)
    return grid

def raster_to_dataframe(rast_fn: rst.DatasetReader | str) -> pd.DataFrame:
    if isinstance(rast_fn, rst.DatasetReader): 
        pass
    elif os.path.isfile(rast_fn): 
        rast = rst.open(rast_fn, 'r')
        
    columns = list(rast.descriptions)
    nodata = rast.nodata
    
    if columns[0] == None:
        base_name = os.path.basename(rast_fn)
        base_name, _ = os.path.splitext(base_name)
        columns[0] = base_name
        
    data = np.vstack([rast.read(i+1).flatten() for i in tqdm(range(len(columns)))])
    rast.close()
    
    df = pd.DataFrame(data = data.T, columns = columns)
    df = df.replace(nodata, np.nan)
    
    return df
    

def main(): 
    IN_FOLDER = os.path.join('data', 'Valpo')
    
    region_fn = os.path.join(IN_FOLDER, 'shape.shp')
    region = gpd.read_file(region_fn)['geometry']
    
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
