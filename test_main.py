#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:51:46 2023

@author: rodrigo
"""

import pytest 
import os
import pandas as pd 
from main import CHE, intersection
from numpy.typing import ArrayLike 

IN_FOLDER = 'data'
d_fn = os.path.join(IN_FOLDER, 'D.csv')
b_bar = pd.read_csv(d_fn, usecols = ['Bbar', 'Eisa'])

pcs_fn = os.path.join(IN_FOLDER, 'TablaPCs.csv')
# Visual Python: Data Analysis > File
pcs = pd.read_csv(pcs_fn, sep = ' ', 
                  usecols = ['PC1', 'PC2', 'PC3', 'PC4'])

# convert dataframe to arrays
data = pd.concat([b_bar, pcs], axis=1)


@pytest.mark.parametrize(
    'data1, data2, p, expected', 
    [
     
     (
      data.query('Bbar == 1')[['PC1', 'PC2']].sample(100),
      data.query('Bbar == 1')[['PC1', 'PC2']].sample(100),
      (50 + i/2)/100, 
      True
      ) 
     
     for i in range(100) 
     ]
    
    )
def test_intersection(data1: ArrayLike | pd.DataFrame, 
data2: ArrayLike | pd.DataFrame, p: float, expected: bool): 
    che1 = CHE(data1)
    che1.cov(p)
    
    che2 = CHE(data2)
    che2.cov(p)
    
    assert (che1.volume != che2.volume) == expected
    
    che3 = intersection([che1, che2])
    
    assert (che3.volume <= che1.volume) == expected
    assert (che3.volume <= che2.volume) == expected
    
    # che1.mcd(p)
    
    # che2.mcd(p)
    
    # assert (che1.volume != che2.volume) == expected
    
    # che3 = intersection([che1, che2])
    
    # assert (che3.volume <= che1.volume) == expected
    # assert (che3.volume <= che2.volume) == expected

@pytest.mark.parametrize(
    'data, expected', 
    [
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3']], 58.79), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 110.55),
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3']], 30.99), 
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 44.87)
     ])
def test_cov(data: pd.DataFrame, expected: float): 
    che = CHE(data)
    che.cov()
    assert round(che.volume,2) == expected
    
@pytest.mark.parametrize(
    'data, expected', 
    [
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3']], 54.06), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 64.87),
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3']], 22.29), 
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 20.96)
     ])
def test_mcd(data: pd.DataFrame, expected: float): 
    che = CHE(data)
    che.mcd()
    assert che.volume == expected
