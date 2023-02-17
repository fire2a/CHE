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

@pytest.mark.parametrize(
    'data1, data2, p, expected', 
    [
     
     (
      data[['PC1', 'PC2']].sample(100, random_state=1),
      data[['PC1', 'PC2']].sample(100, random_state=2),
      (50 + i/2)/100, 
      True
      ) 
     
     for i in range(100) 
     ]
    
    )
def test_intersection(data1, data2, p, expected): 
    che1 = CHE()
    che1.cov(data1, p)
    
    che2 = CHE()
    che2.cov(data2, p)
    
    assert (che1.volume != che2.volume) == expected
    
    che3 = intersection(che1, che2)
    
    assert (che3.volume <= che1.volume) == expected
    assert (che3.volume <= che2.volume) == expected

