import pytest 
import os
import pandas as pd 
from tools import * 
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
    'data, expected', 
    [
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3']], 132.8), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 592.24),
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3']], 109.2), 
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 513.09)
     ])
def test_cov(data: ArrayLike | pd.DataFrame, expected: float): 
    A, c = cov(data)
    volume = round (vol_ellipsoid(A), 2)
    
    assert volume == expected

@pytest.mark.parametrize(
    'data, expected', 
    [
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3']], 119.03), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 349.94),
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3']], 70.71), 
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 167.09)
     ])
def test_mcd(data: ArrayLike | pd.DataFrame, expected: float): 
    A, c = mcd(data)
    volume = round (vol_ellipsoid(A), 2)
    
    assert volume == expected
