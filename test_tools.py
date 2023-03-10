import pytest 
import os
import pandas as pd 
from tools import EllipticEnvelope
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
    'data, model, expected', 
    [
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3']], 'COV', 132.8), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 'COV', 592.24),
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3']], 'COV', 109.2), 
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 'COV', 513.09), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3']], 'MCD', 119.03), 
     (data.query('Bbar == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 'MCD', 349.94),
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3']], 'MCD', 70.71), 
     (data.query('Eisa == 1')[['PC1', 'PC2', 'PC3', 'PC4']], 'MCD', 167.09)
     ])
def test_EllipticEnvelope(data: ArrayLike | pd.DataFrame, model: str, expected: float): 
    volume = EllipticEnvelope(data, model).volume
    volume = round(volume, 2)
    
    assert volume == expected
