import pytest 
import os
import pandas as pd 
from plot import * 
from main import CHE


IN_FOLDER = 'data'
d_fn = os.path.join(IN_FOLDER, 'D.csv')
b_bar = pd.read_csv(d_fn, usecols = ['Bbar'])

pcs_fn = os.path.join(IN_FOLDER, 'TablaPCs.csv')
# Visual Python: Data Analysis > File
pcs = pd.read_csv(pcs_fn, sep = ' ', 
                    usecols = ['PC1', 'PC2', 'PC3', 'PC4'])

# convert dataframe to arrays
data = pd.concat([b_bar, pcs], axis=1)
data = data.query('Bbar == 1')


def test_che_plot(): 
    points = data.loc[:, data.columns != 'Bbar']
    che = CHE(points)
    che.cov()
    che.plot()

