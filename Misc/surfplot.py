from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import numpy.ma as ma
import matplotlib.cm as cm
from scipy.interpolate import griddata
from scipy.interpolate.ndgriddata import NearestNDInterpolator

data = pd.read_csv('Results.csv')
x = data['Longitude'].to_numpy()
y = data['Latitude'].to_numpy()
z = data['LCOE'].to_numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z)
plt.show()