import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
from mpl_toolkits.basemap import Basemap, maskoceans

#m = Basemap(projection='laea', resolution='i', width=9E6, height=12E6, lat_0=10, lat_1=0, lat_2=0, lon_0=10, lon_1=0, lon_2=0)
m = Basemap(projection='merc',llcrnrlat=-35,urcrnrlat=60,llcrnrlon=-120,urcrnrlon=98,lat_ts=20,resolution='i')

plt.rcParams["figure.figsize"] = (6,3)
plt.rcParams['figure.dpi'] = 300

Data = pd.read_csv('MultiplePoint.csv',index_col=None)

x = Data['Job.Longitude'].to_numpy()
y = Data['Job.Latitude'].to_numpy()
z = Data['Finance.LCOE'].to_numpy()
txt = ['ZAF','MEX','IND','ESP','BRA','GBR','USA']

#Z = interpolate.griddata((x,y),z,(X,Y),method='cubic')
#m.scatter(x,y, latlon=True)
#m.drawcoastlines(linewidth=0.1)
m.shadedrelief()
#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='green',zorder=1)
#m.drawmapboundary(fill_color='black')
m.scatter(x, y, c=z, cmap='inferno_r', latlon=True,zorder=2)
for i, txt in enumerate(txt):
    plt.annotate(txt, m(x[i], y[i]),m(x[i]+1, y[i]+2))
#plt.clim(-16,16)
#m.drawcoastlines()
#m.drawmapboundary(fill_color='black',zorder=10)
#m.scatter(x,y, latlon=True)
plt.tight_layout()
plt.colorbar(label="Levelised Cost of Energy ($/kWh)")
plt.show()