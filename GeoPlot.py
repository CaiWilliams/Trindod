import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
from mpl_toolkits.basemap import Basemap, maskoceans

#m = Basemap(projection='laea', resolution='i', width=9E6, height=12E6, lat_0=10, lat_1=0, lat_2=0, lon_0=10, lon_1=0, lon_2=0)
m = Basemap(projection='merc',llcrnrlat=-35,urcrnrlat=65,llcrnrlon=-20,urcrnrlon=60,lat_ts=20,resolution='i')

plt.rcParams["figure.figsize"] = (5,4)
plt.rcParams['figure.dpi'] = 300

Data = pd.read_csv('Results.csv',index_col=None)

x = Data['Job.Longitude'].to_numpy()
y = Data['Job.Latitude'].to_numpy()
z = Data['Panel.WindSpeed'].to_numpy()
xx,yy = np.meshgrid(x,y)

X = np.linspace(-20,60,500)
Y = np.linspace(-35,65,500)
XX,YY = np.meshgrid(X,Y)

#Z = interpolate.griddata((x,y),z,(X,Y),method='cubic')
interp = interpolate.Rbf(x,y,z, smooth=1)
#interp = interpolate.CloughTocher2DInterpolator(list(zip(x, y)), z)
Z = interp(XX,YY)
#m.scatter(x,y, latlon=True)
#m.shadedrelief()
m.drawcoastlines(linewidth=1)
#m.drawmapboundary(fill_color='black')
data = maskoceans(XX,YY,Z)
m.pcolormesh(XX, YY, data, latlon=True, cmap="inferno_r")
#plt.clim(-16,16)
#m.drawcoastlines()
#m.drawmapboundary(fill_color='black',zorder=10)
#m.scatter(x,y, latlon=True)
plt.tight_layout()
plt.colorbar(label="Mean Wind Speed (ms$^{-1}$)")
plt.savefig("GeoMeanWind.png",transparent=True)
plt.show()