import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np

def ticks(dt):
    return (dt - datetime(2015, 1, 1)).total_seconds()/3600

fig, ax = plt.subplots(dpi=300)

data = pd.read_csv('SinglePointPowerRoll-4107.csv')
x = data['Date']
x = pd.to_datetime(x,format='%d/%m/%Y %H:%M').to_numpy()
y = data['Peak Capacity'].to_numpy()
ax.plot(x,y/1000)
ax.set_xlim(left=datetime(year=2015,month=1,day=1), right=datetime(year=2020,month=1,day=1))
ax.set_ylabel("Farm Capacity (MW)")

X0 = 0.5
XD = 0.4
#XD = X0 + XD
Y0 = 0.5
YD = 0.4
#YD = Y0 + YD
axins = ax.inset_axes([X0,Y0,XD,YD])
axins.plot(x,y/1000)
axins.set_xlim(left=datetime(year=2017,month=1,day=1),right=datetime(year=2017,month=3,day=1))
axins.set_ylim(3.38,3.4)
axins.set_yticks([3.38,3.40])
axins.set_xticks([datetime(year=2017,month=1,day=1),datetime(year=2017,month=3,day=1)])

ax.indicate_inset_zoom(axins, edgecolor='black')
plt.show()