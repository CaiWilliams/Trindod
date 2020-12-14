import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.lib.npyio import loadtxt
import calendar

fig, ax = plt.subplots()
locations = ['Stockholm','Warsaw','Varna','Beirut','Medina','Khartoum','Juba','Dodoma','Lusaka','Gaborone','PortElizabeth']
Latitude = ['59.321981','52.238258','43.224682','33.877827','24.58894','15.537885','4.877593','-5.969234','-15.447529','-24.580186','-33.964914']
i = 10
while True:
    j = 0
    for Loc in locations:
        print(Loc)
        Data = np.loadtxt("AvHr/"+Loc+".csv",delimiter=',')
        X = Data[0][1:]
        Y = Data[1:,0]
        Z = Data[1:,1:]


        X,Y = np.meshgrid(np.arange(len(X)),np.arange(len(Y)))
        surf = ax.pcolormesh(X, Y, Z*100)
        plt.title("Location: " + Loc +" Latitude: " + Latitude[j])
        plt.xlabel("Hours")
        plt.ylabel("Months")
        plt.yticks(np.arange(12),calendar.month_name[1:13],)
        plt.xticks(np.arange(24),['00:00','01:00','02:00','03:00','04,00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],rotation=90)
        cb = plt.colorbar(surf)
        plt.draw()
        plt.pause(0.5)
        cb.remove()
        plt.cla()
        j = j + 1
    i = i - 1