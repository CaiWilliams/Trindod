import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

fig, ax = plt.subplots()

irr = pd.read_hdf("d.hdf5",key="Irradiance")
A = irr.to_numpy()
zmax = np.max(A[:,1])
zmin = np.min(A[:,1])
Days = np.unique(A[:,6])
Hours = ['0011','0111','0211','0311','0411','0511','0611','0711','0811','0911','1011','1111','1211','1311','1411','1511','1611','1711','1811','1911','2011','2111','2211','2311']
arr = np.zeros((len(np.unique(A[:,6])),len(Hours)))
j = 0
for Hour in A[:,0]:
    for Value in Hours:
        if Hour == Value:
            arr[np.where(Days == A[j,6]),Hours.index(Hour)] = A[j,1]
    j = j + 1
#print(arr)

c = ax.pcolor(arr[:])
fig.colorbar(c,ax=ax)
plt.savefig('foo.png')