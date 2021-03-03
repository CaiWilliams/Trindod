import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Dir = "Generations/"
Files = os.listdir(Dir)


Results0 = np.zeros(0)
Results1 = np.zeros(0)
Results2 = np.zeros(0)
Results3 = np.zeros(0)
Results4 = np.zeros(0)
Results5 = np.zeros(0)
for file in Files:
    print(file)
    GeneratinoResults = pd.read_csv(Dir+file)
    MinResults0 = np.average(GeneratinoResults['0'].to_numpy())
    MinResults1 = np.average(GeneratinoResults['1'].to_numpy())
    MinResults2 = np.average(GeneratinoResults['2'].to_numpy())
    MinResults3 = np.average(GeneratinoResults['3'].to_numpy())
    MinResults4 = np.average(GeneratinoResults['4'].to_numpy())
    MinResults5 = np.average(GeneratinoResults['Results'].to_numpy()[np.where(GeneratinoResults['Results'].to_numpy() != np.inf)])
    Results0 = np.append(Results0, MinResults0)
    Results1= np.append(Results1, MinResults1)
    Results2 = np.append(Results2, MinResults2)
    Results3 = np.append(Results3, MinResults3)
    Results4 = np.append(Results4, MinResults4)
    Results5 = np.append(Results5, MinResults5)
#plt.plot(Results0,label='life')
#plt.plot(Results1,label='Burn-in')
#plt.plot(Results2,label='Long-termDegradation')
#plt.plot(Results3,label='Burn-inPeakSunHours')
#plt.plot(Results4,label='PowerDensity')
plt.plot(Results5,label='LCOE')
plt.legend()
plt.show()
