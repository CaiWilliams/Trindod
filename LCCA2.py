from Trindod import *
from Ryfeddod import *
import pandas as pd

def RunQue(L, LCOE, Jobs):
    for J in L.Q.Jobs:
        print(J)
        O = LCOE.WorkerNonMP(L,J)
        Data = pd.DataFrame()
        Data['Settlement Date'] = O.Panel.Dates
        Data['Settlement Date'] = [x.replace(tzinfo=pytz.UTC) for x in Data['Settlement Date']]
        Data['Generation'] = O.Panel.PVGen / 1000
        Data = Data.set_index('Settlement Date')

    return

def Original_Emissions(File, Device, Lat, Lon):
    NG = Setup(File, Device, Lat, Lon)
    NG = Scaling(NG, 1, 1, 0, 0)
    NG = Expand_Generation(NG, 20)
    NG = Expand_Sacler(NG, 20)
    for Asset in NG.
    return

def New_Emissions():
    return


Filename = 'ResultSets/LCCA/Location'
PanelData = 'Data/PanelDataNonGA2.csv'
L = LCOE(Filename,PanelData)
LCOE.GenerateJBS(L)
LCOE.LoadJBS(L)

