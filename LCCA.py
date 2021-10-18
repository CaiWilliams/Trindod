import numpy as np

from Trindod import *
from Ryfeddod import *
import pandas as pd

Filename = 'ResultSets/LCCA/Location'
PanelData = 'Data/PanelDataNonGA2.csv'
L = LCOE(Filename,PanelData)
LCOE.GenerateJBS(L)
LCOE.LoadJBS(L)
#print(L.Q.Jobs[0])
O = LCOE.WorkerNonMP(L, L.Q.Jobs[0])
Data = pd.DataFrame()
Data['Settlement Date'] = O.Panel.Dates
Data['Settlement Date'] = [x.replace(tzinfo=pytz.UTC) for x in Data['Settlement Date']]
Data['Generation'] = O.Panel.PVGen / 1000
Data = Data.set_index('Settlement Date')
#plt.plot(Data)
#plt.show()


NG1 = Setup('Data/2015RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
NG1 = Scaling(NG1, 1, 1, 0, 0)
NG1 = Expand_Generation(NG1, 20)
NG1 = Expand_Sacler(NG1, 20)
for Asset in NG1.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        NG1.Mix['Technologies'].remove(Asset)
NG1 = Grid.Demand(NG1)
DNG1 = Dispatch(NG1)
x = DNG1.CarbonEmissions

NG2 = Setup('Data/2015RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
NG2 = Scaling(NG2, 1, 1, 1, 0)
NG2 = Expand_Generation(NG2, 20)
NG2 = Expand_Sacler(NG2, 20)
for Asset in NG2.Mix['Technologies']:
    if Asset['Technology'] == 'SolarBTMNT':
        NG2.Mix['Technologies'].remove(Asset)
NG2.EndDate = NG2.EndDate.replace(year=2015+20)
NG2 = Grid.Demand(NG2)
NG2 = Add_to_SolarNT(NG2, Data)
NG2 = Grid.MatchDates(NG2)
DNG2 = Dispatch(NG2)
y = DNG2.CarbonEmissions
Y = (x-y) /2 * (1*10**-3)
#plt.plot(np.cumsum(Y))
#plt.xlim(left=datetime(year=2016, month=1,day=1), right=datetime(year=2017, month=1,day=1))
#plt.plot(np.cumsum(y))

plt.show()

def StaticLCOE(TotalCosts, E1, E0):
    tc = np.cumsum(TotalCosts)[-1]
    E0 = np.cumsum(E0)[-1] /2 * (1*10**-3)
    E1 = np.cumsum(E1)[-1] /2 * (1*10**-3)
    LCOE = tc/(E0 - E1)
    return LCOE

def LCOECalculate(Dates1, Dates2, TotalCosts, PVGen, InterestDivisor, NewPrice, DCR):
    i1 = np.linspace(0, len(Dates1), len(Dates1))
    i2 = np.linspace(0, len(Dates2), len(Dates2))
    tc = TotalCosts[:]
    pv = PVGen[:]
    ii1 = i1[:] / InterestDivisor
    ii2 = i2[:] / InterestDivisor
    #print(xnpv(DCR, tc[:], ii1[:]))
    #print(xnpv(DCR, pv[:], ii2[:]))
    print(NewPrice)
    LCOE = (NewPrice + np.abs(xnpv(DCR, tc[:], ii1[:]))) / xnpv(DCR, pv[:], ii2[:])
    return LCOE

def xnpv(dcr, values, date):
    V = np.sum(values[:] / (1.0 + dcr) ** (date[:]))
    return V

LCCA = LCOECalculate(O.Panel.Dates, DNG2.CarbonEmissions.index.to_numpy(), O.Finance.TotalCosts, Y, O.Finance.InterestDivisor, O.Finance.NewPrice, O.Finance.DCR)
print(LCCA)
print("Total Costs", np.cumsum(O.Finance.TotalCosts)[-1])
print("Area", O.Finance.NewArea)

#LCCA_S = StaticLCOE(O.Finance.TotalCosts, y, x)
#print(LCCA_S)