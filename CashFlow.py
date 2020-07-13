import numpy as np
import pandas as pd 
import os
import h5py
from datetime import datetime

def Setup(ProjName):
    CFC = ['Date','Project Year','Months until panel replacement','Months until inverter replacement','Panel Replacement Year','Peak Sun Hours per Month','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Retnal','Total Cost','Cost Check','LCOE']
    df = pd.DataFrame(columns=CFC)
    with h5py.File(ProjName + ".hdf5", "a") as f:

        Inputs = f['Inputs']
        EPC = f['EPC Model']
        Panel = f['Pannel Data']

        Initial = {'Date': StartDate(Inputs.attrs['ModSta']),
        'Project Year': 0,
        'Months until panel replacement': Panel.attrs['Life'] * 12,
        'Months until inverter replacement': Inputs.attrs['InvLif'] * 12,
        'Panel Replacement Year' : False,
        'Peak Sun Hours per Month': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
        'Cumilative Sun Hours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
        'Burn in (absolute)':0,
        'Long Term Degredation':0,
        'Long Term Degredation (abs after burn in)':0,
        'Panel State of Health':100,
        'Peak Capacity': EPC.attrs['PV Size'],
        'Monthly Yeild': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName),
        'PV Generation':0,
        'Capital Cost': EPC['Original Price']['Price excluding panels'] + 1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp'],
        'Refurbishment Cost (Panels - PV)':0,
        'Refurbishment Cost (Panels - Other)':0,
        'Refurbishment Cost (Panels)':0,
        'Panel Price This Year': Panel.attrs['Cost, USD/Wp'],
        'Refurbishment Cost (Inverter)':0,
        'Annual O&M Cost':0,
        'Land Retnal':0,
        'Total Cost': EPC['Original Price']['Price excluding panels'] + 1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp'],
        'Cost Check': np.abs((EPC['Original Price']['Price excluding panels'] + 1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp'])/EPC.attrs['PV Size'])/1000,
        'LCOE':0,
        }
    Initial = pd.Series(Initial)
    df = df.append(Initial, ignore_index=True)
    print(df)
    return


def StartDate(Date):
    DTobj = datetime.strptime(Date, '%d/%m/%y')
    return DTobj
    

def IrrInit(Date,Prop,ProjName):
    DTobj = datetime.strptime(Date, '%d/%m/%y')
    Month = DTobj.month - 1
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Irr = f['Irradiance']
        P = np.asarray(Irr[Prop])
    return P[Month]

def BurnInCoef(ProjName):
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Panel = f['Pannel Data']
        Irr = f['Irradiance']
        DeltaD = Panel.attrs['Burn in %, Δd']
        DeltaD = DeltaD.replace("%", "")
        DeltaD = float(DeltaD)

        dL = Panel.attrs['Long-term Degradation, dL, %/year']
        dL = dL.replace("%", "")
        dL = float(dL)
        dL = -dL /np.sum(Irr['PeakSunHours'])
        
        ST = Panel.attrs['Burn in peak sun hours, St, hours']

        a = (DeltaD - dL * ST)/(np.power(ST,2))
        b = -dL - 2 * a * ST
        m = dL
        c = (1-dL) - m * ST

        print(a)
        print(b)
        print(m)
        print(c)
    

    return

BurnInCoef('1')