import numpy as np
import pandas as pd 
import os
import h5py
from datetime import datetime
from datetime import timedelta

def Main(ProjName):
    InitialS, Initial =Setup(ProjName)
    BurnInCoef(ProjName)
    ProjectLife(Initial, 730, ProjName, InitialS)
    return

def Setup(ProjName):
    CFC = ['Date','Project Year','Panel Lifetime','Inverter Lifetime','Panel Replacement Year','Peak Sun Hours','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Rental','Total Cost','Cost Check','LCOE']
    df = pd.DataFrame(columns=CFC)
    with h5py.File(ProjName + ".hdf5", "a") as f:

        Inputs = f['Inputs']
        EPC = f['EPC Model']
        Panel = f['Pannel Data']

        Initial = {'Date': StartDate(Inputs.attrs['ModSta']),
        'Project Time': 0,
        'Panel Lifetime': timedelta(weeks=Panel.attrs['Life'] * 52) ,
        'Inverter Lifetime': timedelta(weeks=float(Inputs.attrs['InvLif'] * 52)),
        'Panel Replacement Year' : False,
        'Peak Sun Hours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
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
        'Annual O&M Cost': (1000 * EPC.attrs['PV Size'] * 0.01) / 12,
        'Land Rental': Inputs.attrs['OprCos'] * EPC.attrs['System area'] / 12,
        'Total Cost': EPC['Original Price']['Price excluding panels'] + 1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp'],
        'Cost Check': np.abs((EPC['Original Price']['Price excluding panels'] + 1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp'])/EPC.attrs['PV Size'])/1000,
        'LCOE':0,
        }
    InitialS = pd.Series(Initial)
    df = df.append(InitialS, ignore_index=True)
    return df, Initial

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

def Irr(Date,Prop,ProjName):
    Month = Date.month - 1
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
        Panel.attrs['a'] = a

        b = -dL - 2 * a * ST
        Panel.attrs['b'] = b 

        m = dL
        Panel.attrs['m'] = m 

        c = (1-dL) - m * ST
        Panel.attrs['c'] = c 

    return

def MonthsTODatetime(X):
    X = datetime.strptime(X,'%m')
    return X

def ProjectLife(Initial, TimeRes, ProjName, Data):
    Prev = Initial
    Curr = Initial.copy()
    df = Data
    CFC = ['Date','Project Year','Panel Lifetime','Inverter Lifetime','Panel Replacement Year','Peak Sun Hours','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Rental','Total Cost','Cost Check','LCOE']
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Panel = f['Pannel Data']
        EPC = f['EPC Model']
        PrjLif = Inputs.attrs['PrjLif'] * 365
        PrjEndDate = Initial['Date'] + timedelta(days=float(PrjLif))
        while Curr['Date'] < PrjEndDate:
            Curr['Date'] = Prev['Date'] + timedelta(hours=TimeRes)
            Curr['Project Time'] = np.abs(Initial['Date'] - Curr['Date'])
            Curr['Panel Lifetime'] = Prev['Panel Lifetime'] - Curr['Project Time']
            print(Curr['Panel Lifetime'])
            Curr['Inverter Lifetime'] = Prev['Inverter Lifetime'] - Curr['Project Time']

            if Curr['Panel Lifetime'].days >= 0:
                Curr['Panel Replacement Year'] = False
            else:
                Curr['Panel Replacement Year'] = True
                Curr['Panel Lifetime'] = Initial['Panel Lifetime']

            Curr['Peak Sun Hours'] = Irr(Curr['Date'],'PeakSunHours',ProjName)
            Curr['Cumilative Sun Hours'] = Prev['Cumilative Sun Hours'] + Curr['Peak Sun Hours']
            Curr['Burn in (absolute)'] = Panel.attrs['a'] * Curr['Cumilative Sun Hours'] * Curr['Cumilative Sun Hours'] + Panel.attrs['b'] * Curr['Cumilative Sun Hours'] + 1
            Curr['Long Term Degredation'] = Panel.attrs['m'] * Curr['Cumilative Sun Hours'] + Panel.attrs['c']
            Curr['Long Term Degredation (abs after burn in)'] = Curr['Long Term Degredation'] + float(Panel.attrs['Burn in %, Δd'].strip('%'))

            if Curr['Cumilative Sun Hours'] > Panel.attrs['Burn in peak sun hours, St, hours']:
                Curr['Panel State of Health'] = Curr['Long Term Degredation (abs after burn in)']
            else:
                Curr['Panel State of Health'] = Curr['Burn in (absolute)']
            
            if Curr['Cumilative Sun Hours'] > Panel.attrs['Burn in peak sun hours, St, hours']:
                Curr['Peak Capacity'] = float(Initial['Peak Capacity']) * (1 - float(Panel.attrs['Burn in %, Δd'].strip('%'))) * float(Curr['Panel State of Health'])
            else:
                Curr['Peak Capacity'] = Initial['Peak Capacity'] * Curr['Panel State of Health']

            Curr['Monthly Yeild'] = Irr(Curr['Date'],'Yeild',ProjName)

            if Curr['Date'] >= PrjEndDate:
                Curr['PV Generation'] = 0
            else:
                Curr ['PV Generation'] = Curr['Monthly Yeild'] * np.average([Curr['Peak Capacity'],Prev['Peak Capacity']])
            
            Curr['Capital Cost'] = 0
            
            Curr['Panel Price This Year'] = Panel.attrs['Cost, USD/Wp'] + ((Prev['Panel Price This Year'] - Panel.attrs['Cost, USD/Wp'])*(1 - Inputs.attrs['PanPriDef']/12))

            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Panels - PV)'] = 0
            else:
                if Curr['Panel Replacement Year'] == True:
                    Curr['Refurbishment Cost (Panels - PV)'] = 1000 * Initial['Peak Capacity'] * Curr['Panel Price This Year']
                else:
                    Curr['Panel Replacement Year'] = 0
            
            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Panels - Other)'] = 0
            else:
                if Curr['Panel Replacement Year'] == True:
                    Curr['Refurbishment Cost (Panels - Other)'] = EPC['New Price']['New price'] * np.power((1+Inputs.attrs['InvCosInf']),((Curr['Project Time'].days/365) - 1))
                else:
                    Curr['Panel Replacement Year'] = 0

            Curr['Refurbishment Cost (Panels)'] = Curr['Refurbishment Cost (Panels - Other)']  + Curr['Refurbishment Cost (Panels - PV)']

            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Inverter)'] = 0
            else:
                if Curr['Inverter Lifetime'].days < 0: 
                    Curr['Inverter Lifetime'] = Initial['Inverter Lifetime']
                    Curr['Refurbishment Cost (Inverter)'] = EPC['Original Price']['Inverter cost'] * np.power((1 + Inputs.attrs['InvCosInf']),((Curr['Project Time'].days/365) - 1))
                else:
                    Curr['Refurbishment Cost (Inverter)'] = 0

            if Curr['Date'] >= PrjEndDate:
                Curr['Annual O&M Cost'] = 0
            else:
                Curr['Annual O&M Cost'] = Prev['Annual O&M Cost'] * (1 + Inputs.attrs['OprCosInf']/12)
            
            if Curr['Date'] >= PrjEndDate:
                Curr['Land Retnal'] = 0
            else:
                Curr['Land Rental'] = Prev['Land Rental'] * (1 + Inputs.attrs['OprCosInf']/12)

            if Curr['Date'] >= PrjEndDate:
                Curr['Total Cost'] = 0
            else:
                Curr['Total Cost'] = Curr['Capital Cost'] + Curr['Refurbishment Cost (Panels)'] + Curr['Refurbishment Cost (Inverter)'] + Curr['Annual O&M Cost'] + Curr['Land Rental']
            
            Curr['Cost Check'] = 0

            Curr['LCOE'] = 0

            Prev = Curr
            CurrS = pd.Series(Curr)
            df = df.append(CurrS,ignore_index=True)
        df.to_excel('1.xlsx')
    return

Main('1')