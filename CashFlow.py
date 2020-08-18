import numpy as np
import pandas as pd 
import os
import h5py
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *

#Main Run order of CashFlow model
def Cashflow(ProjName):
    BurnInCoef(ProjName)
    InitialS, Initial = Setup(ProjName)
    ProjectLife(Initial, 1, ProjName, InitialS)
    return

#Setsup cash flow model with initial data
def Setup(ProjName):
    CFC = ['Date','Project Year','Panel Lifetime','Inverter Lifetime','Panel Replacement Year','Peak Sun Hours','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Rental','Total Cost','Cost Check','LCOE']
    df = pd.DataFrame(columns=CFC)
    with h5py.File(ProjName + ".hdf5", "a") as f:

        Inputs = f['Inputs']
        EPC = f['EPC Model']
        Panel = f['Pannel Data']

        Initial = {'Date': StartDate(Inputs.attrs['ModSta']),
        'Project Time': timedelta(days=0),
        'Panel Lifetime': timedelta(weeks=float(Panel.attrs['Life']) * 52) ,
        'Inverter Lifetime': timedelta(weeks=float(Inputs.attrs['InvLif'] * 52)),
        'Panel Replacement Year' : False,
        'Peak Sun Hours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
        'Cumilative Sun Hours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
        'Burn in (absolute)':(Panel.attrs['a'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) + 1),
        'Long Term Degredation':(Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c'],
        'Long Term Degredation (abs after burn in)':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01),
        'Panel State of Health':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01),
        'Peak Capacity': EPC.attrs['PV Size'] * (1 - (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)),
        'Monthly Yeild': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName),
        'PV Generation': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName) * ((EPC.attrs['PV Size'])+(EPC.attrs['PV Size'] * (1 - (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01))))/2,
        'Capital Cost': 0,#EPC['New Price']['Installation cost exc. panels'] + (1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp']),
        'Refurbishment Cost (Panels - PV)':0,
        'Refurbishment Cost (Panels - Other)':0,
        'Refurbishment Cost (Panels)':0,
        'Panel Price This Year': Panel.attrs['Cost, USD/Wp'],
        'Refurbishment Cost (Inverter)':0,
        'Annual O&M Cost': (1000 * EPC.attrs['PV Size'] * 0.01) / 12,
        'Land Rental': Inputs.attrs['OprCos'] * EPC.attrs['System area'] / 12,
        'Total Cost': ((1000 * EPC.attrs['PV Size'] * 0.01) / 12) + (Inputs.attrs['OprCos'] * EPC.attrs['System area'] / 12),
        'Cost Check': np.abs((EPC['New Price']['Installation cost exc. panels'] + 1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp'])/EPC.attrs['PV Size'])/1000,
        'LCOE':0,
        }
        print(Panel.attrs['Cost, USD/Wp'])
    InitialS = pd.Series(Initial)
    df = df.append(InitialS, ignore_index=True)
    return df, Initial

#Converts date to datetime object
def StartDate(Date):
    DTobj = datetime.strptime(Date, '%d/%m/%Y')
    return DTobj

#Fetches property for selected month
def IrrInit(Date,Prop,ProjName):
    DTobj = datetime.strptime(Date, '%d/%m/%Y')
    Month = DTobj.month
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Irr = f['Irradiance']
        P = np.asarray(Irr[Prop])
    return P[Month]


#Fetches property for selected month
def Irr(Date,Prop,ProjName):
    Month = Date.month - 1
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Irr = f['Irradiance']
        P = np.asarray(Irr[Prop])
    return P[Month]

#Calculates and Inputs Burn-in coefficients
def BurnInCoef(ProjName):
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Panel = f['Pannel Data']
        Irr = f['Irradiance']
        DeltaD = Panel.attrs['Burn in %, Δd']
        DeltaD = DeltaD.replace("%", "")
        DeltaD = float(DeltaD) * 0.01

        dL = Panel.attrs['Long-term Degradation, dL, %/year']
        dL = dL.replace("%", "")
        dL = float(dL) * 0.01
        dL = -dL /np.sum(Irr['PeakSunHours'])
        Panel.attrs['dL, %/hr'] = dL
        
        ST = Panel.attrs['Burn in peak sun hours, St, hours']


        a = (DeltaD - dL * ST)/(np.power(ST,2))
        Panel.attrs['a'] = a

        b = -dL - 2 * a * ST
        Panel.attrs['b'] = b 

        m = dL
        Panel.attrs['m'] = m
        c = (1 - DeltaD) - m * ST
        Panel.attrs['c'] = c 

    return

#Converts datetime object to month number
def MonthsTODatetime(X):
    X = datetime.strptime(X,'%m')
    return X

#Calculates XNPV for selected rate, values and dates
def xnpv(rate, values, dates):
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])

#Main loop of the Cash flow model
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
            Curr['Date'] = Prev['Date'] + relativedelta(months=+1)
            Curr['Project Year'] = np.abs(Initial['Date'] - Curr['Date']).days // 365
            Curr['Project Time'] = np.abs(Initial['Date'] - Curr['Date'])
            Curr['Panel Lifetime'] = Prev['Panel Lifetime'] - (Curr['Project Time']-Prev['Project Time'])
            Curr['Inverter Lifetime'] = Prev['Inverter Lifetime'] - (Curr['Project Time']-Prev['Project Time'])

            if Curr['Panel Lifetime'].days > 0:
                Curr['Panel Replacement Year'] = False
            else:
                Curr['Panel Replacement Year'] = True
                Curr['Panel Lifetime'] = Initial['Panel Lifetime']

            Curr['Peak Sun Hours'] = Irr(Curr['Date'],'PeakSunHours',ProjName)
            Curr['Cumilative Sun Hours'] = Prev['Cumilative Sun Hours'] + Curr['Peak Sun Hours']
            Curr['Burn in (absolute)'] = (Panel.attrs['a'] * Curr['Cumilative Sun Hours'] * Curr['Cumilative Sun Hours']) + (Panel.attrs['b'] * Curr['Cumilative Sun Hours'] + 1)
            Curr['Long Term Degredation'] = (Panel.attrs['m'] * Curr['Cumilative Sun Hours']) + Panel.attrs['c']
            Curr['Long Term Degredation (abs after burn in)'] = Curr['Long Term Degredation'] + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)

            if Curr['Cumilative Sun Hours'] > Panel.attrs['Burn in peak sun hours, St, hours']:
                Curr['Panel State of Health'] = Curr['Long Term Degredation (abs after burn in)']
            else:
                Curr['Panel State of Health'] = Curr['Burn in (absolute)']
            
            if Curr['Cumilative Sun Hours'] > Panel.attrs['Burn in peak sun hours, St, hours']:
                Curr['Peak Capacity'] = float(EPC.attrs['PV Size']) * (1 - float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01) * float(Curr['Panel State of Health'])
            else:
                Curr['Peak Capacity'] = EPC.attrs['PV Size'] * Curr['Panel State of Health']

            Curr['Monthly Yeild'] = Irr(Curr['Date'],'Yeild',ProjName)

            if Curr['Date'] >= PrjEndDate:
                Curr['PV Generation'] = 0
            else:
                Curr ['PV Generation'] = Curr['Monthly Yeild'] * np.average([Curr['Peak Capacity'],Prev['Peak Capacity']])
            
            Curr['Capital Cost'] = 0
            
            Curr['Panel Price This Year'] = Panel.attrs['Cost, USD/Wp'] + ((Prev['Panel Price This Year'] - Panel.attrs['Cost, USD/Wp'])*(1 - (Inputs.attrs['Dcr'] * 0.01)/12))

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
                    Curr['Refurbishment Cost (Panels - Other)'] = EPC['New Price']['New price'] * np.power((1+(Inputs.attrs['InvCosInf']*0.01)),((Curr['Project Time'].days/365) - 1))
                    Curr['Cumilative Sun Hours'] = IrrInit(Curr['Date'],'PeakSunHours',ProjName)
                    Curr['Burn in (absolute)'] = (Panel.attrs['a'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName) * IrrInit(Curr['Date'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName) + 1)
                    Curr['Long Term Degredation'] = (Panel.attrs['m'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']
                    Curr['Long Term Degredation (abs after burn in)'] = ((Panel.attrs['m'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)
                    Curr['Panel State of Health'] = ((Panel.attrs['m'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)
                    Curr['Peak Capacity'] = EPC.attrs['PV Size'] * (1 - (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01))
                    Curr['PV Generation'] = IrrInit(Curr['Date'],'Yeild',ProjName) * ((EPC.attrs['PV Size'])+(EPC.attrs['PV Size'] * (1 - (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn in %, Δd'].strip('%'))*0.01))))/2

                else:
                    Curr['Panel Replacement Year'] = 0

            Curr['Refurbishment Cost (Panels)'] = Curr['Refurbishment Cost (Panels - Other)']  + Curr['Refurbishment Cost (Panels - PV)']

            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Inverter)'] = 0
            else:
                if Curr['Inverter Lifetime'].days < 0: 
                    Curr['Inverter Lifetime'] = Initial['Inverter Lifetime']
                    Curr['Refurbishment Cost (Inverter)'] = EPC['Original Price']['Inverter cost'] * np.power((1 + (Inputs.attrs['InvCosInf']*0.01)),((Curr['Project Time'].days/365) - 1))
                else:
                    Curr['Refurbishment Cost (Inverter)'] = 0

            if Curr['Date'] >= PrjEndDate:
                Curr['Annual O&M Cost'] = 0
            else:
                Curr['Annual O&M Cost'] = Prev['Annual O&M Cost'] * (1 + (Inputs.attrs['OprCosInf']*0.01)/12)
            
            if Curr['Date'] >= PrjEndDate:
                Curr['Land Retnal'] = 0
            else:
                Curr['Land Rental'] = Prev['Land Rental'] * (1 + (Inputs.attrs['OprCosInf']*0.01)/12)

            if Curr['Date'] >= PrjEndDate:
                Curr['Total Cost'] = 0
            else:
                Curr['Total Cost']  = Curr['Capital Cost'] + Curr['Refurbishment Cost (Panels)'] + Curr['Refurbishment Cost (Inverter)'] + Curr['Annual O&M Cost'] + Curr['Land Rental']
            
            Curr['Cost Check'] = 0

            if Curr['Date'] == Initial['Date']:
                Curr['LCOE'] = 0
            elif Curr['Date'] == Initial['Date'] + timedelta(hours=TimeRes):
                Curr['LCOE'] = 0
            else:
                TCost = df["Total Cost"].to_numpy().copy()
                PVGen = df["PV Generation"].to_numpy().copy()
                PPD = Inputs.attrs['Dcr']*0.01
                D = df["Date"]
                Curr['LCOE'] = (np.abs(EPC['New Price']['New price']) + np.abs(xnpv(PPD,TCost,D))) / xnpv(PPD,PVGen,D)
                #print(np.abs(EPC['New Price']['New price']) + np.abs(xnpv(PPD,TCost,D)))
                #print(xnpv(PPD,PVGen,D))
            CurrS = pd.Series(Curr)
            df = df.append(CurrS,ignore_index=True)
            Prev = Curr.copy()
        Results = pd.read_csv('Results.csv')
        Vals = np.empty(0)
        Heads = list(Results.columns)
        for Head in Heads:
            Val = Curr[Head]
            Vals = np.append(Vals,Val)
        Vals = pd.Series(Vals)
        with open('Results.csv','a') as f:
            Vals.to_csv(f, mode='a', header=False, index=False)
        df.to_excel(ProjName+'.xlsx')
    return
