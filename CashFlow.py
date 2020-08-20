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
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:

        Inputs = f['Inputs']
        EPC = f['EPC Model']
        Panel = f['Pannel Data']

        df = pd.DataFrame(columns=CFC)
        
        Initial = {'Date': datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y'),
        'Project Time': timedelta(days=0),
        'Panel Lifetime': timedelta(weeks=float(Panel.attrs['Life']) * 52) ,
        'Inverter Lifetime': timedelta(weeks=float(Inputs.attrs['InvLif'] * 52)),
        'Panel Replacement Year' : False,
        'Peak Sun Hours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
        'Cumilative Sun Hours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName),
        'Burn in (absolute)':(Panel.attrs['a'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) + 1),
        'Long Term Degredation':(Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c'],
        'Long Term Degredation (abs after burn in)':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01),
        'Panel State of Health':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01),
        'Peak Capacity': EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)),
        'Monthly Yeild': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName),
        'PV Generation': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName) * ((EPC.attrs['PVSize'])+(EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))))/2,
        'Capital Cost': 0,#EPC['New Price']['Installation cost exc. panels'] + (1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp']),
        'Refurbishment Cost (Panels - PV)':0,
        'Refurbishment Cost (Panels - Other)':0,
        'Refurbishment Cost (Panels)':0,
        'Panel Price This Year': Panel.attrs['Cost'],
        'Refurbishment Cost (Inverter)':0,
        'Annual O&M Cost': (1000 * EPC.attrs['PVSize'] * 0.01) / 12,
        'Land Rental': Inputs.attrs['RenCos'] * EPC['New Price']['NewArea'] / 12,
        'Total Cost': ((1000 * EPC.attrs['PVSize'] * 0.01) / 12) + (Inputs.attrs['RenCos'] * EPC['New Price']['NewArea'] / 12),
        'Cost Check': np.abs((EPC['New Price']['InstallationCostExcPanels'] + 1000 * EPC.attrs['PVSize'] * Panel.attrs['Cost'])/EPC.attrs['PVSize'])/1000,
        'LCOE':0,
        }
        if Initial['Panel State of Health'] > 1:
            Initial['Panel State of Health'] = Initial['Burn in (absolute)']
            Initial['Peak Capacity'] = EPC.attrs['PVSize'] * Initial['Panel State of Health']
            Initial['PV Generation'] = Initial['Monthly Yeild'] * np.average([EPC.attrs['PVSize'], Initial['Peak Capacity']])
        InitialS = pd.Series(Initial)
        df = df.append(InitialS, ignore_index=True)
        dfA = df.to_numpy()
        z = np.zeros([1,(len(CFC)+1)])
        I=0
        while I < (Inputs.attrs['PrjLif']*12):
            dfA = np.append(dfA, z,axis=0)
            I = I + 1
        a = pd.DataFrame(dfA)
        a.to_csv('test.csv')
    return dfA, Initial

#Converts date to datetime object
def StartDate(Date):
    DTobj = datetime.strptime(Date, '%d/%m/%Y')
    print(DTobj)
    return DTobj

#Fetches property for selected month
def IrrInit(Date,Prop,ProjName):
    DTobj = datetime.strptime(Date, '%d/%m/%Y')
    Month = DTobj.month - 1
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Irr = f['Irradiance']
        P = np.asarray(Irr[Prop])
    return P[Month]


#Fetches property for selected month
def Irr(Date,Prop,ProjName):
    Month = Date.month - 1
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Irr = f['Irradiance']
        P = np.asarray(Irr[Prop])
    return P[Month]

#Calculates and Inputs Burn-in coefficients
def BurnInCoef(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Panel = f['Pannel Data']
        Irr = f['Irradiance']
        DeltaD = Panel.attrs['Burn-in']
        DeltaD = DeltaD.replace("%", "")
        DeltaD = float(DeltaD) * 0.01

        dL = Panel.attrs['Long-termDegradation']
        dL = dL.replace("%", "")
        dL = float(dL) * 0.01
        dL = -dL /np.sum(Irr['PeakSunHours'])
        Panel.attrs['dL'] = dL
        
        ST = Panel.attrs['Burn-inPeakSunHours']


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
    d0 = dates[0]
    return sum([ vi / (1.0 + rate)**(((di - d0).days) / 365) for vi, di in zip(values, dates)])

#Main loop of the Cash flow model
def ProjectLife(Initial, TimeRes, ProjName, Data):
    Prev = Initial
    Curr = Initial.copy()
    df = Data
    df[0,0] = df[0,0].to_pydatetime()
    CFC = ['Date','Project Year','Panel Lifetime','Inverter Lifetime','Panel Replacement Year','Peak Sun Hours','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Rental','Total Cost','Cost Check','LCOE','Project Time']
    with h5py.File(str(ProjName) + ".hdf5", "r+") as f:
        Inputs = f['Inputs']
        Panel = f['Pannel Data']
        EPC = f['EPC Model']
        PrjLif = Inputs.attrs['PrjLif'] * 365
        PrjEndDate = Initial['Date'] + timedelta(days=float(PrjLif))
        i = 1
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
            Curr['Long Term Degredation (abs after burn in)'] = Curr['Long Term Degredation'] + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)

            if Curr['Cumilative Sun Hours'] > Panel.attrs['Burn-inPeakSunHours']:
                Curr['Panel State of Health'] = Curr['Long Term Degredation (abs after burn in)']
            else:
                Curr['Panel State of Health'] = Curr['Burn in (absolute)']
            
            if Curr['Cumilative Sun Hours'] > Panel.attrs['Burn-inPeakSunHours']:
                Curr['Peak Capacity'] = float(EPC.attrs['PVSize']) * (1 - float(Panel.attrs['Burn-in'].strip('%'))*0.01) * float(Curr['Panel State of Health'])
            else:
                Curr['Peak Capacity'] = EPC.attrs['PVSize'] * Curr['Panel State of Health']

            Curr['Monthly Yeild'] = Irr(Curr['Date'],'Yeild',ProjName)

            if Curr['Date'] >= PrjEndDate:
                Curr['PV Generation'] = 0
            else:
                Curr ['PV Generation'] = Curr['Monthly Yeild'] * np.average([Curr['Peak Capacity'],Prev['Peak Capacity']])
            
            Curr['Capital Cost'] = 0
            
            Curr['Panel Price This Year'] = Panel.attrs['Cost'] + ((Prev['Panel Price This Year'] - Panel.attrs['Cost'])*(1 - (Inputs.attrs['Dcr'] * 0.01)/12))

            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Panels - PV)'] = 0
            else:
                if Curr['Panel Replacement Year'] == True:
                    Curr['Refurbishment Cost (Panels - PV)'] = 1000 * EPC.attrs['PVSize'] * Curr['Panel Price This Year']
                else:
                    Curr['Panel Replacement Year'] = 0
                    Curr['Refurbishment Cost (Panels - PV)'] = 0
            
            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Panels - Other)'] = 0
            else:
                if Curr['Panel Replacement Year'] == True:
                    Curr['Refurbishment Cost (Panels - Other)'] = (np.abs(EPC['New Price']['NewPrice']) * 0.1) * np.power((1+(Inputs.attrs['InvCosInf']*0.01)),((Curr['Project Time'].days/365) - 1))
                    Curr['Cumilative Sun Hours'] = Irr(Curr['Date'],'PeakSunHours',ProjName)
                    Curr['Burn in (absolute)'] = (Panel.attrs['a'] * Irr(Curr['Date'],'PeakSunHours',ProjName) * Irr(Curr['Date'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * Irr(Curr['Date'],'PeakSunHours',ProjName) + 1)
                    Curr['Long Term Degredation'] = (Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']
                    Curr['Long Term Degredation (abs after burn in)'] = ((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
                    Curr['Panel State of Health'] = ((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
                    Curr['Peak Capacity'] = EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))
                    Curr['PV Generation'] = Irr(Curr['Date'],'Yeild',ProjName) * ((EPC.attrs['PVSize'])+(EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))))/2
                    if Curr['Panel State of Health'] > 1:
                        Curr['Panel State of Health'] = Curr['Burn in (absolute)']
                        Curr['Peak Capacity'] = EPC.attrs['PVSize'] * Curr['Burn in (absolute)']
                        
                    Curr['PV Generation'] = Initial['Monthly Yeild'] * np.average([Prev['Peak Capacity'], Curr['Peak Capacity']])
                else:
                    Curr['Panel Replacement Year'] = 0
                    Curr['Refurbishment Cost (Panels - Other)'] = 0

            Curr['Refurbishment Cost (Panels)'] = Curr['Refurbishment Cost (Panels - Other)']  + Curr['Refurbishment Cost (Panels - PV)']

            if Curr['Date'] >= PrjEndDate:
                Curr['Refurbishment Cost (Inverter)'] = 0
            else:
                if Curr['Inverter Lifetime'].days < 0: 
                    Curr['Inverter Lifetime'] = Initial['Inverter Lifetime']
                    Curr['Refurbishment Cost (Inverter)'] = (np.abs(EPC['New Price']['InstallationCostExcPanels']) * np.abs(EPC['New Price']['InverterCostAsPercentofCiepPrice'])) * np.power((1 + (Inputs.attrs['InvCosInf']*0.01)),int((Curr['Project Time'].days/365)))
                else:
                    Curr['Refurbishment Cost (Inverter)'] = 0

            if Curr['Date'] >= PrjEndDate:
                Curr['Annual O&M Cost'] = 0
            else:
                Curr['Annual O&M Cost'] = Prev['Annual O&M Cost'] * (1 + (Inputs.attrs['OprCosInf']*0.01)/12)
            
            if Curr['Date'] >= PrjEndDate:
                Curr['Land Rental'] = 0
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
                TCost = df[:i,22]
                PVGen = df[:i,13]
                PPD = Inputs.attrs['Dcr']*0.01
                D = df[:i,0]
                Curr['LCOE'] = ((np.abs(EPC['New Price']['InstallationCostExcPanels']) + (EPC.attrs['PVSize']*Panel.attrs['Cost']*1000)) + np.abs(xnpv(PPD,TCost,D))) / xnpv(PPD,PVGen,D)
                
            CurrS = pd.Series(Curr)
            CurrS = CurrS.to_numpy()
            df[i] = CurrS
            i = i + 1

            Prev = Curr.copy()
        Results(ProjName, Curr)
        df = pd.DataFrame(df)
        df.columns=CFC
        f.close()
        df.to_hdf(str(ProjName) + ".hdf5",key='CashFlow', mode='a')  
    return

def Results(ProjName,Data):
    Headers = {
        'Outputs': ResOutputs,
        'Inputs':  ResInputs,
        'EPCM': ResEPCM,
        'EPCMOP':  ResEPCMOP,
        'EPCMNP':  ResEPCMNP,
        'Panel': ResPanel,
    }
    Results = pd.read_csv('Results.csv')
    Vals = np.empty(0)
    Heads = list(Results.columns)
    for Head in Heads:
        Head = Head.split('.')
        ResFetch = Headers.get(Head[0])
        Val = ResFetch(Head[1],ProjName,Data)
        Vals = np.append(Vals,Val)
    Vals = pd.DataFrame(Vals,Heads).T
    with open('Results.csv','a') as f:
        Vals.to_csv(f, mode='a', header=False, index=False)
    return

def ResOutputs(Prop,ProjName,Data):
    Val = Data[Prop]
    return Val

def ResInputs(Prop,ProjName,Data):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Val = Inputs.attrs[Prop]
    return Val

def ResEPCM(Prop,ProjName,Data):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        EPCM = f['EPC Model']
        Val = EPCM.attrs[Prop]
    return Val

def ResEPCMOP(Prop,ProjName,Data):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        EPCM = f['EPC Model']
        Val = EPCM['Original Price'][Prop]
    return Val 

def ResEPCMNP(Prop,ProjName,Data):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        EPCM = f['EPC Model']
        Val = EPCM['New Price'][Prop]
    return Val

def ResPanel(Prop,ProjName,Data):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Panel = f['Pannel Data']
        Val = Panel.attrs[Prop]
    return Val