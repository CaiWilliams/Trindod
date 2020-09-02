import math
import os
from calendar import isleap
from datetime import datetime, timedelta

import h5py
import numexpr
import numpy as np
import pandas as pd
from dateutil.relativedelta import *


#Main Run order of CashFlow model
def Cashflow(ProjName):
    BurnInCoef(ProjName)
    InitialS, Initial = Setup(ProjName)
    ProjectLife(Initial, 1, ProjName, InitialS)
    return

#Setsup cash flow model with initial data
def Setup(ProjName):
    CFC = ['Date','ProjectYear','PanelLifetime','InverterLifetime','PanelReplacementYear','PeakSunHours','CumilativeSunHours','Burn-inAbsolute','LongTermDegredation','LongTermDegredationAbsolute','PanelStateofHealth','PeakCapacity','EffectiveCapacity','MonthlyYeild','PVGeneration','CapitalCost','RefurbishmentCost(Panels-PV)','RefurbishmentCost(Panels-Other)','RefurbishmentCost(Panels)','PanelPriceThisYear','RefurbishmentCost(Inverter)','AnnualO&MCost','LandRental','TotalCost','CostCheck','LCOE']
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:

        Inputs = f['Inputs']
        EPC = f['EPC Model']
        Panel = f['Pannel Data']

        df = pd.DataFrame(columns=CFC)
        
        Initial = {'Date': datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y'),
        'ProjectTime': timedelta(days=0),
        'PanelLifetime': timedelta(weeks=float(Panel.attrs['Life']) * 52) ,
        'InverterLifetime': timedelta(weeks=float(Inputs.attrs['InvLif'] * 52)),
        'PanelReplacementYear' : False,
        'PeakSunHours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)/SunHourDev(ProjName),
        'CumilativeSunHours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)/SunHourDev(ProjName),
        'Burn-inAbsolute':(Panel.attrs['a'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) + 1),
        'LongTermDegredation':(Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c'],
        'LongTermDegredationAbsolute':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01),
        'PanelStateofHealth':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01),
        'PeakCapacity': EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)),
        'EffectiveCapacity': EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)),
        'MonthlyYeild': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName)/SunHourDev(ProjName),
        'PVGeneration': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName)/SunHourDev(ProjName) * ((EPC.attrs['PVSize'])+(EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))))/2,
        'CapitalCost': 0,#EPC['New Price']['Installation cost exc. panels'] + (1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp']),
        'RefurbishmentCost(Panels-PV)':0,
        'RefurbishmentCost(Panels-Other)':0,
        'RefurbishmentCost(Panels)':0,
        'PanelPriceThisYear': Panel.attrs['Cost'],
        'RefurbishmentCost(Inverter)':0,
        'AnnualO&MCost': (1000 * EPC.attrs['PVSize'] * 0.01)/12 /SunHourDev(ProjName),
        'LandRental': Inputs.attrs['RenCos'] * EPC['New Price']['NewArea'] /12 /SunHourDev(ProjName),
        'TotalCost': ((1000 * EPC.attrs['PVSize'] * 0.01) / 12 /SunHourDev(ProjName)) + (Inputs.attrs['RenCos'] * EPC['New Price']['NewArea'] / 12 /SunHourDev(ProjName)),
        'CostCheck': np.abs((EPC['New Price']['InstallationCostExcPanels'] + 1000 * EPC.attrs['PVSize'] * Panel.attrs['Cost'])/EPC.attrs['PVSize'])/1000,
        'LCOE':0,
        }
        if Initial['PanelStateofHealth'] > 1:
            Initial['PanelStateofHealth'] = Initial['Burn-inAbsolute']
            Initial['PeakCapacity'] = EPC.attrs['PVSize'] * Initial['PanelStateofHealth']
            Initial['EffectiveCapacity'] = Initial['PeakCapacity']
            Initial['PVGeneration'] = Initial['MonthlyYeild'] * np.average([EPC.attrs['PVSize'], Initial['PeakCapacity']])
        Initial['PanelLifetime'] = Initial['PanelLifetime'].days
        Initial['InverterLifetime'] = Initial['InverterLifetime'].days

        if Inputs.attrs['TimStp'] == 'hour':
            print('Yatzze')
            Initial['PeakCapacity'] = EPC.attrs['PVSize']
            Initial['EffectiveCapacity'] = EPC.attrs['PVSize']
            Initial['Burn-inAbsolute'] = 1
            Initial['LongTermDegredation'] = 1
            Initial['LongTermDegredationAbsolute'] = 1
            Initial['PanelStateofHealth'] = 1
            Initial['PVGeneration'] = Initial['MonthlyYeild'] * EPC.attrs['PVSize']

        InitialS = pd.Series(Initial)
        df = df.append(InitialS, ignore_index=True)
        dfA = df.to_numpy()
        z = np.zeros([((Inputs.attrs['PrjLif']*TimeStepDev(ProjName))+5),(len(CFC)+1)])
        dfA = np.append(dfA, z,axis=0)
        a = pd.DataFrame(dfA)
    return dfA, Initial

def TimestepRevDelt(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Timestep = Inputs.attrs['TimStp'].lower()
        if Timestep == 'month':
            Rev = relativedelta(months=1)
        elif Timestep == 'hour':
            Rev = timedelta(hours=1)
    return Rev

def TimeStepDev(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Timestep = Inputs.attrs['TimStp'].lower()
        if Timestep == 'month':
            dev = 12
        elif Timestep == 'hour':
            if isleap(datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y').year) == True:
                dev = 366 * 24
            else:
                dev = 365 * 24
    return dev

def SunHourDev(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Timestep = Inputs.attrs['TimStp'].lower()
        if Timestep == 'month':
            dev = 1
        elif Timestep == 'hour':
            dev = 31 * 24
    return dev

#Converts date to datetime object
def StartDate(Date):
    DTobj = datetime.strptime(Date, '%d/%m/%Y')
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
    D = dates[:]/8760
    V = sum(values[:] / (1.0 + rate)**(D[:]))
    return V

def to_relativedelta(tdelta):
    return relativedelta(seconds=int(tdelta.total_seconds()),microseconds=tdelta.microseconds)

#Main loop of the Cash flow model
def ProjectLife(Initial, TimeRes, ProjName, Data):
    Prev = Initial
    Curr = Initial.copy()
    df = Data
    df[0,0] = df[0,0].to_pydatetime()
    df[0,1] = 0
    TMY = pd.read_csv('TMY.csv')
    
    with h5py.File(str(ProjName) + ".hdf5", "r+") as f:
        Inputs = f['Inputs']
        if Inputs.attrs['TimStp'] == 'hour':
            TMY['time'] = pd.to_datetime(TMY['time'],format='%Y-%m-%d %H:%M:%S')
            IrradianceStartDate = TMY['time'].values[0]
            ModelStartDate = df[0,0]
            IrradianceStartDate = datetime.utcfromtimestamp(IrradianceStartDate.tolist()/1e9)
            Delta = ModelStartDate - IrradianceStartDate
            IDelta = Delta/TimestepRevDelt(ProjName)
        elif Inputs.attrs['TimStp'] == 'month':
            TMY['time'] = pd.to_datetime(TMY['time'],format='%Y-%m-%d %H:%M:%S')
            IrradianceStartDate = TMY['time'].values[0]
            ModelStartDate = df[0,0]
            IrradianceStartDate = datetime.utcfromtimestamp(IrradianceStartDate.tolist()/1e9)
            Delta = ModelStartDate - IrradianceStartDate
            IDelta = Delta.days/((ModelStartDate+TimestepRevDelt(ProjName))-ModelStartDate).days

    Max  = len(TMY)-1

    with h5py.File(str(ProjName) + ".hdf5", "r+") as f:
        Inputs = f['Inputs']
        Type = Inputs.attrs['Tech'].lower()
        if Type == 'opv':
            TFile = 'Data/Panel/OPV.csv'
        elif Type == 'pvk':
            TFile = 'Data/Panel/PVK.csv'
        elif Type == 'dssc':
            TFile = 'Data/Panel/DSSC.csv'
        elif Type == 'xsi':
            TFile = 'Data/Panel/XSI.csv'
        else:
            TFile = 'Data/Panel/Disable.csv'
    f.close()

    Params = pd.read_csv(TFile,header=None)
    Params = Params.loc[1].values[1:]
    Poly = np.polynomial.Polynomial(Params)

    CFC = ['Date','ProjectYear','PanelLifetime','InverterLifetime','PanelReplacementYear','PeakSunHours','CumilativeSunHours','Burn-inAbsolute','LongTermDegredation','LongTermDegredationAbsolute','PanelStateofHealth','PeakCapacity','EffectiveCapacity','MonthlyYeild','PVGeneration','CapitalCost','RefurbishmentCost(Panels-PV)','RefurbishmentCost(Panels-Other)','RefurbishmentCost(Panels)','PanelPriceThisYear','RefurbishmentCost(Inverter)','AnnualO&MCost','LandRental','TotalCost','CostCheck','LCOE']
    CFCD = ['Date','Project Year','Panel Lifetime','Inverter Lifetime','Panel Replacement Year','Peak Sun Hours','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Effective Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Rental','Total Cost','Cost Check','LCOE','Project Time']
    with h5py.File(str(ProjName) + ".hdf5", "r+") as f:
        Inputs = f['Inputs']
        Panel = f['Pannel Data']
        EPC = f['EPC Model']
        PrjLif = Inputs.attrs['PrjLif'] * 365
        PrjEndDate = Initial['Date'] + timedelta(days=float(PrjLif))
        i = 1
        while Curr['Date'] < PrjEndDate:

            Curr['Date'] = Prev['Date'] + TimestepRevDelt(ProjName)
            EM = EffceftiveMultiplier(IDelta,i,Max,TMY,Poly)
            Curr['ProjectYear'] = float((Curr['Date']-Prev['Date']).days) /365
            
            Curr['ProjectTime'] = float(np.abs(Initial['Date'] - Curr['Date']).total_seconds())/3600
            Curr['PanelLifetime'] = Prev['PanelLifetime'] - ((Curr['Date'] - Prev['Date']).total_seconds()/86400)
            Curr['InverterLifetime'] = Prev['InverterLifetime'] - ((Curr['Date'] - Prev['Date']).total_seconds()/604800)
            if Curr['PanelLifetime'] > 0:
                Curr['PanelReplacementYear'] = False
            else:
                Curr['PanelReplacementYear'] = True
                Curr['PanelLifetime'] = Initial['PanelLifetime']

            Curr['PeakSunHours'] = Irr(Curr['Date'],'PeakSunHours',ProjName)/SunHourDev(ProjName)
            Curr['CumilativeSunHours'] = Prev['CumilativeSunHours'] + Curr['PeakSunHours']
            Curr['Burn-inAbsolute'] = (Panel.attrs['a'] * Curr['CumilativeSunHours'] * Curr['CumilativeSunHours']) + (Panel.attrs['b'] * Curr['CumilativeSunHours'] + 1)
            Curr['LongTermDegredation'] = (Panel.attrs['m'] * Curr['CumilativeSunHours']) + Panel.attrs['c']
            Curr['LongTermDegredationAbsolute'] = Curr['LongTermDegredation'] + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)

            if Curr['CumilativeSunHours'] > Panel.attrs['Burn-inPeakSunHours']:
                Curr['PanelStateofHealth'] = Curr['LongTermDegredationAbsolute']
                Curr['PeakCapacity'] = float(EPC.attrs['PVSize']) * (1 - float(Panel.attrs['Burn-in'].strip('%'))*0.01) * float(Curr['PanelStateofHealth'])
            else:
                Curr['PanelStateofHealth'] = Curr['Burn-inAbsolute']
                Curr['PeakCapacity'] = EPC.attrs['PVSize'] * Curr['PanelStateofHealth']

            Curr['MonthlyYeild'] = Irr(Curr['Date'],'Yeild',ProjName)/SunHourDev(ProjName)

            Curr['CapitalCost'] = 0
            
            Curr['PanelPriceThisYear'] = Panel.attrs['Cost'] + ((Prev['PanelPriceThisYear'] - Panel.attrs['Cost'])*(1 - (Inputs.attrs['Dcr'] * 0.01)/12))
            
            Curr['RefurbishmentCost(Panels)'] = Curr['RefurbishmentCost(Panels-Other)']  + Curr['RefurbishmentCost(Panels-PV)']

            Curr['CostCheck'] = 0

            Curr['EffectiveCapacity'] = Curr['PeakCapacity'] * EM

            Curr ['PVGeneration'] = Curr['MonthlyYeild'] * (Curr['EffectiveCapacity'] + Prev['EffectiveCapacity'])/2
            Curr['AnnualO&MCost'] = Prev['AnnualO&MCost'] * (1 + ((Inputs.attrs['OprCosInf']*0.01)/12/SunHourDev(ProjName)))
            Curr['LandRental'] = Prev['LandRental'] * (1 + ((Inputs.attrs['OprCosInf']*0.01)/12/SunHourDev(ProjName)))

            #if Curr['Date'] > PrjEndDate:
            #    Curr['PVGeneration'] = 0
            #    Curr['RefurbishmentCost(Inverter)'] = 0
            #    Curr['AnnualO&MCost'] = 0
            #    Curr['LandRental'] = 0
            #    Curr['TotalCost'] = 0
                
            if Curr['PanelReplacementYear'] == True:
                Curr['RefurbishmentCost(Panels-PV)'] = 1000 * EPC.attrs['PVSize'] * Curr['PanelPriceThisYear']
                Curr['RefurbishmentCost(Panels-Other)'] = (np.abs(EPC['New Price']['NewPrice']) * 0.1) * np.power((1+(Inputs.attrs['InvCosInf']*0.01)),((Curr['ProjectTime'].days/365) - 1))
                Curr['CumilativeSunHours'] = Irr(Curr['Date'],'PeakSunHours',ProjName)
                Curr['Burn-inAbsolute'] = (Panel.attrs['a'] * Irr(Curr['Date'],'PeakSunHours',ProjName) * Irr(Curr['Date'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * Irr(Curr['Date'],'PeakSunHours',ProjName) + 1)
                Curr['LongTermDegredation'] = (Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']
                Curr['LongTermDegredationAbsolute'] = ((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
                Curr['PanelStateofHealth'] = ((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
                Curr['PeakCapacity'] = EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * Irr(Curr['Date'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))
                Curr['EffectiveCapacity'] = Curr['PeakCapacity'] * EM
                Curr['PVGeneration'] = Irr(Curr['Date'],'Yeild',ProjName) * ((EPC.attrs['PVSize']) + Curr['EffectiveCapacity'])/2
                if Curr['PanelStateofHealth'] > 1:
                    Curr['PanelStateofHealth'] = Curr['Burn-inAbsolute']
                    Curr['PeakCapacity'] = EPC.attrs['PVSize'] * Curr['Burn-inAbsolute']
                    Curr['EffectiveCapacity'] = Curr['PeakCapacity'] * EM
                Curr['PVGeneration'] = Initial['MonthlyYeild'] * np.average([Prev['EffectiveCapacity'], Curr['EffectiveCapacity']])
            else:
                Curr['PanelReplacementYear'] = 0
                Curr['RefurbishmentCost(Panels-PV)'] = 0
                Curr['RefurbishmentCost(Panels-Other)'] = 0

            if Curr['InverterLifetime'] < 0: 
                Curr['InverterLifetime'] = Initial['InverterLifetime']
                Curr['RefurbishmentCost(Inverter)'] = (np.abs(EPC['New Price']['InstallationCostExcPanels']) * np.abs(EPC['New Price']['InverterCostAsPercentofCiepPrice'])) * np.power((1 + (Inputs.attrs['InvCosInf']*0.01)),math.ceil(Curr['ProjectTime']/3600/365))
            else:
                Curr['RefurbishmentCost(Inverter)'] = 0
            
            Curr['TotalCost']  = Curr['CapitalCost'] + Curr['RefurbishmentCost(Panels)'] + Curr['RefurbishmentCost(Inverter)'] + Curr['AnnualO&MCost'] + Curr['LandRental']
            
            
            TCost = df[:i,23]
            PVGen = df[:i,14]
            PPD = Inputs.attrs['Dcr']*0.01
            D = df[:i,1]
            Curr['LCOE'] = ((np.abs(EPC['New Price']['InstallationCostExcPanels']) + (EPC.attrs['PVSize']*Panel.attrs['Cost']*1000)) + np.abs(xnpv(PPD,TCost,D))) / xnpv(PPD,PVGen,D)
            CurrS = pd.Series(Curr)
            CurrS = CurrS.to_numpy()
            df[i] = CurrS
            i = i + 1
            Prev = Curr.copy()
            #print(((np.abs(EPC['New Price']['InstallationCostExcPanels']) + (EPC.attrs['PVSize']*Panel.attrs['Cost']*1000)) + np.abs(xnpv(PPD,TCost,D))))
            #print(xnpv(PPD,PVGen,D))
        Results(ProjName, Curr)
        df = pd.DataFrame(df)
        df.columns=CFCD
        f.close()
        df.to_excel(str(ProjName)+".xlsx")
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

def EffceftiveMultiplier(IDelta,I,Max,TMY,Poly):
    TMYI = ((I+IDelta) % (Max))
    G = TMY['G(i)'].loc[int(TMYI)]
    if G != 0:
        G = np.log(G * 118)
    else:
        return 1
    G = Poly(G)
    return G
