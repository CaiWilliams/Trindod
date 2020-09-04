import math
import os
from calendar import isleap
import calendar
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
        'PeakSunHours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)/SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y')),
        'CumilativeSunHours': IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)/SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y')),
        'Burn-inAbsolute':(Panel.attrs['a'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName) + 1),
        'LongTermDegredation':(Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c'],
        'LongTermDegredationAbsolute':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01),
        'PanelStateofHealth':((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01),
        'PeakCapacity': EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)),
        'EffectiveCapacity': EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)),
        'MonthlyYeild': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName)/SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y')),
        'PVGeneration': IrrInit(Inputs.attrs['ModSta'],'Yeild',ProjName)/SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y')) * ((EPC.attrs['PVSize'])+(EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * IrrInit(Inputs.attrs['ModSta'],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))))/2,
        'CapitalCost': 0,#EPC['New Price']['Installation cost exc. panels'] + (1000 * EPC.attrs['PV Size'] * Panel.attrs['Cost, USD/Wp']),
        'RefurbishmentCost(Panels-PV)':0,
        'RefurbishmentCost(Panels-Other)':0,
        'RefurbishmentCost(Panels)':0,
        'PanelPriceThisYear': Panel.attrs['Cost'],
        'RefurbishmentCost(Inverter)':0,
        'AnnualO&MCost': (1000 * EPC.attrs['PVSize'] * 0.01)/12 /SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y')),
        'LandRental': Inputs.attrs['RenCos'] * EPC['New Price']['NewArea'] /12 /SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y')),
        'TotalCost': ((1000 * EPC.attrs['PVSize'] * 0.01) / 12 /SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y'))) + (Inputs.attrs['RenCos'] * EPC['New Price']['NewArea'] / 12 /SunHourDev(ProjName,datetime.strptime(Inputs.attrs['ModSta'], '%d/%m/%Y'))),
        'CostCheck': np.abs((EPC['New Price']['InstallationCostExcPanels'] + 1000 * EPC.attrs['PVSize'] * Panel.attrs['Cost'])/EPC.attrs['PVSize'])/1000,
        'LCOE':0,
        'ProjectYear':0,
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
        z = np.zeros([((Inputs.attrs['PrjLif']*8760)),(len(CFC)+1)])
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
                dev = 366 * 24 * 12
            else:
                dev = 365 * 24 * 12
    return dev

def SunHourDev(ProjName,CD):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Timestep = Inputs.attrs['TimStp'].lower()
        if Timestep == 'month':
            dev = 1
        elif Timestep == 'hour':
            D = calendar.monthrange(CD.year,CD.month)[1]
            dev = D * 24
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
    Month = Date[:].month - 1
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Irr = f['Irradiance']
        P = np.asarray(Irr[Prop])
    return P[Month[:]]

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
def xnpv(rate, values, D):
    if rate <= -1.0:
        return float('inf')
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
    #df[0,0] = 0
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

    CFC = ['Date','ProjectTime','PanelLifetime','InverterLifetime','PanelReplacementYear','PeakSunHours','CumilativeSunHours','Burn-inAbsolute','LongTermDegredation','LongTermDegredationAbsolute','PanelStateofHealth','PeakCapacity','EffectiveCapacity','MonthlyYeild','PVGeneration','CapitalCost','RefurbishmentCost(Panels-PV)','RefurbishmentCost(Panels-Other)','RefurbishmentCost(Panels)','PanelPriceThisYear','RefurbishmentCost(Inverter)','AnnualO&MCost','LandRental','TotalCost','CostCheck','LCOE','ProjectYear']
    CFCD = ['Date','Project Time','Panel Lifetime','Inverter Lifetime','Panel Replacement Year','Peak Sun Hours','Cumilative Sun Hours','Burn in (absolute)','Long Term Degredation','Long Term Degredation (abs after burn in)','Panel State of Health','Peak Capacity','Effective Capacity','Monthly Yeild','PV Generation','Capital Cost','Refurbishment Cost (Panels - PV)','Refurbishment Cost (Panels - Other)','Refurbishment Cost (Panels)','Panel Price This Year','Refurbishment Cost (Inverter)','Annual O&M Cost','Land Rental','Total Cost','Cost Check','LCOE','Project Year']
    with h5py.File(str(ProjName) + ".hdf5", "r+") as f:

        Inputs = f['Inputs']
        Panel = f['Pannel Data']
        EPC = f['EPC Model']

        PrjLif = Inputs.attrs['PrjLif'] * 365
        PrjEndDate = Initial['Date'] + timedelta(days=float(PrjLif))

        # Date = df[i,0]
        # ProjectTime = df[i,1]
        # PanelLifetime = df[i,2]
        # InverterLifetime = df[i,3]
        # PanelReplacementYear = df[i,4]
        # PeakSunHours = df[i,5]
        # CumulativeSunHours = df[i,6]
        # Burn-inAbsolute = df[i,7]
        # LongTermDegredation = df[i,8]
        # LongTermDegredationAbsolute = df[i,9]
        # PanelStateofHealth = df[i,10]
        # PeakCapacity = df[i,11]
        # EffectiveCapacity = df[i,12]
        # MonthlyYeild = df[i,13]
        # PVGeneration = df[i,14]
        # CapitalCost = df[i,15]
        #RefurbishmentCost(Panels-PV) = df[i,16]
        # RefurbishmentCost(Panels-Other) = df[i,17]
        # RefurbishmentCost(Panels) = df[i,18]
        # PanelPriceThisYear = df[i,19]
        # RefurbishmentCost(Inverter) = df[i,20]
        # AnnualO&MCost = df[i,21]
        # LandRental = df[i,22]
        # TotalCost = df[i,23]
        # CostCheck = df[i,24]
        # LCOE = df[i,25]
        # ProjectYear = df[i,26]

        i = 1
        #PSH = np.zeros(len(df))
        #SHD = np.zeros(len(df))
        df[1,0] = df[0,0] + TimestepRevDelt(ProjName)
        while df[i-1,0] < PrjEndDate:
            df[i,0] = df[i-1,0] + TimestepRevDelt(ProjName)
            df[i,1] = (df[i,0] - df[0,0]).days /365
            #df[i,2] = (df[i-1,0] - df[i,0]).total_seconds()/3600
            df[i,2] = df[i-1,2] - ((df[i,0] - df[i-1,0]).total_seconds()/86400)
            df[i,3] = df[i-1,3] - ((df[i,0] - df[i-1,0]).total_seconds()/604800)
            if df[i,2] > 0:
                df[i,4] = False
            else:
                df[i,4] = True
                df[i,2] = df[0,2]
            i = i + 1
        i = 1

        IrrVec = np.vectorize(Irr)
        SHDVec = np.vectorize(SunHourDev)
        SumVec = np.vectorize(suma)

        PSH = Irr(df[:,0],'PeakSunHours',ProjName)
        SHD = SunHourDev(ProjName,df[:,0])
        YEL = Irr(df[:,0],'Yeild',ProjName)

        df[:,5] = PSH[:]/SHD[:]
        df[:,6] = np.cumsum(df[:,5])
        df[:,7] = (Panel.attrs['a'] * df[:,6] * df[:,6]) + (Panel.attrs['b'] * df[:,6] + 1)
        df[:,8] = (Panel.attrs['m'] * df[:,6]) + Panel.attrs['c']
        df[:,9] = df[:,8] + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
        df[:,10] = df[:,9]
        df[:,10] = np.where(df[:,10]>1,df[:,7],df[:,9])
        df[:,13] = YEL[:]/SHD[:]
        df[:,15] = 0
        #df[:,21] = df[0,21] + (1+(Inputs.attrs['OprCosInf']*0.01)/TimeStepDev(ProjName))**range(len(df))
        #df[:,22] = df[0,22] + (1+(Inputs.attrs['OprCosInf']*0.01)/TimeStepDev(ProjName))**range(len(df))
        EMi = np.array(len(df))
        ID = np.linspace(0,len(df),len(df))
        EMi = EffceftiveMultiplier(IDelta,ID,Max,TMY,Poly)
        while df[i-1,0] < PrjEndDate:
                    
            EM = EMi[i]

            if df[i,6] > Panel.attrs['Burn-inPeakSunHours']:
                df[i,11] = float(EPC.attrs['PVSize']) * (1 - float(Panel.attrs['Burn-in'].strip('%'))*0.01) * float(df[i,10])
            else:
                df[i,11] = EPC.attrs['PVSize'] * df[i,10]
            
            df[i,19] = Panel.attrs['Cost'] + ((df[i-1,19] - Panel.attrs['Cost'])*(1 - (Inputs.attrs['Dcr'] * 0.01)/12))
            
            df[i,18] = df[i,16]  + df[i,17]

            df[i,24] = 0

            df[i,12] = df[i,11] * EM

            df[i,14] = df[i,12] * (df[i,13] + df[i-1,13])/2
            df[i,21] = df[i-1,21] * (1 + ((Inputs.attrs['OprCosInf']*0.01)/TimeStepDev(ProjName)))
            df[i,22] = df[i-1,22] * (1 + ((Inputs.attrs['OprCosInf']*0.01)/TimeStepDev(ProjName)))
    
            if df[i,4] == True:
                df[i,16] = 1000 * EPC.attrs['PVSize'] * df[i,19]
                df[i,17] = (np.abs(EPC['New Price']['NewPrice']) * 0.1) * np.power((1+(Inputs.attrs['InvCosInf']*0.01)),((df[i,1].days/365) - 1))
                df[i,7] = Irr(df[i,0],'PeakSunHours',ProjName)
                df[i,8] = (Panel.attrs['a'] * Irr(df[i,0],'PeakSunHours',ProjName) * Irr(df[i,0],'PeakSunHours',ProjName)) + (Panel.attrs['b'] * Irr(df[i,0],'PeakSunHours',ProjName) + 1)
                df[i,9] = (Panel.attrs['m'] * Irr(df[i,0],'PeakSunHours',ProjName)) + Panel.attrs['c']
                df[i,10] = ((Panel.attrs['m'] * Irr(df[i,0],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
                df[i,11] = ((Panel.attrs['m'] * Irr(df[i,0],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01)
                df[i,12] = EPC.attrs['PVSize'] * (1 - (float(Panel.attrs['Burn-in'].strip('%'))*0.01)) * (((Panel.attrs['m'] * Irr(df[i,0],'PeakSunHours',ProjName)) + Panel.attrs['c']) + (float(Panel.attrs['Burn-in'].strip('%'))*0.01))
                df[i,13] = df[i,12] * EM
                df[i,15] = Irr(df[i,0],'Yeild',ProjName) * ((EPC.attrs['PVSize']) + df[i,13])/2
                if df[i,10] > 1:
                    df[i,10] = df[i,7]
                    df[i,12] = EPC.attrs['PVSize'] * df[i,8]
                    df[i,13] = df[i,12] * EM
                df[i,15] = df[0,14] * np.average([df[i-1,13], df[i,13]])
            else:
                df[i,17] = 0
                df[i,17] = 0

            if df[i,4] < 0: 
                df[i,4] = df[0,4]
                df[i,20] = (np.abs(EPC['New Price']['InstallationCostExcPanels']) * np.abs(EPC['New Price']['InverterCostAsPercentofCiepPrice'])) * np.power((1 + (Inputs.attrs['InvCosInf']*0.01)),math.ceil(df[i,1]/3600/365))
            else:
                df[i,20] = 0
            
            df[i,23]  = df[i,16] + df[i,18] + df[i,20] + df[i,21] + df[i,22]
            
            #TCost = df[:i,23]
            #PVGen = df[:i,14]
            #PPD = Inputs.attrs['Dcr']*0.01
            #D = df[:i,1]

            #df[i,25] = ((np.abs(EPC['New Price']['InstallationCostExcPanels']) + (EPC.attrs['PVSize']*Panel.attrs['Cost']*1000)) + np.abs(xnpv(PPD,TCost,D))) / xnpv(PPD,PVGen,D)
            #CurrS = pd.Series(Curr)
            #CurrS = CurrS.to_numpy()
            #df[i] = CurrS

            i = i + 1
            #Prev = Curr.copy()
        
        #Results(ProjName, Curr)
        TCost = np.zeros((len(df)-1,len(df)-1))
        PVGen = np.zeros((len(df)-1,len(df)-1))
        D = np.zeros((len(df)-1,len(df)-1))
        for j in range(len(df)-2):
            TCost[:j,j] = df[:j,23]
            PVGen[:j,j] = df[:j,14]
            D[:j,j] = df[:j,1]/8760
        PPD = Inputs.attrs['Dcr']*0.01
       #print(TCost[:])
        df[:-1,25] = ((np.abs(EPC['New Price']['InstallationCostExcPanels']) + (EPC.attrs['PVSize']*Panel.attrs['Cost']*1000)) + np.abs(xnpv(PPD,TCost[:,:],D[:,:]))) / xnpv(PPD,PVGen[:,:],D[:,:])
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
    TMYI = np.round((I+IDelta) % (Max))
    G = (TMY['G(i)'].loc[TMYI]).index
    G = G.to_numpy()
    for n in range(len(G)):
        if G[n] != 0:
            G[n] = np.log(G[n] * 118)
        else:
            G[n] = 1
    G[:] = Poly(G[:])
    return G

def suma(A,B):
    return A+B