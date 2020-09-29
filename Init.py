import numpy as np
import os
import h5py
import io
import requests
import csv
import pandas as pd
from datetime import datetime
from Variables import Load

#Fectches and inputs info regarding project type
def TypeCheck(ProjectName):
    with h5py.File(str(ProjectName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Type = Inputs.attrs['PrjTyp']
        Type = Type.replace(" ","")
        Type = Type + ".csv"
        if Type in os.listdir('Data/Type/'):
            df = pd.read_csv("Data/Type/" + Type,float_precision='round_trip')
            Keys =  list(df.columns)
            Values = df.values[0]
            i = 0 
            for Key in Keys:
                EPC = f['EPC Model']
                EPC.attrs[Key] = Values[i]
                i = i + 1
        else:
            print("Invalid Project Type")
    return

#Fetches and Inputs panel data
def PanCheck(ProjectName):
    with h5py.File(str(ProjectName) + ".hdf5", "a") as f:
        Pan = f['Pannel Data']
        Inputs = f['Inputs']
        PanTyp = Inputs.attrs['PanTyp']
        df = pd.read_csv("Data/PanelData.csv",float_precision='round_trip')
        Info = df.loc[df['PanelID'] == str(PanTyp)]
        Keys = list(Info.columns)
        Values = Info.values[0]
        i = 0
        for Key in Keys:
            Pan.attrs[Key] = Values[i]
            i = i + 1
    return

def LocProps(ProjName):
    with h5py.File(str(ProjName) + ".hdf5","a") as f:
        Inputs = f['Inputs']
        try:
            if Inputs.attrs['LocProps'] == "y":
                tilt = Inputs.attrs['tilt']
                azimuth = Inputs.attrs['azimuth']
            return tilt, azimuth
        except:
            LocationProperties = pd.read_csv('Data/LocationData.csv')
            Loc = LocationProperties.loc[LocationProperties['Location'] == Inputs.attrs['PrjLoc'].lower()]
            
            tilt = Loc['Tilt'].values[0]
            azimuth =  0
            return tilt, azimuth

    return

def SaveProject(ProjName,PreCal):
    LocDat = pd.read_csv('Temp.csv',header=None,index_col=False)
    Yeild = LocDat.loc[0].values[1:]
    Peak = LocDat.loc[1].values[1:]
    lat = LocDat.loc[2].values[1]
    lon = LocDat.loc[3].values[1]
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Irr = f.require_group("Irradiance")
        Irr.require_dataset("Yeild",data=Yeild,shape=np.shape(Yeild),dtype='f8')
        Irr.require_dataset("PeakSunHours",data=Peak,shape=np.shape(Peak),dtype='f8')
        Inputs = f['Inputs']
        tilt,azimuth = LocProps(ProjName)
        r = requests.get('https://re.jrc.ec.europa.eu/api/seriescalc?'+'lat=' +str(lat) + '&lon='+str(lon) + '&angle='+str(tilt)+'&aspect='+str(azimuth)+'&startyear=2015&endyear=2015')
        TMY = io.StringIO(r.content.decode('utf-8'))
        TMY = pd.read_csv(TMY,skipfooter=9,skiprows=[0,1,2,3,4,5,6,7],engine='python')
        for n in TMY.index:
            TMY.at[n,'time'] = datetime.strptime(TMY.at[n,'time'][:-2], '%Y%m%d:%H').replace(year=datetime.strptime(Inputs.attrs['ModSta'],'%d/%m/%Y').year)
    f.close() 
    #hdf = pd.HDFStore(ProjName+'.hdf5')
    #hdf.put('TMY',TMY)
    TMY.to_hdf(str(ProjName)+".hdf5",key='TMY',format='fixed')
    TMY.to_csv('TMY.csv')
    os.remove('Temp.csv')
    return

def RiskFetch(ProjName):
    with h5py.File(str(ProjName) + ".hdf5","a") as p:
        Inputs = p['Inputs']
        with open ('Data/RiskData.csv') as f:
            Risk = pd.read_csv(f)
            SRisk = Risk.loc[Risk['Scenario'] == Inputs.attrs['Irr']]
            DCR = SRisk['IRR'].values
            Inputs.attrs['Dcr'] = DCR[0]
    return