import numpy as np
import os
import h5py
import requests
import csv
import pandas as pd
from Variables import Load


#Checks if Location is valid PVGIS Location
def LocCheck(Lat, Lon):
    url="https://re.jrc.ec.europa.eu/api/seriescalc?lat="+str(Lat)+"&lon="+str(Lon)+"&outputformat=csv"
    Response = requests.get(url)
    Code = int(Response.status_code)
    if Code != 200:
        print("Location is not supported by PVGIS! Enter Valid Location!")
        ProjectLoc()
    else:
        open('Temp.csv','wb').write(Response.content)
    return 

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

#Processing for standard PVGIS output
def dftreat():
    df = pd.read_csv("Temp.csv",skiprows=8,skipfooter=9,engine='python')
    T = df["time"].str.split(":", n=1,expand=True)
    df["time"] = T[1]
    df["date"] = T[0]
    df.to_csv("Temp.csv")
    return df


    #IrrData = dftreat()
    #IrrData.to_hdf(PrjName + ".hdf5",key='IrradianceRaw', mode='a')  

def SaveProject(ProjName,PreCal):
    if PreCal == 'y':
        with open('Temp.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 0:
                    Yeild = np.asarray(row)
                    Yeild = Yeild[1:].astype(float)
                    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
                        Irr = f.require_group("Irradiance")
                        Irr.require_dataset("Yeild",data=Yeild,shape=np.shape(Yeild),dtype='f8')
                elif line == 1:
                    Peak = np.asarray(row)
                    Peak = Peak[1:].astype(float)
                    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
                        Irr = f.require_group("Irradiance")
                        Irr.require_dataset("PeakSunHours",data=Peak,shape=np.shape(Peak),dtype='f8')
                line = line + 1
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