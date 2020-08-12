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
    with h5py.File(ProjectName + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Type = Inputs.attrs['PrjTyp']
        Type = Type.replace(" ","")
        Type = Type + ".csv"
        if Type in os.listdir('Data/Type/'):
            df = pd.read_csv("Data/Type/" + Type)
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
    with h5py.File(ProjectName + ".hdf5", "a") as f:
        Pan = f['Pannel Data']
        Inputs = f['Inputs']
        PanTyp = Inputs.attrs['PanTyp']
        df = pd.read_csv("Data/PanelData.csv")
        Info = df.loc[df['Panel ID'] == str(PanTyp)]
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