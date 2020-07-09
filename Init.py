import numpy as np
import os
import h5py
import requests
import csv
import pandas as pd
from Variables import Load


def CreateProject():
    ProName = ProjectName()
    ProLoc, Lat, Lon = ProjectLoc()
    Irr = LocCheck(Lat, Lon)
    ProjectSetup(ProName, ProLoc, Irr)
    os.remove('Temp.csv')

    
def ProjectName():
    os.system('clear')
    print("What is the name of the project?")
    name = input()
    print("Is " + name + " correct? Y/N")
    X = input().lower()
    if X == "y":
        return name
    elif X == "n":
        ProjectName()
    else:
        print("Invalid Entry!")
        ProjectName()
    return

def ProjectLoc():
    os.system('clear')
    print("Location Name:")
    name = input()
    print("Location Latitude:")
    Lat = float(input())
    print("Location Longitude:")
    Lon = float(input())
    os.system('clear')
    print("Location Name: " + name)
    print("Location Latitude: " + str(Lat))
    print("Location Longitude: "+ str(Lon))
    print("Location Details Correct? Y/N")
    X = input().lower()
    if X == "y":
        return name,Lat,Lon 
    elif X == "n":
        ProjectLoc()
    else:
        print("Invalid Entry!")
        ProjectLoc()
    return

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

def dftreat():
    df = pd.read_csv("Temp.csv",skiprows=8,skipfooter=9,engine='python')
    T = df["time"].str.split(":", n=1,expand=True)
    df["time"] = T[1]
    df["date"] = T[0]
    return df



def ProjectSetup(PrjName,Loc,Content): 
    with h5py.File(PrjName + ".hdf5", "a") as f:
        Project = f.create_group("Project")
        Inputs = f.create_group("Inputs")
        Outputs = f.create_group("Outputs")
        EPC = f.create_group("EPC Model")
        PanData = f.create_group("Pannel Data")
        TecEco = f.create_group("Techno Economics")
        Loc = f.create_group("Location")
        Irr = f.create_group("Irradiance")
        Const = f.create_group("Constants")
        Deg = f.create_group("Degredation Rate Check")

        Variables = Load("Data/Variables.p")
        for Key in Variables.keys():
            Inputs.attrs[Key] = Variables[Key]
        f.close()

    IrrData = dftreat()
    IrrData.to_hdf(PrjName + ".hdf5",key='Irradiance', mode='a')

CreateProject()