import numpy as np
import os
import h5py
import requests
import json
import pandas as pd
from Variables import Load


def CreateProject():
    ProName = ProjectName()
    ProLoc, Lat, Lon = ProjectLoc()
    Irr = LocCheck(Lat, Lon)
    ProjectSetup(ProName, Irr)

    
def ProjectName():
    os.system('cls')
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
    os.system('cls')
    print("Location Name:")
    name = input()
    print("Location Latitude:")
    Lat = float(input())
    print("Location Longitude:")
    Lon = float(input())
    os.system('cls')
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
    Response = requests.get("https://re.jrc.ec.europa.eu/api/seriescalc?lat="+str(Lat)+"&lon="+str(Lon)+"&outputformat=json")
    Code = int(Response.status_code)
    if Code != 200:
        print("Location is not supported by PVGIS! Enter Valid Location!")
        ProjectLoc()
    else:
        Content = Response.json()
        return Content
    return 

def Jsontonumpy():
    df = pd.read_json('IrrData.json')
    arr = df.to_numpy()
    return arr



def ProjectSetup(PrjName,Content): 
    with h5py.File(PrjName + ".hdf5", "a") as f:
        Project = f.create_group("Project")
        Inputs = f.create_group("Inputs")
        Outputs = f.create_group("Outputs")
        EPC = Project.create_group("EPC Model")
        PanData = Project.create_group("Pannel Data")
        TecEco = Project.create_group("Techno Economics")
        Loc = Project.create_group("Location")
        Irr = Project.create_group("Irradiance")
        Const = Project.create_group("Constants")
        Deg = Project.create_group("Degredation Rate Check")

        Variables = Load("Variables.p")
        for Key in Variables.keys():
            Inputs.attrs[Key] = Variables[Key]
        IrrData = JsontoPandas()
        print(IrrData)
        Irr.create_dataset("Irradiance Data", data=IrrData)
        f.close()

CreateProject()