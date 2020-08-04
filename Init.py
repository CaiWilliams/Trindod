import numpy as np
import os
import h5py
import requests
import csv
import pandas as pd
from Variables import Load

#Creates project
def CreateProject():
    ProName = ProjectName()
    ProLoc, Lat, Lon, PreCal = ProjectLoc()
    if PreCal == 'y':
        with open('Temp.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 0:
                    Yeild = np.asarray(row)
                    Yeild = Yeild[1:].astype(float)
                    with h5py.File(ProName + ".hdf5", "a") as f:
                        Irr = f.require_group("Irradiance")
                        Irr.require_dataset("Yeild",data=Yeild,shape=np.shape(Yeild),dtype='f8')
                elif line == 1:
                    Peak = np.asarray(row)
                    Peak = Peak[1:].astype(float)
                    with h5py.File(ProName + ".hdf5", "a") as f:
                        Irr = f.require_group("Irradiance")
                        Irr.require_dataset("PeakSunHours",data=Peak,shape=np.shape(Peak),dtype='f8')
                line = line + 1
        os.remove('Temp.csv')

    ProjectSetup(ProName)
    TypeCheck(ProName)
    PanCheck(ProName)
    return ProName

def AutoCreateProject(ProName, ProLoc, Lat, Lon, PreCal):
    ProjectSetup(ProName)
    LocCheck(Lat, Lon)
    TypeCheck(ProName)
    PanChekc(ProName)
    return

#Qeries for project name
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

#Queries for project location 
def ProjectLoc():
    os.system('clear')
    print("Precalculated Location? Y/N")
    J = input().lower()
    os.system('clear')
    if J == "y":
        Locations = os.listdir('Data/Location')
        print("Select Location:")
        i = 1
        for Location in Locations:
            Location = Location.split('.')
            LocP = Location[0]
            print(str(i)+':'+LocP)
            i = i + 1
        LocNum = input()
        file = Locations[int(LocNum)-1]
        name = file
        name = name.split('.')
        name = name[0]
        with open('Data/Location/'+file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 2:
                    Lat = row[0]
                elif line == 3:
                    Lon = row[0]
                line = line + 1

        df = pd.read_csv('Data/Location/'+file)
        print(df)
        df.to_csv('Temp.csv')
    elif J == "n":
        print("Location Name:")
        name = input()
        print("Location Latitude:")
        Lat = float(input())
        print("Location Longitude:")
        Lon = float(input())
        os.system('clear')
        LocCheck(Lat, Lon)
    os.system('clear')
    print("Location Name: " + name)
    print("Location Latitude: " + str(Lat))
    print("Location Longitude: "+ str(Lon))
    print("Location Details Correct? Y/N")
    X = input().lower()
    if X == "y":
        return name,Lat,Lon,J 
    elif X == "n":
        ProjectLoc()
    else:
        print("Invalid Entry!")
        ProjectName()
    return

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

#Setsup project file
def ProjectSetup(PrjName): 
    with h5py.File(PrjName + ".hdf5", "a") as f:
        Project = f.require_group("Project")
        Inputs = f.require_group("Inputs")
        Outputs = f.require_group("Outputs")
        EPC = f.require_group("EPC Model")
        PanData = f.require_group("Pannel Data")
        TecEco = f.require_group("Techno Economics")
        Loc = f.require_group("Location")
        Irr = f.require_group("Irradiance")
        Const = f.require_group("Constants")
        Deg = f.require_group("Degredation Rate Check")

        Variables = Load("Data/Variables.p")
        for Key in Variables.keys():
            Inputs.attrs[Key] = Variables[Key]
        f.close()

        
    
    #IrrData = dftreat()
    #IrrData.to_hdf(PrjName + ".hdf5",key='IrradianceRaw', mode='a')