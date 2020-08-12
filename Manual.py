import os
import csv
import h5py
import numpy as np
import pandas as pd
from Init import LocCheck,TypeCheck,PanCheck
from Variables import GenVars,PrintVars,Load,Save

def Main():
    Name = ProjectName()
    Location, Latitude, Longitude, PreCal = ProjectLocation()
    SaveProject(Name,PreCal)
    ProjectSetup(Name)
    GenVars()
    ChangeVars()
    TypeCheck(Name)
    PanCheck(Name)
    return Name

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

def ProjectLocation():
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

def ChangeVars():
    PrintVars()
    Variables = Load("Data/Variables.p")
    VariablesDict = Load("Data/VariablesDict.p")
    VariablesUnits = Load("Data/VariablesUnits.p")
    VariablesIntVals = Load("Data/VariablesIntVals.p")
    print("")
    print("Would you like to change any value? Enter number (1-14) to do so. Otherwise just hit enter")
    x = input()
    if x == '':
        x = 0
    if 15 > int(x)  and int(x) > 0:
        print(VariablesDict[VariablesIntVals[int(x)]] + ": ")
        NewVal = input()
        SrcType = type(Variables[VariablesIntVals[int(x)]])
        print(SrcType)
        if SrcType == str:
            NewVal = str(NewVal)
        elif SrcType == int:
            NewVal = int(NewVal)
        else:
            NewVal = float(NewVal)
        print(type(NewVal))
        Variables[VariablesIntVals[int(x)]] = NewVal
        Save(Variables,"Data/Variables.p")
        PrintVars()
    
    print("Are you done making changes: Y/N")
    Conf = input().lower()
    if Conf == 'n':
        ChangeVars()
    else:
        Save(Variables,"Data/Variables.p")
        return

def SaveProject(Name,PreCal):
    if PreCal == 'y':
        with open('Temp.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 0:
                    Yeild = np.asarray(row)
                    Yeild = Yeild[1:].astype(float)
                    with h5py.File(Name + ".hdf5", "a") as f:
                        Irr = f.require_group("Irradiance")
                        Irr.require_dataset("Yeild",data=Yeild,shape=np.shape(Yeild),dtype='f8')
                elif line == 1:
                    Peak = np.asarray(row)
                    Peak = Peak[1:].astype(float)
                    with h5py.File(Name + ".hdf5", "a") as f:
                        Irr = f.require_group("Irradiance")
                        Irr.require_dataset("PeakSunHours",data=Peak,shape=np.shape(Peak),dtype='f8')
                line = line + 1
        os.remove('Temp.csv')
    return

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