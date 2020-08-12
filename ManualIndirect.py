import os
import csv
import h5py
import numpy as np
import pandas as pd
from Init import LocCheck,TypeCheck,PanCheck
from Variables import GenVars,PrintVars,Load,Save

colums = ["Project Name", "Pre Calculated", "Location", "Latitude", "Longitude", "Pannel Type", "Pannel Cost Inflation", "Pannel Floor Price",
            "Inverter Life", "Project Life", "IRR Selection", "Model Start", "Project Type", "Project Location", "Operating Cost Inflation", "Inverter Cost Inflation", "Operating Cost", "Rental Inflation"]


def ProjName():
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

def ProjLocation():
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

def LoadVars():
    Var = Load("Data/Variables.p")
    return Var.values()

def ToRunQue(ProjectName="Default", PreCalculated="Y", Location="Fiji", Latitude="-18.112108", Longitude="178.45359", PannelType="1", PannelCostInflation="-1", PannelFloorPrice="0.245", InverterLife="10", 
            ProjectLife="20", IRRSelection="High Risk", ModelStart="01/01/19", ProjectType="Groundmount PV Array", ProjectLocation="Fiji", OperatingCostInflation="2.1", InverterCostInflation="2.1", OperatingCost="0.5", RentalInflation="2.1"):
        Project = [ProjectName, PreCalculated, Location, Latitude, Longitude, PannelType, PannelCostInflation, PannelFloorPrice, InverterLife, ProjectLife, IRRSelection, ModelStart,
                    ProjectType, ProjectLocation, OperatingCostInflation, InverterCostInflation, OperatingCost, RentalInflation]
        df = pd.DataFrame(columns = colums)
        Rq = pd.read_csv('RunQue.csv')
        df.loc[len(Rq.index) + 1] = Project
        df.to_csv('RunQue.csv',mode='a',header=False)
        return

def Main():
    Name = ProjName()
    Location, Latitude, Longitude, PreCal = ProjLocation()
    GenVars()
    ChangeVars()
    PannelType,PannelCostInflation,PannelFloorPrice,InverterLife,ProjectLife,IRRSelection,ModelStart,ProjectType,ProjectLocation,OperatingCostInflation,InverterCostInflation,OperatingCost,RentalInflation = LoadVars()
    ToRunQue(Name, PreCal, Location, Latitude, Longitude,PannelType,PannelCostInflation,PannelFloorPrice,InverterLife,ProjectLife,IRRSelection,ModelStart,ProjectType,ProjectLocation,OperatingCostInflation,InverterCostInflation,OperatingCost,RentalInflation)
    return

Main()

