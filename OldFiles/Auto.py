import os
import csv
import h5py
import numpy as np
import pandas as pd
from EPCM import Epcm
from CashFlow import Cashflow
from Init import TypeCheck,PanCheck,SaveProject,RiskFetch
from Variables import GenVars,PrintVars,Load,Save

def Main():
    Que ='RunQue.csv'
    RQ = LoadQue(Que)
    RunItem(RQ)
    return

def LoadQue(Que):
    RQ = pd.read_csv(Que)
    return RQ

def RunItem(RQ):
    #RQD = RQ.drop(['ProjectName','PreCalculated','Latitude','Longitude'],axis=1)
    n = 0
    for row in RQ.itertuples(index=False):
        Vars = row._asdict()
        ProjName = RQ.loc[n]['ProjectName']
        PreCalc = RQ.loc[n]['PreCalculated']
        Lat = RQ.loc[n]['Latitude']
        Lon = RQ.loc[n]['Longitude']
        n = n + 1
        Model(ProjName,PreCalc,Lat,Lon,Vars)
    return

def Model(ProjName,PreCalc,Lat,Lon,Vars):
    Save(Vars,"Data/Variables.p")
    ProjectSetup(ProjName,PreCalc)
    SaveProject(ProjName,PreCalc)
    RiskFetch(ProjName)
    TypeCheck(ProjName)
    PanCheck(ProjName)
    Epcm(ProjName)
    Cashflow(ProjName)
    return

def ProjectSetup(ProjName,PreCalc): 
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
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
        if PreCalc == 'y':
            Locations = os.listdir('Data/Location')
            LocationsLower = [os.path.splitext(Location)[0].lower() for Location in Locations]
            Loc = Inputs.attrs['PrjLoc'].lower()
            LocNum = LocationsLower.index(Loc)
            file = Locations[int(LocNum)]
            name = file
            name = name.split('.')
            name = name[0]
            df = pd.read_csv('Data/Location/'+file,header=None)
            df.to_csv('Temp.csv')
    f.close()
    return

Main()