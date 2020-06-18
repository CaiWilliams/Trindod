import numpy as np
import h5py
import time
from Variables import Load
import json


def ProjectSetup(PrjName): 
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
        
        with open('IrrData.json') as IrrData:
            IrrD = json.load(IrrData)
            Irr.create_dataset('Irradiance_Data', data=IrrD)
            Irr.close()