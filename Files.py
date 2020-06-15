import numpy as np
import h5py
import time
from Variables import Load

class Save: 
     
     def ProjectSetup(PrjName): 
        with h5py.File(PrjName + ".hdf5", "a") as f:
            Project = f.create_group("Project")
            Inputs = f.create_group("Inputs")
            Outputs = f.create_group("Outputs")
            Project.create_group("EPC Model")
            Project.create_group("Pannel Data")
            Project.create_group("Techno Economics")
            Project.create_group("Location")
            Project.create_group("Irradiance")
            Project.create_group("Constants")
            Project.create_group("Degredation Rate Check")

            Variables = Load("Variables.p")
            for Key in Variables.keys():
                Inputs.attrs[Key] = Variables[Key]    