import numpy as np
import os
import h5py
from datetime import datetime

def Setup():

    CF = {'Date':0,
    'Project Year':0,
    'Months until panel replacement':0,
    'Months until inverter replacement':0,
    'Panel Replacement Year':False,
    'Peak Sun Hours per Month':0,
    'Cumilative Sun Hours':0,
    'Burn in (absolute)':0,
    'Long Term Degredation':0,
    'Long Term Degredation (abs after burn in)':0,
    'Panel State of Health':100,
    'Peak Capacity':0,
    'Monthly Yeild':0,
    'PV Generation':0,
    'Capital Cost':0,
    'Refurbishment Cost (Panels - PV)':0,
    'Refurbishment Cost (Panels - Other)':0,
    'Refurbishment Cost (Panels)':0,
    'Panel Price This Year':0,
    'Refurbishment Cost (Inverter)':0,
    'Annual O&M Cost':0,
    'Land Retnal':0,
    'Total Cost':0,
    'Cost Check':0,
    'LCOE':0,
    }

    hf = h5py.File("test.hdf5","r")
    Inputs = hf.get('Inputs')

    CF["Date"] = Inputs.attrs["ModSta"]
    CF["Date"] = datetime.strptime(CF["Date"],'%d/%m/%y')

    CF["Months until inverter replacement"] = Inputs.attrs["InvLif"]




    
Setup()