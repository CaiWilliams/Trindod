import numpy as np
import pandas as pd
import pickle
import h5py

#Calcualtes or Fetches Data relateing to EPC Model
def Epcm(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:

        EPC = f['EPC Model']
        Inputs = f['Inputs']
        PanelData = f['Pannel Data']

        OPG = EPC.require_group("Original Price")
        NPG = EPC.require_group("New Price")

        #Original Price
        Costs = ['Design', 'Construction', 'Framing', 'DCcabling', 'ACcabling', 'CivilWork(Panels)', 'CivilWork(general)', 'PVPanels', 'FixedProjectCosts', 'Freight(Panels)', 'Freight(other)', 'Inverters', 'Controls']
        OPC = 0
        for Cost in Costs:
            OPC = OPC + EPC.attrs[Cost]
        OP = OPG.require_dataset("OriginalPrice", shape=np.shape(OPC), data=OPC ,dtype='f8')

        PEPC = OPC - EPC.attrs['PVPanels']
        PEP = OPG.require_dataset('PriceExcludingPanels', shape=np.shape(PEPC), data=PEPC, dtype='f8')

        PSC = 410 #EPC.attrs['Panel size']
        PS = OPG.require_dataset('PanelSize', shape=np.shape(PSC), data=PSC, dtype='f8')

        NPC  = 1000 * (EPC.attrs['PVSize'] / PSC)
        NP = OPG.require_dataset('NumberofPanels', shape=np.shape(NPC), data=NPC, dtype='f8')

        ICPC = PEPC/NPC
        ICP = OPG.require_dataset('InstallCostPerPanel(excpanels)', shape=np.shape(ICPC), data=ICPC, dtype='f8')

        IVC = EPC.attrs['Inverters']
        IV = OPG.require_dataset('InverterCost', shape=np.shape(IVC), data=IVC, dtype='f8')

        OAC = EPC.attrs['SystemArea']
        OA = OPG.require_dataset('OldArea', shape=np.shape(OAC), data=OAC, dtype='f8')

        #New Price
        
        PCC = PanelData.attrs['Cost']
        PC = NPG.require_dataset('PanelCost', shape=np.shape(PCC), data=PCC, dtype='f8')

        ERPC = PanelData.attrs['PowerDensity'] * 1.968 * 0.992 
        ERP = NPG.require_dataset('EqRatingofPanels', shape=np.shape(ERPC), data=ERPC, dtype='f8')

        RNPC = 1000 * (EPC.attrs['PVSize']/ERPC)
        RNP = NPG.require_dataset('RequiredNumberofPanels', shape=np.shape(RNPC), data=RNPC, dtype='f8')

        ICEPC = RNPC * ICPC
        ICEP = NPG.require_dataset('InstallationCostExcPanels', shape=np.shape(ICEPC), data=ICEPC, dtype='f8')

        NPCC = PCC * 1000 * EPC.attrs['PVSize']
        NPC2 = NPG.require_dataset('PanelCost', shape=np.shape(NPCC), data=NPCC, dtype='f8')

        NPC3 = ICEPC + NPCC

        NP = NPG.require_dataset('NewPrice', shape=np.shape(NPC3), data=NPC3, dtype='f8')

        ICPPC = IVC/ICEPC
        ICPP = NPG.require_dataset('InverterCostAsPercentofCiepPrice', shape=np.shape(ICPPC), data=ICPPC, dtype='f8')

        NAC = ((((1.92 * np.cos(np.radians((TiltDeg(ProjName)))) * 2 + ArraySpaceing(ProjName))* 0.99)/2) * (RNPC))
        NA = NPG.require_dataset('NewArea', shape=np.shape(NAC), data=NAC, dtype='f8')

    return

#Fetches panel tilt for location
def TiltDeg(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Loc = Inputs.attrs['PrjLoc']
        Prop = "Tilt. deg"
        Rec = FetchLocInfo(Loc, Prop)
        Ref = float(Rec)
    return Rec

#Fetches array spaceing for location
def ArraySpaceing(ProjName):
    with h5py.File(str(ProjName) + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Loc = Inputs.attrs['PrjLoc']
        Prop = "Table spacing, m"
        Rec = FetchLocInfo(Loc, Prop)
        Ref = float(Rec)
    return Rec

#Fetches location info from file
def FetchLocInfo(Name, Prop):
    df = pd.read_csv("Data/LocationData.csv")
    Name = Name.replace(" ","")
    Name = Name.lower()
    Location = df.loc[df['Location'] == Name]
    Rec = Location.loc[:,Prop]
    Rec = Rec.values[0]
    return Rec
