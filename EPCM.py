import numpy as np
import pandas as pd
import pickle
import h5py

def Main(ProjName):
    with h5py.File(ProjName + ".hdf5", "a") as f:

        EPC = f['EPC Model']
        Inputs = f['Inputs']
        PanelData = f['Pannel Data']

        OPG = EPC.require_group("Original Price")
        NPG = EPC.require_group("New Price")

        #Original Price
        Costs = ['Design', 'Construction', 'Framing', 'DC cabling', 'AC cabling', 'Civil work (Panels)', 'Civil work (general)', 'PV Panels', 'Fixed project costs', 'Freight (Panels)', 'Freight (other)', 'Inverters', 'Controls']
        OPC = 0
        for Cost in Costs:
            OPC = OPC + EPC.attrs[Cost]
        OP = OPG.require_dataset("Original Price", shape=np.shape(OPC), data=OPC ,dtype='f8')

        PEPC = OPC - EPC.attrs['PV Panels']
        PEP = OPG.require_dataset('Price excluding panels', shape=np.shape(PEPC), data=PEPC, dtype='f8')

        PSC = EPC.attrs['Panel size']
        PS = OPG.require_dataset('Panel size', shape=np.shape(PSC), data=PSC, dtype='f8')

        NPC  = 1000 * (EPC.attrs['PV Size'] / PSC)
        NP = OPG.require_dataset('Number of panels', shape=np.shape(NPC), data=NPC, dtype='f8')

        ICPC = PEPC/NPC
        ICP = OPG.require_dataset('Install cost per panel (exc. panels)', shape=np.shape(ICPC), data=ICPC, dtype='f8')

        IVC = EPC.attrs['Inverters']
        IV = OPG.require_dataset('Inverter cost', shape=np.shape(IVC), data=IVC, dtype='f8')

        OAC = EPC.attrs['System area']
        OA = OPG.require_dataset('OldArea, sqm', shape=np.shape(OAC), data=OAC, dtype='f8')

        #New Price
        
        PCC = PanelData.attrs['Cost, USD/Wp']
        PC = NPG.require_dataset('Panel cost', shape=np.shape(PCC), data=PCC, dtype='f8')

        ERPC = PanelData.attrs['Power density, Wp/sq.m'] * 1.968 * 0.992 
        ERP = NPG.require_dataset('Eq. rating of panels', shape=np.shape(ERPC), data=ERPC, dtype='f8')

        RNPC = 1000 * EPC.attrs['PV Size']/ERPC
        RNP = NPG.require_dataset('Required number of panels', shape=np.shape(RNPC), data=RNPC, dtype='f8')

        ICEPC = RNPC * ICPC
        ICEP = NPG.require_dataset('Installation cost exc. panels', shape=np.shape(ICEPC), data=ICEPC, dtype='f8')

        NPCC = PCC * 1000 * EPC.attrs['PV Size']
        NPC2 = NPG.require_dataset('Panel cost', shape=np.shape(NPCC), data=NPCC, dtype='f8')

        NPC3 = ICEPC + NPCC
        NP = NPG.require_dataset('New price', shape=np.shape(NP), data=NP, dtype='f8')

        ICPPC = IVC/ICEPC
        ICPP = NPG.require_dataset('Inverter cost as % of Ciep price', shape=np.shape(ICPPC), data=ICPPC, dtype='f8')

        NAC = ((((1.92 * np.cos(TiltDeg(ProjName))) * 2 + ArraySpaceing(ProjName))* 0.99)/2) * (RNPC)
        NA = NPG.require_dataset('New area, sqm', shape=np.shape(NAC), data=NAC, dtype='f8')

    return

def TiltDeg(ProjName):
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Loc = Inputs.attrs['PrjLoc']
        Prop = "Tilt. deg"
        Rec = FetchLocInfo(Loc, Prop)
        Ref = float(Rec)
    return Rec

def ArraySpaceing(ProjName):
    with h5py.File(ProjName + ".hdf5", "a") as f:
        Inputs = f['Inputs']
        Loc = Inputs.attrs['PrjLoc']
        Prop = "Table spacing, m"
        Rec = FetchLocInfo(Loc, Prop)
        Ref = float(Rec)
    return Rec

def FetchLocInfo(Name, Prop):
    df = pd.read_csv("Data/Location/LocationData.csv")
    Name = Name.replace(" ","")
    Name = Name.lower()
    Location = df.loc[df['Location'] == Name]
    Rec = Location.loc[:,Prop]
    Rec = Rec.values[0]
    return Rec

Main('19')