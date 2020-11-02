import pandas as pd
import multiprocessing as mp
import os
import json
import time

class JobQue:
    def __init__(self, Que):
        self.QueFile = Que
        self.Locations = 'Data\Location'
        self.Panels = 'Data\Panel'
        self.PanelData = 'Data\PanelData.csv'
        self.Types = 'Data\Type'
        return
    
    def LoadQue(self):
        self.QueDataset = pd.read_csv(self.QueFile)
        self.Jobs = self.QueDataset.to_dict(orient='records')
        return

    def LoadLoc(self):
        extn = ".json"
        self.Loc = list()
        i =  0 
        for Job in self.Jobs:
            with open((self.Locations + "\\" + Job['PrjLoc'] + extn)) as f:
                self.Loc.append(json.load(f))
            self.Jobs[i].update(self.Loc[i])
            i = i+1
        return
    
    def LoadPan(self):
        P = pd.read_csv(self.PanelData)
        self.Pan = list()
        self.EM = list()
        for i in range(len(self.Jobs)):
            self.Pan.append(P.iloc[self.Jobs[i]['PanTyp']].to_dict())
            self.Jobs[i].update(self.Pan[i])
        i = 0
        for Job in self.Jobs:
            f = self.Panels + "\\" + str(Job['Tech']) + ".csv"
            self.EM.append(pd.read_csv(f).to_dict(orient='records'))
            self.Jobs[i].update(self.EM[i][0])
            i = i + 1
        return

    def LoadTyp(self):
        extn = ".csv"
        self.Typ = list()
        i = 0 
        for Job in self.Jobs:
            f = self.Types + "\\" + Job['PrjTyp'] +extn
            self.Typ.append(pd.read_csv(f).to_dict(orient='records'))
            self.Jobs[i].update(self.Typ[i][0])
            i = i + 1
        return
