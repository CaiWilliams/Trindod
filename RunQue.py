import numpy as np
import pandas as pd
import json
import itertools
from Job import *

class Que:
    def __init__(self, filename):
        self.filename = filename
        self.defaultParms = "DefaultParams.json"
        with open(self.defaultParms) as default:
            defaultdict = json.load(default)
            key = list(defaultdict.keys())
            value = list(defaultdict.values())
            for idx, val in enumerate(value):
                setattr(self, key[idx], value[idx])
        with open(self.filename) as params:
            paramsDict = json.load(params)
            key = list(paramsDict.keys())
            value = list(paramsDict.values())
            for idx, val in enumerate(value):
                setattr(self, key[idx], value[idx])

    def Declare(self, **kwargs):
        keys, values = kwargs.items()
        for idx in range(len(keys)):
            setattr(self, keys[idx], values[idx])
        return

    def GenFile(self):
        params = [self.ProjectName, self.PanTyp, self.PanCosInf, self.PanFlrPri, self.InvLif, self.PrjLif,self.Irr, self.ModSta, self.PrjTyp, self.PrjLoc, self.OprCosInf, self.InvCosInf, self.OprCos, self.RenCos, self.RenInf, self.TimStp, self.Tech]
        paramNames = ['ProjectName','PanTyp','PanCosInf','PanFlrPri','InvLif','PrjLif','Irr','ModSta','PrjTyp','PrjLoc','OprCosInf','InvCosInf','OprCos','RenCos','RenInf','TimStp','Tech']
        Jobs = list(itertools.product(*params))
        Jobs = np.vstack(Jobs)
        Jobs = pd.DataFrame(data=Jobs, index=None, columns=paramNames)
        self.filename = self.filename.split('.')[0]
        Jobs.to_csv(self.filename + ".csv", index=False)
        return

    def SaveQue(self):
        self.filename = self.filename.split('.')[0]
        JB = JobQue(self.filename + ".csv")  # Initialies job que object
        JB.LoadQue()  # Loads RunQue as job que object
        JB.LoadLoc()  # Loads locations in job que object
        JB.LoadPan()  # Loads panel in job que object
        JB.LoadTyp()  # Load panel type in job que object
        with open(self.filename + '.JBS', 'wb') as handle:
            pickle.dump(JB.Jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


Q = Que("Params.json")
Q.GenFile()
Q.SaveQue()