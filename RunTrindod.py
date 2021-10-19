from Trindod import *
import time
import os

class LCOERun:

    def __init__(self, filename, paneldatafile):
        self.filename = filename
        self.paneldatafile = paneldatafile

    def ModPop(self,key,value):
        Exp = LCOE(self.filename, self.paneldatafile)
        with open(self.filename + '.json') as params:
            paramsDict = json.load(params)
            paramsDict[key][0] = value
            with open(self.filename + 'TEMP' + '.json', 'w') as outfile:
                json.dump(paramsDict,outfile)
        self.filenameTemp = self.filename + 'TEMP'
        Exp.Q = Que(self.filenameTemp, self.paneldatafile)
        PopKey = list(np.where(np.array(Exp.Q.key) == str(key)))[0][0]
        Exp.Q.value[PopKey] = np.arange(1,value+1,1)
        Exp.Q.filename = self.filename
        Exp.Q.GenFile()
        Exp.Q.SaveQue()
        os.remove(self.filenameTemp + '.json')
        return

    def Run(self):
        Exp = LCOE(self.filename, self.paneldatafile)
        Exp.GenerateJBS()
        Exp.LoadJBS()
        Exp.Run()
        return

    def ReRun(self):
        Exp = LCOE(self.filename, self.paneldatafile)
        Exp.LoadJBS()
        Exp.Run()
        return

if __name__ == '__main__':
    experiment = LCOERun('Experiments/RandomLocs/RandomLocs', 'Data/PanelData.csv')
    #experiment.ModPop('Tech','NoEnhancment')
    #experiment.Run()
    experiment.ReRun()
