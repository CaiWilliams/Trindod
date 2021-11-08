from GeneticAlgorithm import *
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

    def RunWithLCCA(self):
        Exp = LCOE(self.filename, self.paneldatafile)
        Exp.GenerateJBS()
        Exp.LoadJBS()
        Exp.RunLCCA()
        return

    def ReRunWithLCCA(self):
        Exp = LCOE(self.filename, self.paneldatafile)
        Exp.LoadJBS()
        Exp.RunLCCA()
        return

    def GA_Load_Population(self, GApaneldatafile, tq, population, genes, lifetimes, bestcarryover, mutationrate, target, maxiter, lowvalues, highvalues):
        GAJB = GeneticAlgorithumJob(tq, population, genes, bestcarryover, mutationrate, target, maxiter)
        GAJB = GAJB.Load_Population(self.paneldatafile, lowvalues, highvalues, lifetimes)
        GAR = GeneticAlgorithum(GAJB, GApaneldatafile)
        GAR.Itterate()
        filename = time.strftime('%Y%m%d-%H%M%S')
        GAR.NamedPopulation.to_csv('Generations\\' + filename + '.csv')
        return

    def GA_Upscale_Population(self, GApaneldatafile, tq, population, genes, lifetimes, bestcarryover, mutationrate, target, maxiter, lowvalues, highvalues, CostMean=0.37875, CostStd=0.0559):
        self.ModPop('PanTyp',population)
        GAJB = GeneticAlgorithumJob(tq, population, genes, bestcarryover, mutationrate, target, maxiter, CostMean, CostStd)
        GAJB = GAJB.Upscale_Population(self.paneldatafile, lowvalues, highvalues, lifetimes)
        GAR = GeneticAlgorithum(GAJB, GApaneldatafile)
        GAR.Itterate()
        filename = time.strftime('%Y%m%d-%H%M%S')
        GAR.NamedPopulation.to_csv('Generations\\' + filename + '.csv')
        return

    def GA_Random_Population(self, GApaneldatafile, tq, population, genes, lowlimits, highlimits, mutationrate, target, maxiter):
        GAJB = GeneticAlgorithumJob(tq, population, genes, mutationrate, target, maxiter)
        GAJB = GAJB.Random_Popultaion(lowlimits, highlimits)
        GAR = GeneticAlgorithum(GAJB, GApaneldatafile)
        GAR.Itterate()
        filename = time.strftime('%Y%m%d-%H%M%S')
        GAR.NamedPopulation.to_csv('Generations\\' + filename + '.csv')
        return

if __name__ == '__main__':
    A = LCOERun('ResultSets/RandomPoints/RandomPoints', 'Data/PanelDataNonGA2.csv')
    #A.ModPop('PanTyp',4110)
    #A.Run()
    A.ReRunWithLCCA()
    #A.GA_Load_Population('Data/PanelData.csv', 'ResultSets/GA/Uk/GAMonthlyUk', 36, 5, 5, 0.15, 0, 5, [0,0,0,0.06,0.01], [25,1,1,0.245,392])
    #Tests = ['ResultSets/Presenation/Australia/Australia','ResultSets/Presenation/Brazil/Brazil','ResultSets/Presenation/India/India','ResultSets/Presenation/Japan/Japan','ResultSets/Presenation/SouthAfrica/SouthAfrica','ResultSets/Presenation/Spain/Spain','ResultSets/Presenation/UK/UK','ResultSets/Presenation/USA/USA']
    #for Test in Tests:
    #    A = LCOERun(Test, 'Data/Initialpopulation.csv')
    ##    #A.GA_Load_Population('Data/PanelData.csv', Test, 36, 5, [1,2,5,10,25], 5, 0.15, 0, 5,[0, 0, 0, 0.06, 0.01], [25, 1, 1, 0.245, 392])
    #    A.GA_Upscale_Population('Data/PanelData.csv', Test, 100, 5, [2,5,10,25], 50, 0.25, 0, 50, [0, 0, 0, 0.06, 0.01], [25, 1, 1, 0.7, 392], 0.37875, 0.559)
    #   #A.GA_Random_Population('Data/PanelData.csv', 'ResultSets/GA/Uk/GAMonthlyUk', 500, 4, [0,0,0,0], [100,100,100,100], j, 0, 50)