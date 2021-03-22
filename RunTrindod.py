from GeneticAlgorithm import *
from Trindod import *
import time


class LCOERun:

    def __init__(self, filename, paneldatafile):
        self.filename = filename
        self.paneldatafile = paneldatafile

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

    def GA_Load_Population(self, GApaneldatafile, tq, population, genes, mutationrate, target, maxiter):
        GAJB = GeneticAlgorithumJob(tq, population, genes, mutationrate, target, maxiter)
        GAJB = GAJB.Load_Population(self.paneldatafile)
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

    def GA_Real_Population(self, GApaneldatafile, TrindodQue, NP, PB, PC, PM, target,maxiter):
        GAJB = GeneticAlgorithumJob(TrindodQue,NP,target,maxiter)
        GAJB = GAJB.Popultation_Mix(PB,PC,PM)
        GAJB = GAJB.Load_Population_New(self.paneldatafile)
        GAR = GeneticAlgorithum(GAJB, GApaneldatafile)
        GAR.Itterate_New()
        filename = time.strftime('%Y%m%d-%H%M%S')
        GAR.NamedPopulation.to_csv('Generations\\' + filename + '.csv')
        return

if __name__ == '__main__':
    A = LCOERun('ResultSets/InitialPop/Initialpopulation', 'Data/Initialpopulation.csv')
    #A.Run()
    #A.ReRun()
    #A.GA_Load_Population('Data/PanelData.csv', 'ResultSets/InitialPop/Initialpopulation', 500, 4, 0.05, 0, 5)
    A.GA_Real_Population('Data/PanelData.csv','ResultSets/InitialPop/Initialpopulation',37,10,14,13,0,10)
    #Test = [0.5]
    #for j in Test:
    #    print(j)
    #    A = LCOERun('ResultSets/GA/Uk/GAMonthlyUk', 'Data/PanelDataNonGA.csv')
    #    A.GA_Load_Population('Data/PanelData.csv', 'ResultSets/GA/Uk/GAMonthlyUk', 500, 4, j, 0, 200)
    #    #A.GA_Random_Population('Data/PanelData.csv', 'ResultSets/GA/Uk/GAMonthlyUk', 500, 4, [0,0,0,0], [100,100,100,100], j, 0, 50)