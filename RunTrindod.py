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

    def GA_Load_Population(self, GApaneldatafile, tq, population, genes, bestcarryover, mutationrate, target, maxiter, lowvalues, highvalues):
        GAJB = GeneticAlgorithumJob(tq, population, genes, bestcarryover, mutationrate, target, maxiter)
        GAJB = GAJB.Load_Population(self.paneldatafile, lowvalues, highvalues)
        GAR = GeneticAlgorithum(GAJB, GApaneldatafile)
        GAR.Itterate()
        filename = time.strftime('%Y%m%d-%H%M%S')
        GAR.NamedPopulation.to_csv('Generations\\' + filename + '.csv')
        return

    def GA_Upscale_Population(self, GApaneldatafile, tq, population, genes, bestcarryover, mutationrate, target, maxiter, lowvalues, highvalues):
        GAJB = GeneticAlgorithumJob(tq, population, genes, bestcarryover, mutationrate, target, maxiter)
        GAJB = GAJB.Upscale_Population(self.paneldatafile, lowvalues, highvalues)
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
    #A = LCOERun('ResultSets/Presenation/Japan/Japan', 'Data/Initialpopulation.csv')
    #A.Run()
    #A.ReRun()
    #A.GA_Load_Population('Data/PanelData.csv', 'ResultSets/GA/Uk/GAMonthlyUk', 36, 5, 5, 0.15, 0, 5, [0,0,0,0.06,0.01], [25,1,1,0.245,392])
    Tests = ['ResultSets/Presenation/SouthAfrica/SouthAfrica']
    for Test in Tests:
        A = LCOERun(Test, 'Data/Initialpopulation.csv')
        A.GA_Upscale_Population('Data/PanelData.csv', Test, 499, 5, 2, 0.2, 0, 100, [0, 0, 0, 0.06, 0.01], [25, 1, 1, 0.245, 392])
       #A.GA_Random_Population('Data/PanelData.csv', 'ResultSets/GA/Uk/GAMonthlyUk', 500, 4, [0,0,0,0], [100,100,100,100], j, 0, 50)