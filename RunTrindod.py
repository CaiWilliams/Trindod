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

if __name__ == '__main__':
    A = LCOERun('ResultSets/GA/PortElizabeth/GAMonthlyPortElizabeth', 'Data/PanelDataNonGA.csv')
    #A.Run()
    #A.ReRun()
    A.GA_Load_Population('Data/PanelData.csv', 'ResultSets/GA/PortElizabeth/GAMonthlyPortElizabeth', 500, 4, 0.05, 0, 100)
    #A.GA_Random_Population('Data/PanelData.csv', 'ResultSets/GA/PortElizabeth/GAMonthlyPortElizabeth', 500, 5, [0,0,0,0,0], [100,100,100,100,100], 0.01, 0, 10)