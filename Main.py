from Job import *
from Panel import *
from EPC import *
from Finances import *
from Output import *
import time
import cProfile
from multiprocessing import Pool
import multiprocessing
from matplotlib import pyplot as plt


def Worker(Job):
    E = EPC(Job)
    T = TechTime(Job)
    P = Panel(Job,E)
    P.Simulate(T)
    I = Inverter(Job,T)
    I.Simulate()
    F = Finance(Job,E,T,P,I)
    F.Costs()
    F.LCOECalculate()
    O = Out(Job,E,T,P,I,F)
    O.Results()
    return

if __name__ == '__main__':
    JB = JobQue('RunQue.csv')
    JB.LoadQue()
    JB.LoadLoc()
    JB.LoadPan()
    JB.LoadTyp()
    print(JB.Jobs[0])
    with Pool(processes=multiprocessing.cpu_count()-1) as pool:
        pool.map(Worker,JB.Jobs)
