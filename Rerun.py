from Job import *
from Panel import *
from EPC import *
from Finances import *
from Output import *
import time
import cProfile
from multiprocessing import Pool
import multiprocessing
import pickle



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
    lock.acquire()
    O.Results()
    lock.release()
    return

def init(l):
    global lock
    lock = l

S = time.time()
if __name__ == '__main__':
    l = multiprocessing.Lock()
    JB = JobQue('RunQue.csv')
    JB.ReRun('RecentJobs.JBS')
    Devices = [4107,4595]
    Variations = ['NoEnhancment']
    for Device in Devices:
        for Variation in Variations:
            JB.Modify('Tech',Variation)
            JB.Modify('PanTyp',Device)
            JB.LoadPan()
            with Pool(processes=multiprocessing.cpu_count()-1, initializer=init, initargs=(l,)) as pool:
                pool.map(Worker,JB.Jobs)
                pool.close()
                pool.join()

D = S - time.time()
