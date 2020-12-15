from Job import *
from Panel import *
from EPC import *
from Finances import *
from Output import *
from multiprocessing import Pool
import multiprocessing


def Worker(job):
    E = EPC(job)
    t = TechTime(job)
    P = Panel(job)
    P.Simulate(t)
    I = Inverter(job, t)
    I.Simulate()
    F = Finance(job, E, t, P, I)
    F.Costs()
    F.LCOECalculate()
    O = Out(job, E, t, P, I, F)
    lock.acquire()
    O.Results()
    lock.release()
    return


def init(l):
    global lock
    lock = l


if __name__ == '__main__':
    l = multiprocessing.Lock()
    JB = JobQue('RunQue.csv')
    JB.ReRun('RecentJobs.JBS')
    Devices = [4107]
    Variations = ['NoEnhancment']
    for Device in Devices:
        for Variation in Variations:
            JB.Modify('Tech', Variation)
            JB.Modify('PanTyp', Device)
            JB.LoadPan()
            with Pool(processes=multiprocessing.cpu_count() - 1, initializer=init, initargs=(l,)) as pool:
                pool.map(Worker, JB.Jobs)
                pool.close()
                pool.join()

