from Job import *
from Panel import *
from EPC import *
from Finances import *
from Output import *
from multiprocessing import Pool
import multiprocessing
import tqdm


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


def init(l,):
    global lock
    lock = l


if __name__ == '__main__':
    l = multiprocessing.Lock()
    JB = JobQue('Revised.csv')
    JB.ReRun('Revised.JBS')
    Devices = [4882,4883,4884,4885,4886,4887,4888,4889,4890,4891]
    Variations = ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10']
    with tqdm.tqdm(total=(len(JB.Jobs)*len(Devices))) as pbar:
        for idx, Device in enumerate(Devices):
            JB.Modify('Tech', Variations[idx])
            JB.Modify('PanTyp', Device)
            JB.LoadPan()
            with Pool(processes=multiprocessing.cpu_count() - 1, initializer=init, initargs=(l,)) as pool:
                for i,_ in enumerate(pool.imap_unordered(Worker, JB.Jobs)):
                    pbar.update()
                #pool.map(Worker, JB.Jobs)
                pool.close()
                pool.join()

