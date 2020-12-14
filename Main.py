from Job import *
from Panel import *
from EPC import *
from Finances import *
from Output import *
from multiprocessing import Pool
import multiprocessing
import pickle

# Worker fucnction for multiprocessing


def Worker(job):
    E = EPC(job)  # Economic costs calculated
    T = TechTime(job)  # Time & Timesteps calculated
    P = Panel(job)  # Solar panels setup
    P.Simulate(T)  # Solar panels simulated
    I = Inverter(job, T)  # Inverter setup
    I.Simulate()  # Inverter simulated
    F = Finance(job, E, T, P, I)  # Finacne model setup
    F.Costs()  # Finance model runs
    F.LCOECalculate()  # LCOE Calculated
    O = Out(job, E, T, P, I, F)  # Output object setup
    lock.acquire()  # Aquires lock for writing to file
    O.Results()  # Writes results to file
    lock.release()  # Releases lock
    return

# Initialise function for multiprocessing worker


def init(l):
    global lock  # lock global variable for multprocessing worker
    lock = l


if __name__ == '__main__':
    l = multiprocessing.Lock()
    JB = JobQue('RunQue.csv')  # Initialies job que object
    JB.LoadQue()  # Loads RunQue as job que object
    JB.LoadLoc()  # Loads locations in job que object
    JB.LoadPan()  # Loads panel in job que object
    JB.LoadTyp()  # Load panel type in job que object
    with open('RecentJobs.JBS', 'wb') as handle:
        pickle.dump(JB.Jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with Pool(processes=multiprocessing.cpu_count() - 1, initializer=init, initargs=(l,)) as pool:
        pool.map(Worker, JB.Jobs)
        pool.close()
        pool.join()
