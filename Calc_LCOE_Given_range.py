import natsort
import numpy as np

from Trindod import *
import os
import pandas as pd
import natsort
import dill
from GA import *
import copy
import multiprocessing as mp
import tqdm
import itertools

class bulk_calc:
    def __init__(self, dir,exp_name,name,pce_min,pce_max,cost_min, cost_max):
        pathp = os.path.join(os.getcwd(), dir+'_Power')
        filesp = natsort.natsorted(os.listdir(pathp))
        filesp= [os.path.join(pathp,file) for file in filesp]

        device_num = int(dir.split('_')[-1])
        pathtt = os.path.join(os.getcwd(),exp_name,'OnGoing_Experiment_'+str(device_num)+'.exp')


        job = pd.read_json('job.json', typ='series')
        job['Yield'] = [15,35,101,128,145,137,137,119,92,55,16,8]
        job['PeakSunHours'] = [8.845333333,27.9,64.82933333,128.402,165.38,191.084,175.36,141.422,89.73466667,38.56,12.35866667,4.46]
        job['TimStp'] = 'hour'
        #job['PrjLif'] = 1
        job['Burn-in'] = 0
        job['Long-termDegradation'] = 0
        job['InvLif']
        job = job.to_dict()

        lcoe = np.zeros(len(filesp))
        with open(pathtt,'rb') as handle:
                datat = dill.load(handle)
                datat = datat[datat != 0]
                datat = datat[-1]
        Res_og = datat.result[0, 1]
        Data = pd.read_csv(filesp[0])
        DataE = Data['Energy'].values
        DataE_og = np.sum(DataE)


        Cost  = self.cost(datat)
        Cost = np.round(np.arange(cost_min, cost_max+1e-5, 1e-5), 5)
        Res = np.round(np.arange(pce_min, pce_max+0.1, 0.1), 3)
        CostRes = list(itertools.product(Cost,Res))
        CostRes = np.asarray([list(cr) for cr in CostRes])
        print(CostRes)
        print(len(CostRes))
        #DataE = np.zeros((len(CostRes),len(Data)))
        DataE_Temp = np.ones(len(Data)) * (DataE_og/len(Data))
        DataE = [DataE_Temp * (costres[1]/Res_og) for costres in CostRes]
        vfunc = np.vectorize(bulk_calc.create_job)
        jobs = vfunc(job,CostRes[:,0],CostRes[:,1])
        with mp.Pool(mp.cpu_count() - 1) as p:
            lcoe = list(tqdm.tqdm(p.imap(bulk_calc.worker,zip(jobs,DataE)), total=len(jobs)))
        data = pd.DataFrame(lcoe)
        data.to_csv(name+'.csv')

    def cost(self, datat):
        temp = datat.population[:]
        Cell_area = 0.98  # 6.00005025e-6
        ITO_Desnity = 7.14 * 100
        PEDOTPSS_Density = 1.011 * 100
        P3HTPCBM_Density = 1.3 * 100
        Al_Density = 2.7 * 100

        vfunc = np.vectorize(self.chromo_value)
        ITO_Volume = Cell_area * vfunc(temp,0)
        PEDOTPSS_Volume = Cell_area * vfunc(temp,1)
        P3HTPCBM_Volume = Cell_area * vfunc(temp,2)
        Al_Volume = Cell_area * vfunc(temp,3)

        ITO_mass = ITO_Volume * ITO_Desnity
        PEDOTPSS_mass = PEDOTPSS_Volume * PEDOTPSS_Density
        P3HTPCBM_mass = P3HTPCBM_Volume * P3HTPCBM_Density
        Al_mass = Al_Volume * Al_Density

        ITO_CostPerg = 28.68
        PEDOTPSS_CostPerg = 7.08
        P3HTPCBM_CostPerg = 2
        Al_CostPerg = 0.233

        ITO_Cost = ITO_CostPerg * ITO_mass
        PEDOTPSS_Cost = PEDOTPSS_CostPerg * PEDOTPSS_mass
        P3HTPCBM_Cost = P3HTPCBM_CostPerg * P3HTPCBM_mass
        Al_Cost = Al_CostPerg * Al_mass
        Cost = ITO_Cost + PEDOTPSS_Cost + P3HTPCBM_Cost + Al_Cost
        return Cost

    def chromo_value(self, f, val):
        return f.chromosomes[val]

    @staticmethod
    def create_job(job, cost, res):
        job = copy.deepcopy(job)
        job['Cost'] = cost
        pce = res / 100
        job['EneryEfficiency'] = pce
        job['PowerDensity'] = pce * 980
        return job

    @staticmethod
    def worker(z):
        z = list(z)
        job = z[0]
        data = z[1]
        temp = np.zeros(int((len(data) / 2)))
        for i in range(len(data[1:])):
            temp[int(i / 2)] = np.sum(data[i:i + 1])
        E = EPC(job)
        t = TechTime(job)
        P = Panel(job)
        P.simulate(t)
        temp = np.asarray([temp[i % len(temp)] for i in range(len(P.Dates))])
        P.PVGen = temp[:] * 5.492475
        I = Inverter(job, t)
        I.simulate()
        F = Finance(job, E, t, P, I)
        F.costs()
        F.lcoe_calculate()
        return F.LCOE

class PCE_COST:
    pass
   
class PCE:
	pass

#bulk_calc('PM6Y6_1','PM6Y6_PCECOST','PM6Y6_1_LCOE')
#bulk_calc('PM6Y6_2','PM6Y6_PCECOST','PM6Y6_2_LCOE')
#bulk_calc('PM6Y6_4','PM6Y6_PCECOST','PM6Y6_4_LCOE')
#bulk_calc('PM6Y6_6','PM6Y6_PCECOST','PM6Y6_6_LCOE')
#bulk_calc('PM6Y6_8','PM6Y6_PCECOST','PM6Y6_8_LCOE')

#bulk_calc('PM6Y6_10','PM6Y6_PCECOST','PM6Y6_10_LCOE')
#bulk_calc('PM6Y6_20','PM6Y6_PCECOST','PM6Y6_20_LCOE')
#bulk_calc('PM6Y6_40','PM6Y6_PCECOST','PM6Y6_40_LCOE')
#bulk_calc('PM6Y6_60','PM6Y6_PCECOST','PM6Y6_60_LCOE')
#bulk_calc('PM6Y6_80','PM6Y6_PCECOST','PM6Y6_80_LCOE')

#bulk_calc('PM6Y6_100','PM6Y6_PCECOST','PM6Y6_100_LCOE')
if __name__ == '__main__':
    bulk_calc('PM6Y6_200','PM6Y6_PCECOST','Tabulated_LCOE',0.001,35,0.00001,0.3)
#bulk_calc('PM6Y6_400','PM6Y6_PCECOST','PM6Y6_400_LCOE')
#bulk_calc('PM6Y6_600','PM6Y6_PCECOST','PM6Y6_600_LCOE')
#bulk_calc('PM6Y6_800','PM6Y6_PCECOST','PM6Y6_800_LCOE')
#bulk_calc('PM6Y6_1000','PM6Y6_PCECOST','PM6Y6_1000_LCOE')

#bulk_calc('P3HTPCBM_1','P3HTPCBM_PCECOST','P3HTPCBM_1_LCOE')
#bulk_calc('P3HTPCBM_2','P3HTPCBM_PCECOST','P3HTPCBM_2_LCOE')
#bulk_calc('P3HTPCBM_4','P3HTPCBM_PCECOST','P3HTPCBM_4_LCOE')
#bulk_calc('P3HTPCBM_6','P3HTPCBM_PCECOST','P3HTPCBM_6_LCOE')
#bulk_calc('P3HTPCBM_8','P3HTPCBM_PCECOST','P3HTPCBM_8_LCOE')

#bulk_calc('P3HTPCBM_10','P3HTPCBM_PCECOST','P3HTPCBM_10_LCOE')
#bulk_calc('P3HTPCBM_20','P3HTPCBM_PCECOST','P3HTPCBM_20_LCOE')
#bulk_calc('P3HTPCBM_40','P3HTPCBM_PCECOST','P3HTPCBM_40_LCOE')
#bulk_calc('P3HTPCBM_60','P3HTPCBM_PCECOST','P3HTPCBM_60_LCOE')
#bulk_calc('P3HTPCBM_80','P3HTPCBM_PCECOST','P3HTPCBM_80_LCOE')

#bulk_calc('P3HTPCBM_100','P3HTPCBM_PCECOST','P3HTPCBM_100_LCOE')
#bulk_calc('P3HTPCBM_200','P3HTPCBM_PCECOST','P3HTPCBM_200_LCOE')
#bulk_calc('P3HTPCBM_400','P3HTPCBM_PCECOST','P3HTPCBM_400_LCOE')
#bulk_calc('P3HTPCBM_600','P3HTPCBM_PCECOST','P3HTPCBM_600_LCOE')
#bulk_calc('P3HTPCBM_800','P3HTPCBM_PCECOST','P3HTPCBM_800_LCOE')
#bulk_calc('P3HTPCBM_1000','P3HTPCBM_PCECOST','P3HTPCBM_1000_LCOE')

#bulk_calc('PM6Y6_04','PM6Y6_PCECOST','PM6Y6_04_LCOE')
#bulk_calc('P3HTPCBM_03','P3HTPCBM_PCECOST','P3HTPCBM_03_LCOE')

