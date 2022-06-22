import numpy as np

from Trindod import *
import os
import pandas as pd

class bulk_calc:
    def __init__(self, dir, name):
        path = os.path.join(os.getcwd(),dir)
        files = os.listdir(path)
        files = [os.path.join(path,file) for file in files]
        print(files)

        job = pd.read_json('job.json', typ='series')
        job['Yield'] = [15,35,101,128,145,137,137,119,92,55,16,8]
        job['PeakSunHours'] = [8.845333333,27.9,64.82933333,128.402,165.38,191.084,175.36,141.422,89.73466667,38.56,12.35866667,4.46]
        job['TimStp'] = 'hour'
        #job['PrjLif'] = 1
        job['Burn-in'] = 0
        job['Long-termDegradation'] = 0
        job['InvLif']
        job = job.to_dict()
        print(job)

        lcoe = np.zeros(len(files))
        for idx, file in enumerate(files):
            Cell_area = 0.98  # 6.00005025e-6
            ITO_Desnity = 7.14 * 100
            PEDOTPSS_Density = 1.011 * 100
            P3HTPCBM_Density = 1.3 * 100
            Al_Density = 2.7 * 100

            ITO_Volume = Cell_area * 1e-7
            PEDOTPSS_Volume = Cell_area * 1e-7
            P3HTPCBM_Volume = Cell_area * 2.2e-7
            Al_Volume = Cell_area * 1e-7

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
            job['Cost'] = Cost

            job['PowerDensity'] = 100
            data = pd.read_csv(file)
            data = data['Energy'].values
            temp = np.zeros(int((len(data)/2)))
            for i in range(len(data[1:])):
                temp[int(i/2)] = np.sum(data[i:i+1])

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
            lcoe[idx] = F.LCOE



bulk_calc('PM6Y6_200_Power','name')