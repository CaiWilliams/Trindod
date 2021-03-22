from Trindod import *


class GeneticAlgorithum:

    def __init__(self, job, paneldata):
        self.TrindodQue = job.TrindodQue
        self.paneldata = paneldata
        self.Population = job.Population
        self.LowLimits = job.LowLimits
        self.HighLimits = job.HighLimits
        self.Target = job.Target
        self.MaxIter = job.MaxIter
        self.Children = job.Children
        self.Genes = job.Genes
        self.GenesMean = job.GenesMean
        self.GenesSD = job.GenesSD
        self.NBest = job.NBest
        self.NCrossover = job.NCrossover
        self.NMutation = job.NMutation
        self.Generation = 0
        self.T = LCOE(self.TrindodQue, self.paneldata)
    
    def Save_Popultaion_To_Obj(self):
        Gen = 'G' + str(self.Generation)
        self.Children = np.random.randint(0, 0xFFFFFF, len(self.Population))
        vhex = np.vectorize(hex)
        self.Children = vhex(self.Children)
        if self.Generation == 0:
            self.NamedPopulation = pd.DataFrame(self.Population, index=self.Children)
            self.NamedPopulation['Generation'] = Gen
            self.NamedPopulation['Fitness'] = self.Fitness
            self.NamedPopulation['Results'] = self.Results
        else:
            NamedPopulation = pd.DataFrame(self.Population, index=self.Children)
            NamedPopulation['Generation'] = Gen
            NamedPopulation['Fitness'] = self.Fitness
            NamedPopulation['Results'] = self.Results
            self.NamedPopulation = self.NamedPopulation.append(NamedPopulation)
        return

    def Save_Popultaion_To_File(self):
        Gen = 'G' + str(self.Generation)
        self.Children = np.random.randint(0, 0xFFFFFF, len(self.Population))
        vhex = np.vectorize(hex)
        self.Children = vhex(self.Children)

        if self.Generation == 0:
            NamedPopulation = pd.DataFrame(self.Population, index=self.Children)
            NamedPopulation['Mothes'] = np.zeros(len(self.Population))
            NamedPopulation['Fathers'] = np.zeros(len(self.Population))

        else:
            self.Mothers = self.PreviousName[self.Mothers]
            self.Fathers = self.PreviousName[self.Fathers]
            NamedPopulation = pd.DataFrame(self.Population, index=self.Children)
            NamedPopulation['Mothes'] = self.Mothers
            NamedPopulation['Fathers'] = self.Fathers

        NamedPopulation['Fitness'] = self.Fitness
        NamedPopulation['Results'] = self.Results
        NamedPopulation.to_csv('Generations\\' + Gen + '.csv')
        return
    
    def Save_PanelData(self):
        PanelData = pd.DataFrame()
        PanelData['PanelID'] = np.arange(0, len(self.Population), 1, dtype='int')+1
        PanelData['Type'] = self.Children
        PanelData['Life'] = self.Population[:, 0]
        PanelData['Burn-in'] = self.Population[:, 1]
        PanelData['Long-termDegradation'] = self.Population[:, 2]
        PanelData['Burn-inPeakSunHours'] = 500
        PanelData['Cost'] = 0.245
        PanelData['PowerDensity'] = self.Population[:, 3]
        PanelData['EnergyEfficiency'] = self.Population[:, 3] / 9.8 / 100
        PanelData.to_csv(self.paneldata, index=False)
        return

    def Save_PanelData_New(self):
        PanelData = pd.DataFrame()
        PanelData['PanelID'] = np.arange(0, len(self.Population), 1, dtype='int')+1
        PanelData['Type'] = self.Children
        PanelData['Life'] = self.BestLife
        PanelData['Burn-in'] = self.Population[:, self.Genes.index('Burn-in')]
        PanelData['Long-termDegradation'] = self.Population[:, self.Genes.index('Long-termDegradation')]
        PanelData['Burn-inPeakSunHours'] = 500
        PanelData['Cost'] = 0.245
        PanelData['PowerDensity'] = self.Population[:, self.Genes.index('PowerDensity')]
        PanelData['EnergyEfficiency'] = self.Population[:, self.Genes.index('PowerDensity')] / 9.8 / 100
        PanelData.to_csv(self.paneldata, index=False)
        return

    def Breed_Population(self):
        self.Mothers = np.random.choice(range(len(self.Population)), int(len(self.Population)), p=self.Fitness)
        self.Fathers = np.random.choice(range(len(self.Population)), int(len(self.Population)), p=self.Fitness)

        NP = np.zeros(np.shape(self.Population))
        i = 0
        CrosoverPoints = np.random.randint(0, self.Genes, len(self.Population))
        for NPp, P1, P2, CrosoverPoint in zip(NP, self.Mothers, self.Fathers, CrosoverPoints):
            NPp[:CrosoverPoint] = self.Population[P1, :CrosoverPoint]
            NPp[CrosoverPoint:] = self.Population[P2, CrosoverPoint:]
            NP[i] = NPp
            i = i + 1
        self.Population = NP
        return
    
    def Mutate_Population(self):
        for idx, Pop in enumerate(self.Population):
            Mutation = np.random.uniform(0, 1)
            if Mutation > 1 - self.MutationRate:
                MutatedGene = np.random.randint(0, self.Genes)
                NewGene = np.random.uniform(self.LowLimits[MutatedGene], self.HighLimits[MutatedGene])
                Pop[MutatedGene] = NewGene
                self.Population[idx] = Pop
        return 

    def Test_Population(self):
        self.Fitness = np.zeros(len(self.Population))
        self.Fitness[:] = np.abs(self.Results[:] - self.Target)
        Finf = np.where(self.Fitness == np.inf)
        self.Fitness[Finf] = 100000000000000
        Fmax = np.amax(self.Fitness)
        Fmin = np.amin(self.Fitness)
        self.Fitness[:] = (Fmax - self.Fitness[:])/(Fmax - Fmin)
        self.Fitness[:] = self.Fitness[:] / self.Fitness.sum()
        return 

    def Best_Life(self):
        Lifetimes = [2]
        PrevResults = np.ones(len(self.Population)) * 1000000
        BestLife = np.ones(len(self.Population)) * 2
        for Life in Lifetimes:
            LifeD = {"Life":Life}
            for Job in self.T.Q.Jobs:
                Job.update(LifeD)
            self.T.Run()
            Results = self.T.FetchReults()
            infs = np.where(Results == np.inf)
            Results[infs] = 1000000
            Improved = np.where(Results <= PrevResults)
            BestLife[Improved] = Life
            PrevResults = Results
            for idx,Job in enumerate(self.T.Q.Jobs):
                LifeD = {'Life':BestLife[idx]}
                Job.update(LifeD)
        self.BestLife = BestLife
        return

    def Population_Best(self):
        self.NewPopulation = np.zeros(np.shape(self.Population))
        Best = self.GNP[0:self.NBest].index
        self.NewPopulation[0:self.NBest] = self.Population[Best]
        return

    def Population_Crossover(self):
        self.Mothers = np.random.choice(range(len(self.Population)), int(self.NCrossover), p=self.Fitness)
        self.Fathers = np.random.choice(range(len(self.Population)), int(self.NCrossover), p=self.Fitness)
        NP = np.zeros((self.NCrossover,len(self.Genes)))
        i = 0
        CrosoverPoints = np.random.randint(0, len(self.Genes)+1, self.NCrossover)
        for NPp, P1, P2, CrosoverPoint in zip(NP, self.Mothers, self.Fathers, CrosoverPoints):
            NPp[:CrosoverPoint] = self.Population[P1, :CrosoverPoint]
            NPp[CrosoverPoint:] = self.Population[P2, CrosoverPoint:]
            NP[i] = NPp
            i = i + 1
        NP = self.Mutation_New(NP)
        self.NewPopulation[self.NBest:self.NBest+self.NCrossover] = NP
        return

    def Population_Mutation(self):
        NP = np.random.randint(0, len(self.Population), size=self.NMutation)
        NP = self.Population[NP]
        NP = self.Mutation_New(NP)
        self.NewPopulation[self.NBest+self.NCrossover:] = NP
        return

    def Mutation_New(self, P):
        for Gene in range(len(self.Genes)):
            P[:,Gene] = np.abs(np.random.normal(self.GenesMean[Gene], self.GenesSD[Gene], len(P)))
        return P

    def Itterate_New(self):
        self.T.LoadJBS()
        while self.Generation < self.MaxIter:
            self.Best_Life()
            self.Save_PanelData_New()
            self.T.Q.LoadPan2()
            self.T.Run()
            self.Results = self.T.FetchReults()
            self.Test_Population()
            self.GNP = pd.DataFrame()
            self.GNP['Name'] = self.Children
            self.GNP['Results'] = self.Results
            self.GNP['Fitness'] = self.Fitness
            self.GNP = self.GNP.sort_values(by=['Fitness'],ascending=False)
            self.Save_Popultaion_To_Obj()
            self.PreviousName = self.Children
            self.Population_Best()
            self.Population_Crossover()
            self.Population_Mutation()
            self.Population = self.NewPopulation
            self.Generation = self.Generation + 1
        return

    def Itterate(self):
        self.T.LoadJBS()
        while self.Generation < self.MaxIter:
            print(self.Generation)
            self.Save_PanelData()
            self.T.Q.LoadPan2()
            self.T.Run()
            self.Results = self.T.FetchReults()
            self.Test_Population()
            self.Save_Popultaion_To_Obj()
            self.PreviousName = self.Children
            self.Breed_Population()
            self.Mutate_Population()
            self.Generation = self.Generation + 1
        return


class GeneticAlgorithumJob:

    def __init__(self, tq, Np, target, maxiter):
        self.TrindodQue = tq
        self.Population = Np
        print(self.Population)
        self.Target = target
        self.MaxIter = maxiter
        self.Children = np.random.randint(0, 0xFFFFFF, self.Population)
        vhex = np.vectorize(hex)
        self.Children = vhex(self.Children)

    def Popultation_Mix(self, PB, PC, PM):
        self.NBest = PB
        self.NCrossover = PC
        self.NMutation = PM
        return self

    def Load_Population_New(self, filepath):
        self.FilePath = filepath
        PanelData = pd.read_csv(self.FilePath, index_col=False)
        PanelData = PanelData.iloc[0:self.Population]
        PanelData = PanelData[['PowerDensity', 'Long-termDegradation', 'Burn-in']]
        self.Genes = PanelData.columns.tolist()
        self.GenesMean = PanelData.mean(axis=0).tolist()
        self.GenesSD = PanelData.std(axis=0).tolist()
        PanelData = PanelData.to_numpy()
        self.Population = PanelData
        self.LowLimits = np.amin(PanelData, axis=0)
        self.HighLimits = np.amax(PanelData, axis=0)
        return self

    def Random_Popultaion(self, lowlimits, highlimits):
        self.LowLimits = lowlimits
        self.HighLimits = highlimits
        self.Create_Population()
        return self

    def Load_Population(self, filepath):
        self.FilePath = filepath
        PanelData = pd.read_csv(self.FilePath, index_col=False)
        PanelData = PanelData.iloc[0:self.Population]
        PanelData = PanelData.drop(columns=['PanelID', 'Type', 'Burn-inPeakSunHours', 'Cost', 'EnergyEfficiency'])
        PanelData = PanelData.to_numpy()
        self.Population = PanelData
        self.LowLimits = np.amin(PanelData, axis=0)
        self.HighLimits = np.amax(PanelData, axis=0)
        return self

    def Create_Population(self):
        Pop = np.zeros((self.Population, self.Genes))
        for Gene in range(self.Genes):
            Pop[:, Gene:Gene+1] = np.random.uniform(self.LowLimits[Gene], self.HighLimits[Gene], (self.Population, 1))
        self.Population = Pop
