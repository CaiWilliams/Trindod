import numpy as np
#import matplotlib.pyplot as plt
import pickle


class Population:

    def __init__(self, popultaion_number, mutate = 0.2):
        self.mutate = mutate
        self.population_number = popultaion_number
        self.population = np.zeros(popultaion_number,dtype=object)

    def add_members(self, members):
        if type(members) != list:
            members = [members]
        adding_index = np.where(self.population == 0)[0]
        members_index = np.arange(0,len(members))
        adding_index = adding_index[members_index]
        self.population[adding_index] = members

    def fill_population(self, chromosome_length):
        self.chromosome_length = chromosome_length
        self.limits = np.empty((chromosome_length,2))
        self.limits[:] = np.nan
        self.population[:] = [Member(chromosome_length) for idx in self.population]
        for m in self.population:
            m.set_chromosome_random(range(chromosome_length), 0, 1)

    def set_limits(self,chromosome_number, min, max):
        self.limits[chromosome_number][0] = min
        self.limits[chromosome_number][1] = max
        for m in self.population:
            m.set_chromosome_random(chromosome_number,min,max)

    def rank_population(self, objective_funciton):
        self.objective_funciton = objective_funciton
        self.result = objective_funciton.function(self.population)
        self.rank = np.zeros(len(self.result))
        self.rank[:] = np.argsort(np.abs(self.result[:] - objective_funciton.target))
        #self.rank = np.argsort([np.abs(m - objective_funciton.target) for m in self.result])
        self.error = [np.abs(m - objective_funciton.target)/objective_funciton.target for m in self.result]

    def best_in_population(self, n):
        self.best_n_idx = np.argsort(self.rank)[0:n]
        return self.population[self.best_n_idx]

    def worst_in_population(self, n):
        self.worst_n_idx = np.argsort(self.rank)[-n:]

    def breed(self, n):
        breeding_pool = np.delete(self.population, self.worst_n_idx)
        breeding_pool_results = np.delete(self.result[:], self.worst_n_idx)
        breeding_pool_rank = np.argsort(np.abs(breeding_pool_results - self.objective_funciton.target))
        breeding_pool[:] = breeding_pool[breeding_pool_rank]
        breeding_pool_weight = np.arange(1,0,-1/len(breeding_pool))
        breeding_pool_weight = breeding_pool_weight/sum(breeding_pool_weight)
        mothers = np.random.choice(breeding_pool,int(len(breeding_pool)),p=breeding_pool_weight)
        fathers = np.random.choice(breeding_pool,int(len(breeding_pool)),p=breeding_pool_weight)
        crossover_point = np.random.randint(0,self.chromosome_length-1,len(mothers))
        x = zip(mothers,fathers,crossover_point)
        next_generation = np.zeros(len(breeding_pool),dtype=object)
        for idx,x in enumerate(x):
            next_generation[idx] = Member(self.chromosome_length).set_chromosome_parents(x[0], x[1], x[2])
            next_generation[idx] = next_generation[idx].mutate(self.mutate,self.limits)
        return next_generation

    def next_generation(self, bn):
        TNG = np.zeros(self.population_number, dtype=object)
        TNG[:bn] = self.best_in_population(bn)
        TNG[bn:] = self.breed(self.population_number - bn)
        self.population = TNG

    def save(self,filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)



class Member:

    def __init__(self, chromosomes_length):
        self.chromosomes_length = chromosomes_length
        self.chromosomes = np.zeros(self.chromosomes_length,dtype=float)

    def set_chromosome_random(self, chromo_num, min, max):
        if type(chromo_num) != list:
            chromo_num = [chromo_num]
        if np.max(chromo_num) > self.chromosomes_length-1:
            return print("Chromosome outside of defined length")
        else:
            self.chromosomes[chromo_num] = np.random.uniform(min,max,len(chromo_num))

    def set_chromosome_parents(self, mother, father, crossover):
        self.chromosomes[:crossover] = mother.chromosomes[:crossover]
        self.chromosomes[crossover:] = father.chromosomes[crossover:]
        return self

    def mutate(self, p, limits):
        m_p = np.random.uniform(0,1)
        if p > m_p:
            m_c = np.random.randint(0, self.chromosomes_length-1)
            #if np.isnan(limits[m_c]).any():
            #    limits[m_c][0] = 0
            #    limits[m_c][1] = 1
            self.chromosomes[m_c] = np.random.uniform(low=limits[m_c][0], high=limits[m_c][1])
        for i in range(len(self.chromosomes)):
            if self.chromosomes[i] > limits[i][1]:
                self.chromosomes[i] = limits[i][1]
            if self.chromosomes[i] < limits[i][0]:
                self.chromosomes[i] = limits[0]
        return self


class Objective_Function:

    def __init__(self, target, function):
        self.target = target
        self.function = function

