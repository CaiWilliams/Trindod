import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Results:
    def __init__(self, results, **paramaters):
        self.Categories = paramaters
        self.Categories.update(Results='Results', Fitness='Fitness')
        self.Results = pd.read_csv(results)
        self.Results = self.Results.replace([np.inf, -np.inf], np.nan)
        self.Results = self.Results.dropna()
        self.Results['Generation'] = self.Results['Generation'].str[1:]

    def PlotOne(self, cat_name, xname, yname):

        YAverage = self.Results.groupby(['Generation']).mean()[cat_name].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()[cat_name].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()[cat_name].sort_values(ascending=False).to_numpy()
        X = pd.unique(self.Results['Generation'])
        X = range(len(X))
        plt.plot(X, YAverage)
        plt.fill_between(X, YMin, YMax, color='b', alpha=.1)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.show()
        return

    def PlotAll(self):
        fig, ax = plt.subplots(ncols=2, nrows=4)

        X = range(len(pd.unique(self.Results['Generation'])))

        YAverage = self.Results.groupby(['Generation']).mean()['0'].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()['0'].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()['0'].sort_values(ascending=False).to_numpy()
        ax[0, 0].plot(X,YAverage)
        ax[0, 0].fill_between(X, YMin, YMax, color='b', alpha=.1)
        ax[0, 0].set(ylabel='Life')


        YAverage = self.Results.groupby(['Generation']).mean()['1'].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()['1'].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()['1'].sort_values(ascending=False).to_numpy()
        ax[1, 0].plot(X, YAverage)
        ax[1, 0].fill_between(X, YMin, YMax, color='b', alpha=.1)
        ax[1, 0].set(ylabel='Burn-in')

        YAverage = self.Results.groupby(['Generation']).mean()['2'].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()['2'].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()['2'].sort_values(ascending=False).to_numpy()
        ax[2, 0].plot(X, YAverage)
        ax[2, 0].fill_between(X, YMin, YMax, color='b', alpha=.1)
        ax[2, 0].set(ylabel='Long-termDegradation')

        YAverage = self.Results.groupby(['Generation']).mean()['3'].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()['3'].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()['3'].sort_values(ascending=False).to_numpy()
        ax[3, 0].plot(X, YAverage)
        ax[3, 0].fill_between(X, YMin, YMax, color='b', alpha=.1)
        ax[3, 0].set(xlabel='Generation', ylabel='PowerDensity')

        YAverage = self.Results.groupby(['Generation']).mean()['Fitness'].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()['Fitness'].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()['Fitness'].sort_values(ascending=False).to_numpy()
        ax[1, 1].plot(X, YAverage)
        ax[1, 1].fill_between(X, YMin, YMax, color='b', alpha=.1)
        ax[1, 1].set(ylabel='Fitness')

        YAverage = self.Results.groupby(['Generation']).mean()['Results'].sort_values(ascending=False).to_numpy()
        YMin = self.Results.groupby(['Generation']).min()['Results'].sort_values(ascending=False).to_numpy()
        YMax = self.Results.groupby(['Generation']).max()['Results'].sort_values(ascending=False).to_numpy()
        ax[2, 1].plot(X, YAverage)
        ax[2, 1].fill_between(X, YMin, YMax, color='b', alpha=.1)
        ax[2, 1].set(xlabel='Generation', ylabel='LCOE')

        ax[0,1].remove()
        ax[3,1].remove()

        fig.tight_layout()
        plt.show()
        return

class MultipleResults:

    def __init__(self,Results):
        self.Results = Results

    def Plot_one(self, category, xlable, ylable):
        for Result in self.Results:
            Result = pd.read_csv(Result)
            Result['Generation'] = Result['Generation'].str[1:]
            Result['Generation'] = Result['Generation'].astype(float)
            R_Average = Result.groupby(['Generation']).mean().reset_index()
            R_Min = Result.groupby(['Generation']).min().reset_index()
            R_Max = Result.groupby(['Generation']).max().reset_index()
            plt.plot(R_Average['Generation'], R_Average[category])
            #plt.fill_between(R_Average['Generation'], R_Min[category], R_Max[category], alpha=.1)
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.show()

    def Plot_all(self):
        fig, ax = plt.subplots(nrows=6)

        for Result in self.Results:

            Result = pd.read_csv(Result)
            Result['Generation'] = Result['Generation'].str[1:]
            Result['Generation'] = Result['Generation'].astype(float)
            R_Average = Result.groupby(['Generation']).mean().reset_index()
            R_Best = Result.loc[Result.groupby('Generation')['Results'].idxmin()]
            #R_Min = Result.groupby(['Generation']).min().reset_index()
            #R_Max = Result.groupby(['Generation']).max().reset_index()

            ax[0].plot(R_Average['Generation'], R_Average['0'])
            ax[1].plot(R_Average['Generation'], R_Average['1'])
            ax[2].plot(R_Average['Generation'], R_Average['2'])
            ax[3].plot(R_Average['Generation'], R_Average['3'])
            ax[4].plot(R_Average['Generation'], R_Average['4'])
            ax[5].plot(R_Average['Generation'], R_Average['Results'])

            ax[0].plot(R_Average['Generation'], R_Best['0'])
            ax[1].plot(R_Average['Generation'], R_Best['1'])
            ax[2].plot(R_Average['Generation'], R_Best['2'])
            ax[3].plot(R_Average['Generation'], R_Best['3'])
            ax[4].plot(R_Average['Generation'], R_Best['4'])
            ax[5].plot(R_Average['Generation'], R_Best['Results'])

        ax[0].set_ylabel("Lifetime", fontsize=13)
        ax[0].tick_params(labelsize=13)
        ax[1].set_ylabel("Burn-in", fontsize=13)
        ax[1].tick_params(labelsize=13)
        ax[2].set_ylabel("Long term \n Degredation", fontsize=13)
        ax[2].tick_params(labelsize=13)
        ax[3].set_ylabel("Cost", fontsize=13)
        ax[3].tick_params(labelsize=13)
        ax[4].set_ylabel("Power \n Desnsity", fontsize=13)
        ax[4].tick_params(labelsize=13)
        ax[5].set_ylabel("LCOE", fontsize=13)
        ax[5].tick_params(labelsize=13)
        ax[5].set_xlabel("Generation", fontsize=13)

        fig.legend(['Average Device','Best Device'],loc='center right',prop={'size':13})
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.01,right=0.90,left=0.06,bottom=0.05)
        plt.show()

def directory(Dir):
    Devices = os.listdir(Dir)
    for Device in Devices:
        Results = pd.read_csv(Dir + '\\' + Device)
        Results = Results.replace([np.inf, -np.inf], np.nan)
        Results = Results.dropna()
        Results['Generation'] = Results['Generation'].str[1:]
        X = range(len(pd.unique(Results['Generation'])))
        Y = Results.groupby(['Generation']).mean()['Results'].sort_values(ascending=False).to_numpy()
        #YMin = Results.groupby(['Generation']).min()['Results'].sort_values(ascending=False).to_numpy()
        #YMax = Results.groupby(['Generation']).max()['Results'].sort_values(ascending=False).to_numpy()
        plt.plot(X, Y, label=Device)
        #plt.fill_between(X, YMin, YMax, color='b', alpha=.1)
        plt.legend()
    plt.show()
    return


#directory('Generations')
#R = Results('Generations/20210329-115304.csv')
#R.PlotAll()
#R.PlotOne('3','Generations','LCOE')

#R = MultipleResults(['Generations/Australia5.csv','Generations/Brazil2.csv','Generations/India2.csv','Generations/Japan2.csv','Generations/SouthAfrica2.csv','Generations/Spain2.csv','Generations/UK.csv','Generations/USA.csv'])
R = MultipleResults(['Generations/USA2.csv'])
#R = MultipleResults(['Generations/Australia3.csv','Generations/Australia4.csv'])
#R.Plot_one('0','A','B')
R.Plot_all()