import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Load(File):
    Dataframe = pd.read_csv(File)
    return Dataframe

def Datetime(Dataframe):
    Dataframe['Date'] = pd.to_datetime(Dataframe['Date'],format='%d/%m/%Y %H:%M')
    #Dataframe = Dataframe[Dataframe['Date'].dt.year == 2021]
    return Dataframe

def GenMonthsofYear(Dataframe,label):
    Year = range(1,13,1)
    Gen = np.zeros(len(Year))
    for idx,Month in enumerate(Year):
        Gen[idx] = Dataframe[Dataframe['Date'].dt.month == Month]['PV Generation'].sum()
    plt.plot(Year,Gen,label=label)
    return

def AvgDayOfMonth(Dataframe,label,Month):
    Dataframe = Dataframe[Dataframe['Date'].dt.month == Month]
    Day = range(0,24,1)
    Gen = np.zeros(len(Day))
    for idx, Hour in enumerate(Day):
        Gen[idx] = Dataframe[Dataframe['Date'].dt.hour == Hour]['PV Generation'].mean()
    plt.plot(Day, Gen, label=label)
    return

Bangor = Load('UKAvgBangor.csv')
Bangor = Datetime(Bangor)
#GenMonthsofYear(Bangor,'Bangor')
#AvgDayOfMonth(Bangor,'Bangor',1)

Newcastle = Load('UKAvgNewcastle.csv')
Newcastle = Datetime(Newcastle)
#GenMonthsofYear(Newcastle, 'Newcastle')
#AvgDayOfMonth(Newcastle,'Newcastle',1)

PolySi = Load('UKAvgPolySi.csv')
PolySi = Datetime(PolySi)
#GenMonthsofYear(PolySi, 'PolySi')
#AvgDayOfMonth(PolySi,'Poly-Si',1)

for month in range(1,13,1):
    AvgDayOfMonth(Bangor, 'Bangor', month)
    AvgDayOfMonth(Newcastle, 'Newcastle', month)
    AvgDayOfMonth(PolySi, 'Poly-Si', month)
    plt.ylabel('Mean Energy Generated (Kwh)')
    plt.xlabel('Hour of the Day')
    plt.xticks(range(0,24,2))
    plt.tight_layout()
    plt.legend()
    plt.savefig('DSSCCompPowerGenDay' + str(month)  + '.svg')
    plt.clf()


