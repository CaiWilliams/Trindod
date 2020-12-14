import calendar
import numpy as np
import pandas as pd
import requests
import time
import io
from datetime import datetime, timedelta
from dateutil.relativedelta import *

class Panel:
    def __init__(self,Job,EPC):
        self.Lifetime = Job['Life'] * 365
        self.FullLifetime = Job['Life'] * 365
        self.ProjectLength = Job['PrjLif']
        self.ReplacmentDateIndex = 0
        Job['Long-termDegradation'] = Job['Long-termDegradation'] / np.sum(Job['PeakSunHours'])
        try:
            self.a = (1 - Job['Long-termDegradation'] * Job['Burn-inPeakSunHours'] - (1 - Job['Burn-in'])) /np.power(Job['Burn-inPeakSunHours'],2)
        except:
            self.a = 0
        try:
            self.b = (-Job['Long-termDegradation'] - 2 * self.a * Job['Burn-inPeakSunHours'])
        except:
            self.b = 0
        self.m = -Job['Long-termDegradation']
        self.c = (1 - Job['Burn-in']) - self.m * Job['Burn-inPeakSunHours']
        self.BurnIn = Job['Burn-in']
        self.BrunInPSH = Job['Burn-inPeakSunHours']
        self.StateOfHealth = 1
        self.PVSize = Job['PVSize']
        self.Cost = Job['Cost']
        self.PSH = np.array(Job['PeakSunHours'])
        self.Yield = np.array(Job['Yield'])
        self.Tilt = Job['Tilt']
        self.Latitude = Job['Latitude']
        self.Longitude = Job['Longitude']
        self.LA = Job['0']
        self.UA = Job['1']
        self.C = Job['2']
        self.Q = Job['3']
        self.GR = Job['4']
        self.MG = Job['5']
        self.X = Job['6']
        self.HoursInEn = 0 

    def PVGIS(self,Time):
        self.PVGISData = requests.get('https://re.jrc.ec.europa.eu/api/seriescalc?'+'lat=' +str(self.Latitude) + '&lon='+str(self.Longitude) + '&angle='+str(self.Tilt)+'&startyear=2015&endyear=2015')
        self.PVGISData = io.StringIO(self.PVGISData.content.decode('utf-8'))
        self.PVGISData = pd.read_csv(self.PVGISData,skipfooter=9,skiprows=[0,1,2,3,4,5,6,7],engine='python',usecols=['time','G(i)'])
        self.PVGISData = self.PVGISData.to_numpy()
        for i in range(len(self.PVGISData)):
            self.PVGISData[:,0][i] = datetime.strptime(self.PVGISData[:,0][i][:-2],'%Y%m%d:%H')
            self.PVGISData[:,0][i] = self.PVGISData[:,0][i].replace(year=2019)
        Shift = np.where(self.PVGISData[:,0][:] == Time.StartDate)[0][0]
        self.PVGISData = np.roll(self.PVGISData,-Shift*2)
        self.Dates = self.PVGISData[:,0]
        self.Irradiance = self.PVGISData[:,1]
        return
    
    def Expand(self,Time):
        self.Dates = Time.Dates
        self.Irradiance = np.zeros(len(self.Dates))
        self.Lifetime = np.zeros(len(self.Dates))
        self.PVGen = np.zeros(len(self.Dates))
        self.Lifetime[0] = self.FullLifetime
        Yield = self.Yield 
        PSH = self.PSH
        self.Yield = np.zeros(len(self.Dates))
        self.PSH = np.zeros(len(self.Dates))
        for i in range(len(self.Dates)):
            self.Irradiance[i] = self.PVGISData[:,1][i % len(self.PVGISData)]
            self.Yield[i] = Yield[i % len(self.PVGISData)]
            self.PSH[i] = PSH[i % len(self.PVGISData)]
        self.CPSH = np.cumsum(self.PSH)
        return
    
    def YeildAndPeakSunHours(self,Time):

        Yield = np.zeros(len(self.Dates))
        PeakSunHours = np.zeros(len(self.Dates))
        Days = np.zeros(len(self.Dates))
        Month = np.zeros(len(self.Dates))
        IrradianceSum = np.zeros(12)

        i =  0
        for Date in self.Dates:
            Yield[i] = self.Yield[Date.month - 1]
            PeakSunHours[i] = self.PSH[Date.month - 1]
            Days[i] = calendar.monthrange(Date.year,Date.month)[1]
            Month[i] = np.int(Date.month)
            i = i + 1 

        if Time.TimestepString == 'month':
            self.Yield = Yield
            self.PSH = PeakSunHours
            return
        elif Time.TimestepString == 'week':
            Yield[:] = Yield[:]/4
            PeakSunHours[:] = PeakSunHours[:]/4
            self.Yield = Yield
            self.PSH = PeakSunHours
            return
        elif Time.TimestepString == 'day':
            Yield[:] = Yield[:]/Days[:]
            PeakSunHours[:] = PeakSunHours[:]/Days[:]
            self.Yield = Yield
            self.PSH = PeakSunHours
            return
        elif Time.TimestepString == 'hour':
            for i in range(1,13):
                Irradiance = self.Irradiance[np.where(Month == i)]
                IrradianceSum[i-1] = np.sum(Irradiance) 
            for i in range(len(self.Dates)):
                Yield[i] = self.Irradiance[i] * (Yield[i] / IrradianceSum[int(Month[i]-1)])
                PeakSunHours[i] = self.Irradiance[i] * (PeakSunHours[i] / IrradianceSum[int(Month[i]-1)])
            self.Yield = Yield
            self.PSH = PeakSunHours
            self.CPSH = np.cumsum(self.PSH[:])
            return

    def PanelAge(self,Time):
        self.BurnInAbs = (self.a * self.CPSH * self.CPSH) + (self.b * self.CPSH + 1)
        self.LongTermDeg = (self.m * self.CPSH) + self.c 
        self.LongTermDegAbs = self.LongTermDeg + self.BurnIn
        self.StateOfHealth = self.LongTermDegAbs

        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        BurnInTest = self.CPSH > self.BrunInPSH
        BIF = np.where(BurnInTest == False)[0]
        self.Capacity = self.PVSize * (1 - self.BurnIn) * self.StateOfHealth
        self.Capacity[BIF] = self.PVSize * self.StateOfHealth[BIF]

        self.EffectiveMultiplier()
        self.EffectiveCapacity = self.Capacity  * self.EM
        PrevEffCap = np.roll(self.EffectiveCapacity,-1)
        self.PVGen[:] = self.Yield[:] * (self.EffectiveCapacity[:] + PrevEffCap[:])/2
        I = np.arange(0,len(self.Dates)-self.ReplacmentDateIndex,1,dtype=int)
        self.Lifetime[self.ReplacmentDateIndex:] = self.FullLifetime - (Time.AdvanceInt * I[:])
        try:
            self.ReplacmentDateIndex = np.where(self.Lifetime < Time.AdvanceInt)[0][0]
        except:
            return
        return
    
    def PanelReplacment(self):
        self.CPSH[self.ReplacmentDateIndex:] = np.cumsum(self.PSH[self.ReplacmentDateIndex:])
        self.BurnInAbs = (self.a * self.CPSH * self.CPSH) + (self.b * self.CPSH + 1)
        self.LongTermDeg = (self.m * self.CPSH) + self.c
        self.LongTermDegAbs = self.LongTermDeg + self.BurnIn

        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        BurnInTest = self.CPSH > self.BrunInPSH
        BIF = np.where(BurnInTest == False)[0]
        self.Capacity = self.PVSize * (1 - self.BurnIn) * self.StateOfHealth
        self.Capacity[BIF] = self.PVSize * self.StateOfHealth[BIF]

        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        self.Capacity = self.PVSize * self.StateOfHealth
        return

    def Simulate(self,Time):
        self.PVGIS(Time)
        self.YeildAndPeakSunHours(Time)
        self.Expand(Time)
        self.PanelAge(Time)
        while np.any(self.Lifetime < 0):
            self.PanelReplacment()
            self.PanelAge(Time)
        return

    def EffectiveMultiplier(self):
        WhereZero = np.where(self.Irradiance == 0)
        A = np.exp(-self.GR * (self.Irradiance - self.X))
        self.EM = self.LA + ((self.UA - self.LA)/(self.C + self.Q * A)) ** (1 / self.MG)
        self.EM[WhereZero] = 0
        return

class Inverter:
    def __init__(self,Job,Time):
        self.FullLifetime = Job['InvLif'] * 365
        self.Dates = Time.Dates
        self.AdvanceInt = Time.AdvanceInt
        self.ReplacmentDateIndex = 0
        self.Lifetime = np.zeros(len(self.Dates))

    def Simulate(self):
        self.Lifetime[0] = self.FullLifetime
        I = np.arange(0,len(self.Dates)-self.ReplacmentDateIndex,1,dtype=int)
        self.Lifetime[self.ReplacmentDateIndex:] = self.FullLifetime - (self.AdvanceInt * I[:])
        while np.any(self.Lifetime < 0):
            I = np.arange(0,len(self.Dates)-self.ReplacmentDateIndex,1,dtype=int)
            self.Lifetime[self.ReplacmentDateIndex:] = self.FullLifetime - (self.AdvanceInt * I[:])
            try:
                self.ReplacmentDateIndex = np.where(self.Lifetime < self.AdvanceInt)[0][0]
            except:
                return
        return
        

class TechTime:
    def __init__(self,Job):
        self.StartDateString = Job['ModSta']
        self.TimestepString = Job['TimStp'].lower()

        self.StartDate = datetime.strptime(self.StartDateString,'%d/%m/%Y')
        D = calendar.monthrange(self.StartDate.year,self.StartDate.month)[1]

        self.ProjectTime = Job['PrjLif'] * 12
        self.EndDate = self.StartDate + relativedelta(months = self.ProjectTime)
        if self.TimestepString == "month":
            self.Advance = relativedelta(months = 1)
            self.GHDevisor = 1
            self.InterestDevisor = 12
            self.AdvanceInt = 730/24
            self.Entrants = self.ProjectTime
        elif self.TimestepString == "week":
            self.Advance = relativedelta(weeks = 1)
            self.GHDevisor = 4
            self.AdvanceInt = 168/24
            self.InterestDevisor = 52
            self.Entrants = self.ProjectTime * 4
        elif self.TimestepString == "day":
            self.Advance = relativedelta(days = 1)
            self.GHDevisor = D
            self.AdvanceInt = 24/24
            self.InterestDevisor = 365
            self.Entrants = self.ProjectTime * 52
        elif self.TimestepString == "hour":
            self.Advance = relativedelta(hours = 1)
            self.GHDevisor = D * 24
            self.AdvanceInt = 1/24
            self.InterestDevisor = 8760
            self.Entrants = (self.EndDate - self.StartDate).days * 24
        
        self.Dates = np.empty(self.Entrants, dtype=datetime)
        I = np.linspace(0,self.Entrants,self.Entrants,dtype=np.int)
        self.Dates[:] = self.StartDate  + (self.Advance * I)

    def DateAdvance(self):
        self.CurrentDate  = self.CurrentDate + self.Advance
        return
    
