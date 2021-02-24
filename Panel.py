import calendar
import numpy as np
import pandas as pd
import requests
import io
from datetime import datetime
import time as TTime
from dateutil.relativedelta import *
from scipy.interpolate import interp1d



# noinspection PyBroadException
# Class for the simulation of the panel
class Panel:
    # Initialises the panel object

    def __init__(self, job):
        self.Lifetime = job['Life'] * 365
        self.FullLifetime = job['Life'] * 365
        self.ProjectLength = job['PrjLif']
        self.ReplacementDateIndex = 0
        job['Long-termDegradation'] = job['Long-termDegradation'] / np.sum(job['PeakSunHours'])
        # Tries to calculate the degredation coeficient a, if exception occurs a = 0
        try:
            self.a = (1 - job['Long-termDegradation'] * job['Burn-inPeakSunHours'] - (1 - job['Burn-in'])) / np.power(job['Burn-inPeakSunHours'], 2)
        except BaseException:
            self.a = 0
        # Tries to calculate the degredation coeficient b, if excepion occurs b = 0
        try:
            self.b = (-job['Long-termDegradation'] - 2 * self.a * job['Burn-inPeakSunHours'])
        except BaseException:
            self.b = 0
        self.m = -job['Long-termDegradation']
        self.c = (1 - job['Burn-in']) - self.m * job['Burn-inPeakSunHours']
        self.BurnIn = job['Burn-in']
        self.BurnInPSH = job['Burn-inPeakSunHours']
        self.StateOfHealth = 1
        self.PVSize = job['PVSize']
        self.Cost = job['Cost']
        self.PSH = np.array(job['PeakSunHours'])
        self.Yield = np.array(job['Yield'])
        self.Tilt = job['Tilt']
        self.Latitude = job['Latitude']
        self.Longitude = job['Longitude']
        self.LA = job['0']
        self.UA = job['1']
        self.C = job['2']
        self.Q = job['3']
        self.GR = job['4']
        self.MG = job['5']
        self.X = job['6']
        try:
            self.ET = job['7']
        except:
            self.ET = 'R'
        self.HoursInEn = 0

    # Requests irradiance data from PVGIS
    def PVGIS(self, time):
        # Requests and reformats PVGIS data
        try:
            self.PVGISData = requests.get('https://re.jrc.ec.europa.eu/api/seriescalc?' + 'lat=' + str(self.Latitude) + '&lon=' + str(self.Longitude) + '&angle=' + str(self.Tilt) + '&startyear=2015&endyear=2015')
            self.PVGISData = io.StringIO(self.PVGISData.content.decode('utf-8'))
            self.PVGISData = pd.read_csv(self.PVGISData, skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)'])
            self.PVGISData = self.PVGISData.to_numpy()
            # For loop reformats date
            for i in range(len(self.PVGISData)):
                self.PVGISData[:, 0][i] = datetime.strptime(self.PVGISData[:, 0][i][:-2], '%Y%m%d:%H')
                self.PVGISData[:, 0][i] = self.PVGISData[:, 0][i].replace(year=2019)
            Shift = np.where(self.PVGISData[:, 0][:] == time.StartDate)[0][0]  # Identifies index of start date in PVGIS Data
            self.PVGISData = np.roll(self.PVGISData, -Shift * 2)  # Shifts starts date to index = 0
            self.Dates = self.PVGISData[:, 0]
            self.Irradiance = self.PVGISData[:, 1]
        except:
            TTime.sleep(2)
            self.PVGIS(time)
        return

    # Expands a single years worth of PVGIS Data to length of defined project
    def Expand(self, time):
        self.Dates = time.Dates
        self.Irradiance = np.zeros(len(self.Dates))
        self.Lifetime = np.zeros(len(self.Dates))
        self.PVGen = np.zeros(len(self.Dates))
        self.Lifetime[0] = self.FullLifetime
        Yield = self.Yield
        PSH = self.PSH
        self.Yield = np.zeros(len(self.Dates))
        self.PSH = np.zeros(len(self.Dates))
        for i in range(len(self.Dates)):
            self.Irradiance[i] = self.PVGISData[:, 1][i % len(self.PVGISData)]
            self.Yield[i] = Yield[i % len(self.PVGISData)]
            self.PSH[i] = PSH[i % len(self.PVGISData)]
        self.CPSH = np.cumsum(self.PSH)
        return

    # Calculates the Yield and Peak Sun Hours at the defined timestep
    def YieldAndPeakSunHours(self, time):
        Yield = np.zeros(len(self.Dates))
        PeakSunHours = np.zeros(len(self.Dates))
        Days = np.zeros(len(self.Dates))
        Month = np.zeros(len(self.Dates))
        IrradianceSum = np.zeros(12)

        i = 0
        for Date in self.Dates:
            Yield[i] = self.Yield[Date.month - 1]
            PeakSunHours[i] = self.PSH[Date.month - 1]
            Days[i] = calendar.monthrange(Date.year, Date.month)[1]
            Month[i] = np.int(Date.month)
            i = i + 1
        if time.TimeStepString == 'month':
            self.Yield = Yield
            self.PSH = PeakSunHours
            return
        elif time.TimeStepString == 'week':
            Yield[:] = Yield[:] / 4
            PeakSunHours[:] = PeakSunHours[:] / 4
            self.Yield = Yield
            self.PSH = PeakSunHours
            return
        elif time.TimeStepString == 'day':
            Yield[:] = Yield[:] / Days[:]
            PeakSunHours[:] = PeakSunHours[:] / Days[:]
            self.Yield = Yield
            self.PSH = PeakSunHours
            return
        elif time.TimeStepString == 'hour':
            for i in range(1, 13):
                Irradiance = self.Irradiance[np.where(Month == i)]
                IrradianceSum[i - 1] = np.sum(Irradiance)
            for i in range(len(self.Dates)):
                Yield[i] = self.Irradiance[i] * (Yield[i] / IrradianceSum[int(Month[i] - 1)])
                PeakSunHours[i] = self.Irradiance[i] * (PeakSunHours[i] / IrradianceSum[int(Month[i] - 1)])
            self.Yield = Yield
            self.PSH = PeakSunHours
            self.CPSH = np.cumsum(self.PSH[:])
            return

    # Calculates the degredation of the panel
    def PanelAge(self, time):
        self.BurnInAbs = (self.a * self.CPSH * self.CPSH) + (self.b * self.CPSH + 1)
        self.LongTermDeg = (self.m * self.CPSH) + self.c
        self.LongTermDegAbs = self.LongTermDeg + self.BurnIn
        self.StateOfHealth = self.LongTermDegAbs

        # Checks where SOF > 1 if true  replace with burn-in
        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        # Checks whether burn-in period has passed
        BurnInTest = self.CPSH > self.BurnInPSH
        BIF = np.where(BurnInTest != True)[0]

        # Calculates the degraded capacity of the farm
        self.Capacity = self.PVSize * (1 - self.BurnIn) * self.StateOfHealth
        self.Capacity[BIF] = self.PVSize * self.StateOfHealth[BIF]

        # Calculates the effective capacity of the array
        self.EffectiveMultiplier()
        self.EffectiveCapacity = self.Capacity * self.EM

        PrevEffCap = np.roll(self.EffectiveCapacity, -1)
        self.PVGen[:] = self.Yield[:] * (self.EffectiveCapacity[:] + PrevEffCap[:]) / 2
        I = np.arange(0, len(self.Dates) - self.ReplacementDateIndex, 1, dtype=int)
        self.Lifetime[self.ReplacementDateIndex:] = self.FullLifetime - (time.AdvanceInt * I[:])

        # Ties to find the the index value of the replacment date
        try:
            self.ReplacementDateIndex = np.where(self.Lifetime < time.AdvanceInt)[0][0]
        except BaseException:
            return
        return

    # Replaces the panels with new panels
    def PanelReplacement(self):

        self.CPSH[self.ReplacementDateIndex:] = np.cumsum(self.PSH[self.ReplacementDateIndex:])
        self.BurnInAbs = (self.a * self.CPSH * self.CPSH) + (self.b * self.CPSH + 1)
        self.LongTermDeg = (self.m * self.CPSH) + self.c
        self.LongTermDegAbs = self.LongTermDeg + self.BurnIn

        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        BurnInTest = self.CPSH > self.BurnInPSH
        BIF = np.where(BurnInTest == False)[0]
        self.Capacity = self.PVSize * (1 - self.BurnIn) * self.StateOfHealth
        self.Capacity[BIF] = self.PVSize * self.StateOfHealth[BIF]

        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        self.Capacity = self.PVSize * self.StateOfHealth
        return

    # Simulates the life of of the panels
    def Simulate(self, time):
        self.PVGIS(time)
        self.YieldAndPeakSunHours(time)
        self.Expand(time)
        self.PanelAge(time)
        while np.any(self.Lifetime < 0):
            self.PanelReplacement()
            self.PanelAge(time)
        return

    # Calculates the irradaince dependant effective multiplier
    def EffectiveMultiplier(self):
        WhereZero = np.where(self.Irradiance == 0)
        if self.ET == 'P':
            self.EM = np.power(self.Irradiance * self.LA, 3) - np.power(self.Irradiance * self.UA, 2) + (self.Irradiance * self.C) + self.Q
        elif self.ET == 'Raw':
            Device = pd.read_csv(str(self.LA))
            f = interp1d(Device['Irradiance'], Device['Enhanced'], fill_value="extrapolate")
            self.EM = f(self.Irradiance)
        else:
            A = np.exp(-self.GR * (self.Irradiance - self.X))
            self.EM = self.LA + ((self.UA - self.LA) / (self.C + self.Q * A)) ** (1 / self.MG)
        self.EM[WhereZero] = 0
        return


# Class for the simulation of the inverter
class Inverter:
    # Initialies the inverter object
    def __init__(self, job, time):
        self.FullLifetime = job['InvLif'] * 365
        self.Dates = time.Dates
        self.AdvanceInt = time.AdvanceInt
        self.ReplacementDateIndex = 0
        self.Lifetime = np.zeros(len(self.Dates))

    # Simulates the inverters life
    def Simulate(self):
        self.Lifetime[0] = self.FullLifetime
        I = np.arange(0, len(self.Dates) -
                      self.ReplacementDateIndex, 1, dtype=int)
        self.Lifetime[self.ReplacementDateIndex:] = self.FullLifetime - (self.AdvanceInt * I[:])
        while np.any(self.Lifetime < 0):
            I = np.arange(0, len(self.Dates) -
                          self.ReplacementDateIndex, 1, dtype=int)
            self.Lifetime[self.ReplacementDateIndex:] = self.FullLifetime - (self.AdvanceInt * I[:])
            try:
                self.ReplacementDateIndex = np.where(
                    self.Lifetime < self.AdvanceInt)[0][0]
            except BaseException:
                return
        return


#  Class for time and timesteps

class TechTime:
    #  Initialises the techtime object

    def __init__(self, job):
        self.StartDateString = job['ModSta']
        self.TimeStepString = job['TimStp'].lower()

        self.StartDate = datetime.strptime(self.StartDateString, '%d/%m/%Y')
        D = calendar.monthrange(self.StartDate.year, self.StartDate.month)[1]

        self.ProjectTime = job['PrjLif'] * 12
        self.EndDate = self.StartDate + relativedelta(months=self.ProjectTime)
        if self.TimeStepString == "month":
            self.Advance = relativedelta(months=1)
            self.GHDivisor = 1
            self.InterestDivisor = 12
            self.AdvanceInt = 730 / 24
            self.Entrants = self.ProjectTime
        elif self.TimeStepString == "week":
            self.Advance = relativedelta(weeks=1)
            self.GHDivisor = 4
            self.AdvanceInt = 168 / 24
            self.InterestDivisor = 52
            self.Entrants = self.ProjectTime * 4
        elif self.TimeStepString == "day":
            self.Advance = relativedelta(days=1)
            self.GHDivisor = D
            self.AdvanceInt = 24 / 24
            self.InterestDivisor = 365
            self.Entrants = self.ProjectTime * 52
        elif self.TimeStepString == "hour":
            self.Advance = relativedelta(hours=1)
            self.GHDivisor = D * 24
            self.AdvanceInt = 1 / 24
            self.InterestDivisor = 8760
            self.Entrants = (self.EndDate - self.StartDate).days * 24

        self.Dates = np.empty(self.Entrants, dtype=datetime)
        I = np.linspace(0, self.Entrants, self.Entrants, dtype=np.int)
        self.Dates[:] = self.StartDate + (self.Advance * I)

    def DateAdvance(self):
        self.CurrentDate = self.CurrentDate + self.Advance
        return
