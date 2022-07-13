import pickle
import time

import pandas as pd
import math
import datetime as dt
import calendar
from dateutil.relativedelta import *
import numpy as np
import time as tt
#from scipy.interpolate import interp1d
import pytz
from pysolar.solar import *
import itertools
import io
import requests
import random
import multiprocessing
import tqdm
from multiprocessing import Pool
from tzwhere import tzwhere
import ujson
import os
from datetime import datetime


class JobQue:

    def __init__(self, que, paneldata):
        self.Jobs = que
        self.Locations = os.path.join('Data','Location')
        self.PanelData = paneldata
        self.Types = os.path.join('Data','Type')
        self.num = 0
        self.Loc = list()
        self.EM = list()
        self.Typ = list()
        self.tf2 = tzwhere.tzwhere(forceTZ=True)
        return

    # loads a preexisting .JBS file
    def re_run(self, of):
        with open(of, 'rb') as f:
            self.Jobs = pickle.load(f)
        return

    # modifies a property (prop) of all Jobs to be set to a common value (prop_val)
    def modify(self, prop, prop_val):
        d = {prop: prop_val}
        for idx, val in enumerate(self.Jobs):
            self.Jobs[idx].update(d)
        return

    # loads a generated que from the object self.jobs
    def load_que(self):
        self.QueDataset = pd.read_csv(self.Jobs)
        self.Jobs = self.QueDataset.to_dict(orient='records')
        return

    # loads data relevant to the chosen or random location
    def load_loc(self):
        extn = ".json"
        i = 0
        for Job in self.Jobs:
            if Job['PrjLoc'] == 'Random':
                pvyield, psh, tilt, lat, lon = self.random_loc_gen()
                self.Jobs[i]['Yield'] = pvyield
                self.Jobs[i]['PeakSunHours'] = psh
                self.Jobs[i]['Tilt'] = tilt
                self.Jobs[i]['Latitude'] = lat
                self.Jobs[i]['Longitude'] = lon
                self.Jobs[i]['IRR'] = 7.5
                tz = pytz.timezone(self.tf2.tzNameAt(latitude=float(lat), longitude=float(lon), forceTZ=True))
                date = dt.datetime(2019, 12, 21, hour=15, tzinfo=tz)
                elevation = get_altitude(float(lat), float(lon), date)
            elif "." in Job['PrjLoc']:
                if int(self.Jobs[i]['Latitude']) > 0:
                    tilt = str(np.abs(int(self.Jobs[i]['Latitude']) - 23))
                else:
                    tilt = str(np.abs(int(self.Jobs[i]['Latitude']) + 23))
                self.Jobs[i]['Tilt'] = tilt
                self.Jobs[i]['IRR'] = 7.5

                tz = pytz.timezone(
                    self.tf2.tzNameAt(
                        latitude=float(self.Jobs[i]['Latitude']),
                        longitude=float(self.Jobs[i]['Longitude']),
                        forceTZ=True))
                date = dt.datetime(2019, 12, 21, hour=15, tzinfo=tz)
                elevation = get_altitude(float(self.Jobs[i]['Latitude']), float(self.Jobs[i]['Longitude']), date)
            else:
                with open(os.path.join(self.Locations,str(Job['PrjLoc'])+extn)) as f:
                    self.Loc.append(ujson.load(f))
                x = list(set(self.Loc[i].keys()).intersection(
                    self.Jobs[i].keys()))
                for dk in x:
                    del self.Loc[i][dk]
                self.Jobs[i].update(self.Loc[i])
                tilt = self.Jobs[i]['Tilt']
                tz = pytz.timezone(
                    self.tf2.tzNameAt(
                        latitude=float(self.Jobs[i]['Latitude']),
                        longitude=float(self.Jobs[i]['Longitude']),
                        forceTZ=True))
                date = dt.datetime(2019, 12, 21, hour=15, tzinfo=tz)
                elevation = get_altitude(float(self.Jobs[i]['Latitude']), float(self.Jobs[i]['Longitude']), date)
            width = 1.968
            height_difference = np.sin(np.radians(tilt)) * width
            module_row_spacing = height_difference / np.tan((np.radians(elevation)))
            row_width = module_row_spacing + (np.cos((np.radians(tilt))) * width)
            self.Jobs[i]['Elevation'] = elevation
            self.Jobs[i]['Spacing'] = row_width
            i += 1
        return

    # load data related to panel and its degradation
    def load_pan(self):
        p = pd.read_csv(self.PanelData)
        for i in range(len(self.Jobs)):
            if i >= len(p):
                Pan = p.iloc[1]
            else:
                try:
                    Pan = p[p['PanelID'] == self.Jobs[i]['PanTyp']].to_dict(orient='records')[
                        0]
                except BaseException:
                    Pan = p[p['PanelID'] == str(self.Jobs[i]['PanTyp'])].to_dict(
                        orient='records')[0]
                X = list(set(Pan.keys()).intersection(
                    self.Jobs[i].keys()))
                for dk in X:
                    del Pan[dk]
            self.Jobs[i].update(Pan)
        i = 0
        for Job in self.Jobs:
            f = str(Job['Tech']) + ".csv"
            self.EM.append(pd.read_csv(f).to_dict(orient='records'))
            X = list(set(self.EM[i][0].keys()).intersection(self.Jobs[i].keys()))
            for dk in X:
                del self.EM[i][0][dk]
            self.Jobs[i].update(self.EM[i][0])
            i += 1
        return

    # load data related to the type of project
    def load_typ(self):
        extn = ".csv"
        i = 0
        for Job in self.Jobs:
            f = os.path.join(self.Types,Job['PrjTyp'] + extn)
            self.Typ.append(pd.read_csv(f).to_dict(orient='records'))
            X = list(set(self.Typ[i][0].keys()).intersection(
                self.Jobs[i].keys()))
            for dk in X:
                del self.Typ[i][0][dk]
            self.Jobs[i].update(self.Typ[i][0])
            i += 1
        return

    # fetches a random location and calculates yield
    def get_loc(self):
        Pass = 0
        lat = 0
        lon = 0
        YieldAPSH = 0
        Tilt = 0
        while Pass == 0:
            lat = str(random.randint(-35, 60))
            lon = str(random.randint(20, 30))
            if int(lat) > 0:
                Tilt = str(np.abs(int(lat) - 23))
            else:
                Tilt = str(np.abs(int(lat) + 23))
            YieldAPSHR = requests.get(
                "https://re.jrc.ec.europa.eu/api/PVcalc?lat=" +
                lat +
                "&lon=" +
                lon +
                "&peakpower=1&loss=14&aspect=0&angle=" +
                Tilt +
                "&pvtechchoice=Unknown&outputformat=csv")
            YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
            YieldAPSHV = YieldAPSH.getvalue()
            if "message" in YieldAPSHV:
                Pass = 0
            elif "Response [200]" in YieldAPSHV:
                Pass = 0
            else:
                Pass = 1
                YieldAPSHR = io.StringIO(YieldAPSHR.content.decode('utf-8'))
                YieldAPSH = pd.read_csv(YieldAPSHR, error_bad_lines=False, skipfooter=12, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8], delimiter='\t\t', engine='python')
                self.num += Pass
        return lat, lon, YieldAPSH, int(Tilt)

    # wrapper function for get_loc
    def random_loc_gen(self):
        lat, lon, YP, Tilt = self.get_loc()
        Yield = YP['E_m'].to_numpy()
        PSH = YP['H(i)_m'].to_numpy()
        return Yield, PSH, Tilt, lat, lon


class Que:

    # initialisation, reads experiments .json file and creates list of dictionaries
    def __init__(self, filename, paneldatafile):
        self.filename = filename
        self.paneldatafile = paneldatafile
        with open(self.filename + '.json') as params:
            paramsDict = ujson.load(params)
            key = list(paramsDict.keys())
            value = list(paramsDict.values())
            for idx, val in enumerate(value):
                for element in value[idx]:
                    if isinstance(element, str):
                        if "#" in element:
                            element = element.split('#')
                            if "X" in element[0]:
                                value[idx] = [value[idx][0]] * int(element[1])
                            if "CA" in element[0]:
                                value[idx] = np.arange(
                                    value[idx][0], value[idx][0] + int(element[1]), 1, dtype='int')
                            if "F" in element[0]:
                                # value[idx] =
                                Fa = pd.read_csv(value[idx][0])
                                keyf = list(Fa.columns.values)
                                valuef = [
                                    list(
                                        Fa[col].to_numpy().astype('str')) for col in keyf]

        self.key = key
        self.value = value
        try:
            self.key += keyf
            self.value += valuef
        except BaseException:
            self.key = key
            self.value = value

    # Converts list of dictionaries to que file (a .csv object)
    def gen_file(self):
        Jobs = list(itertools.product(*self.value))
        Jobs = np.vstack(Jobs)
        Jobs = pd.DataFrame(data=Jobs, columns=self.key)
        Jobs.to_csv(self.filename + ".csv", index=False)
        return

    # loads que file and prerequisites and saves as a .JBS file for execution
    def save_que(self):
        self.filename = self.filename.split('.')[0]

        # Initialise job que object
        JB = JobQue(self.filename + ".csv", self.paneldatafile)
        JB.load_que()  # Loads RunQue as job que object
        JB.load_loc()  # Loads locations in job que object
        JB.load_pan()  # Loads panel in job que object # Loads panel in job que object
        JB.load_typ()  # Load panel type in job que object
        with open(self.filename + '.JBS', 'wb') as handle:
            pickle.dump(JB.Jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


class EPC:

    #  the Initialises the EPC object and calculates all economic factors
    def __init__(self, job):
        self.OriginalCost = job['Design'] + job['Construction'] + job['Framing'] + job['DCcabling'] + \
                            job['ACcabling'] + job['CivilWork(Panels)'] + job['CivilWork(general)'] + \
                            job['PVPanels'] + job['FixedProjectCosts'] + job['Freight(Panels)'] + \
                            job['Freight(other)'] + job['Inverters'] + job['Controls']
        self.PriceExcludingPanels = self.OriginalCost - job['PVPanels']
        self.PanelSize = 410
        self.NumberOfPanels = 1000 * (job['PVSize'] / self.PanelSize)
        self.InstallCostPerPanel = self.PriceExcludingPanels / self.NumberOfPanels
        self.InverterCost = job['Inverters']
        self.PanelCost = job['Cost']
        self.EqRatingPanels = job['PowerDensity'] * 1.968 * 0.992
        self.RequiredNumberPanels = 1000 * job['PVSize'] / self.EqRatingPanels
        self.InstallationCostExcPanels = self.RequiredNumberPanels * self.InstallCostPerPanel
        self.PanelCost2 = self.PanelCost * 1000 * job['PVSize']
        self.NewPrice = self.InstallationCostExcPanels + self.PanelCost2
        self.InverterCostAsPercentofCiepPrice = self.InverterCost / self.InstallationCostExcPanels
        self.NewArea = ((((1.92 * math.cos(math.radians(job['Tilt']))) * 2 + job['Spacing']) * 0.99) / 2) * self.RequiredNumberPanels


class TechTime:
    #  Initialises the TechTime object
    def __init__(self, job):
        #from datetime import datetime
        self.StartDateString = job['ModSta']
        self.TimeStepString = job['TimStp'].lower()

        self.StartDate = datetime.strptime(self.StartDateString, '%d/%m/%Y')

        self.ProjectTime = job['PrjLif'] * 12
        self.EndDate = self.StartDate + relativedelta(months=self.ProjectTime)
        if self.TimeStepString == "month":
            self.Advance = relativedelta(months=1)
            self.InterestDivisor = 12
            self.AdvanceInt = 730 / 24
            self.Entrants = self.ProjectTime
        elif self.TimeStepString == "week":
            self.Advance = relativedelta(weeks=1)
            self.AdvanceInt = 168 / 24
            self.InterestDivisor = 52
            self.Entrants = self.ProjectTime * 4
        elif self.TimeStepString == "day":
            self.Advance = relativedelta(days=1)
            self.AdvanceInt = 24 / 24
            self.InterestDivisor = 365
            self.Entrants = self.ProjectTime * 52
        elif self.TimeStepString == "hour":
            self.Advance = dt.timedelta(hours=1)
            self.AdvanceInt = 1 / 24
            self.InterestDivisor = 8760
            self.Entrants = (self.EndDate - self.StartDate).days * 24

        self.Dates = np.empty(self.Entrants, dtype=datetime)
        date_int = np.linspace(0, self.Entrants, self.Entrants, dtype=np.int)
        self.Dates[:] = self.StartDate + (self.Advance * date_int)


class Panel:
    # Initialises the panel object
    def __init__(self, job):
        self.Lifetime = job['Life'] * 365
        self.FullLifetime = job['Life'] * 365
        self.ReplacementDateIndex = 0
        job['Long-termDegradation'] /= np.sum(job['PeakSunHours'])
        # Tries to calculate the degradation coefficient a, if exception occurs a = 0
        try:
            self.a = (1 - job['Long-termDegradation'] * job['Burn-inPeakSunHours'] -(1 - job['Burn-in'])) / np.power(job['Burn-inPeakSunHours'], 2)
        except BaseException:
            self.a = 0
        # Tries to calculate the degradation coefficient b, if exception occurs b= 0
        try:
            self.b = (-job['Long-termDegradation'] - 2 * self.a * job['Burn-inPeakSunHours'])
        except BaseException:
            self.b = 0

        self.m = -job['Long-termDegradation']
        self.c = (1 - job['Burn-in']) - self.m * job['Burn-inPeakSunHours']
        self.BurnIn = job['Burn-in']
        self.BurnInPSH = job['Burn-inPeakSunHours']
        self.StateOfHealth = [1]
        self.PVSize = job['PVSize']
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
        except BaseException:
            self.ET = 'R'


    # Requests irradiance data from PVGIS
    def pvgis(self, time):
        # Requests and reformats PVGIS data
        try:
            PVGISDataCall = requests.get('https://re.jrc.ec.europa.eu/api/seriescalc?' +
                                         'lat=' +
                                         str(self.Latitude) +
                                         '&lon=' +
                                         str(self.Longitude) +
                                         '&angle=' +
                                         str(self.Tilt) +
                                         '&startyear=' +
                                         str(time.StartDate.year) +
                                         '&endyear=' +
                                         str(time.StartDate.year))
            self.PVGISData = io.StringIO(
                PVGISDataCall.content.decode('utf-8'))
            self.PVGISData = pd.read_csv(self.PVGISData, skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)'])
            self.PVGISData = self.PVGISData.to_numpy()
            # For loop reformat date
            for i in range(len(self.PVGISData)):
                self.PVGISData[:, 0][i] = dt.datetime.strptime(self.PVGISData[:, 0][i][:-2], '%Y%m%d:%H')
            # Identifies index of start date in PVGIS Data
            Shift = np.where(self.PVGISData[:, 0][:] == time.StartDate)[0][0]
            # Shifts starts date to index = 0
            self.PVGISData = np.roll(self.PVGISData, -Shift * 2)
            self.Dates = self.PVGISData[:, 0]
            self.Irradiance = self.PVGISData[:, 1]
        except BaseException:
            tt.sleep(2)
            self.pvgis(time)
        return

    # Expands a single years worth of PVGIS Data to length of defined project
    def expand(self, time):
        self.Dates = time.Dates
        len_dates = len(self.Dates)
        self.Irradiance = np.zeros(len_dates)
        self.Lifetime = np.zeros(len_dates)
        self.PVGen = np.zeros(len_dates)
        self.Lifetime[0] = self.FullLifetime
        Yield = self.Yield
        PSH = self.PSH
        if time.TimeStepString == 'hour':
            self.Yield = np.zeros(len_dates)
            self.PSH = np.zeros(len_dates)
            for i in range(len_dates):
                self.Irradiance[i] = self.PVGISData[:, 1][i % len_dates]
                self.Yield[i] = Yield[i % len_dates]
                self.PSH[i] = PSH[i % len_dates]
        self.CPSH = np.cumsum(self.PSH)
        return

    # Calculates the Yield and Peak Sun Hours at the defined time step
    def yield_and_peak_sun_hours(self, time):
        Yield = np.zeros(len(self.Dates))
        PeakSunHours = np.zeros(len(self.Dates))
        Days = np.zeros(len(self.Dates))
        Month = np.zeros(len(self.Dates))
        IrradianceSum = np.zeros(12)

        i = 0
        for Date in self.Dates:
            Yield[i] = self.Yield[Date.month - 1]
            PeakSunHours[i] = self.PSH[Date.month - 1]
            #Days[i] = calendar.monthrange(Date.year, Date.month)[1]
            Month[i] = np.int(Date.month)
            i += 1
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
            if np.sum(self.Irradiance) == 0:
                Yield = np.zeros(len(self.Irradiance))
                PeakSunHours = np.zeros(len(self.Irradiance))
            else:
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

    # Calculates the degradation of the panel
    def panel_age(self, time):
        self.BurnInAbs = (self.a * self.CPSH * self.CPSH) + \
                         (self.b * self.CPSH + 1)

        self.LongTermDeg = (self.m * self.CPSH) + self.c
        LTDCheck = np.where(self.LongTermDeg < 0)[0]
        self.LongTermDeg[LTDCheck] = 0

        self.LongTermDegAbs = self.LongTermDeg + self.BurnIn
        self.StateOfHealth = self.LongTermDegAbs

        # Checks where SOF > 1 if true  replace with burn-in
        SOFCheck = np.where(self.StateOfHealth > 1)[0]
        self.StateOfHealth[SOFCheck] = self.BurnInAbs[SOFCheck]

        # Checks whether burn-in period has passed
        BurnInTest = self.CPSH > self.BurnInPSH
        BIF = np.where(BurnInTest != True)[0]

        SOFCheck2 = np.where(self.StateOfHealth < 0)[0]
        self.StateOfHealth[SOFCheck2] = 0

        # Calculates the degraded capacity of the farm
        self.Capacity = self.PVSize * (1 - self.BurnIn) * self.StateOfHealth
        self.Capacity[BIF] = self.PVSize * self.StateOfHealth[BIF]

        CCheck = np.where(self.Capacity < 0)[0]
        self.Capacity[CCheck] = 0

        # Calculates the effective capacity of the array
        if time.TimeStepString == 'hour':
            #self.effective_multiplier()
            self.EM = 1
            self.EffectiveCapacity = self.Capacity * self.EM
        elif time.TimeStepString == 'month':
            self.EffectiveCapacity = self.Capacity

        PrevEffCap = np.roll(self.EffectiveCapacity, -1)
        self.PVGen[:] = self.Yield[:] * (self.EffectiveCapacity[:] + PrevEffCap[:]) / 2
        index = np.arange(0, len(self.Dates) - self.ReplacementDateIndex, 1, dtype=int)
        self.Lifetime[self.ReplacementDateIndex:] = self.FullLifetime - (time.AdvanceInt * index[:])

        # Ties to find the the index value of the replacement date
        try:
            self.ReplacementDateIndex = np.where(
                self.Lifetime < time.AdvanceInt)[0][0]
        except BaseException:
            return
        return

    # Replaces the panels with new panels
    def panel_replacement(self):

        self.CPSH[self.ReplacementDateIndex:] = np.cumsum(
            self.PSH[self.ReplacementDateIndex:])
        self.BurnInAbs = (self.a * self.CPSH * self.CPSH) + \
                         (self.b * self.CPSH + 1)
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
    def simulate(self, lifetime):
        if lifetime.TimeStepString == 'hour':
            self.Dates = lifetime.Dates
            self.PVGISData = np.zeros((len(lifetime.Dates),2))
            self.Irradiance = np.zeros(len(lifetime.Dates))
            self.Yield = np.zeros(len(self.Irradiance))
            self.PSH = np.zeros(len(self.Irradiance))
            self.CPSH = np.zeros(len(self.Irradiance))
            #self.pvgis(lifetime)
            a = 0
        else:
            self.Dates = lifetime.Dates
        #self.yield_and_peak_sun_hours(lifetime)
        self.expand(lifetime)
        self.panel_age(lifetime)
        while np.any(self.Lifetime < 0):
            self.panel_replacement()
            self.panel_age(lifetime)
        return

    # Calculates the irradiance dependant effective multiplier
    def effective_multiplier(self):
        WhereZero = np.where(self.Irradiance == 0)
        if self.ET == 'P':
            self.EM = np.power(self.Irradiance * self.LA,3) - np.power(self.Irradiance * self.UA,2) + (self.Irradiance * self.C) + self.Q
        elif self.ET == 'R':
            Device = pd.read_csv(str(self.LA))
            f = interp1d(Device['Irradiance'],Device['Enhanced'],fill_value="extrapolate")
            self.EM = f(self.Irradiance)
        else:
            A = np.exp(-self.GR * (self.Irradiance - self.X))
            self.EM = self.LA + ((self.UA - self.LA) / (self.C + self.Q * A)) ** (1 / self.MG)
        self.EM[WhereZero] = 0
        return


class Inverter:
    # Initialise the inverter object
    def __init__(self, job, time):
        self.FullLifetime = job['InvLif'] * 365
        self.Dates = time.Dates
        self.AdvanceInt = time.AdvanceInt
        self.ReplacementDateIndex = 0
        self.Lifetime = np.zeros(len(self.Dates))

    # Simulates the inverters life
    def simulate(self):
        self.Lifetime[0] = self.FullLifetime
        index = np.arange(0, len(self.Dates) - self.ReplacementDateIndex, 1, dtype=int)
        self.Lifetime[self.ReplacementDateIndex:] = self.FullLifetime - (self.AdvanceInt * index[:])
        while np.any(self.Lifetime < 0):
            index = np.arange(0, len(self.Dates) - self.ReplacementDateIndex, 1, dtype=int)
            self.Lifetime[self.ReplacementDateIndex:] = self.FullLifetime - (self.AdvanceInt * index[:])
            try:
                self.ReplacementDateIndex = np.where(self.Lifetime < self.AdvanceInt)[0][0]
            except BaseException:
                return
        return


class Finance:

    # Initialise the finance object
    def __init__(self, job, e, t, p, i):
        self.Dates = p.Dates
        self.PVSize = p.PVSize
        self.PanelLifetime = p.Lifetime
        self.InverterLifetime = i.Lifetime
        self.InstallationCostExcPanels = e.InstallCostPerPanel
        self.InverterCostAsPercentofCiepPrice = e.InverterCostAsPercentofCiepPrice
        self.NewPrice = e.NewPrice
        self.PanelCost = job['Cost']
        self.DCR = job['IRR'] * 0.01
        self.InverterCostInflation = job['InvCosInf'] * 0.01
        self.OperationCostInflation = job['OprCosInf'] * 0.01
        self.InterestDivisor = t.InterestDivisor
        self.RentCost = job['RenCos']
        self.NewArea = e.NewArea
        self.PVGen = p.PVGen

    # Calculates the price panel
    def panel_price(self):
        self.PanelPrice = np.zeros(len(self.Dates))
        self.PanelPrice[:] = self.PanelCost
        self.PanelPrice = self.PanelCost + (self.PanelPrice - self.PanelCost) * (1 - self.DCR) / self.InterestDivisor
        return

    # Calculates the replacement cost of the panels and inverter
    def replacement(self):
        plr = np.roll(self.PanelLifetime, -1)
        ilr = np.roll(self.InverterLifetime, -1)

        self.PanelReplacements = np.where(self.PanelLifetime < plr)[0][:-1]
        self.PanelReplacementCostPV = np.zeros(len(self.Dates))
        self.PanelReplacementCostPV[self.PanelReplacements] = 1000 * self.PVSize * self.PanelPrice[
            self.PanelReplacements]

        i = np.linspace(0, len(self.Dates), len(self.Dates))
        self.PanelReplacementCostOther = np.zeros(len(self.Dates))
        self.PanelReplacementCostOther[self.PanelReplacements] = (self.NewPrice * 0.1) * np.power((1 + self.InverterCostInflation), (((i[self.PanelReplacements] / self.InterestDivisor) / 365) - 1))

        self.PaneReplacementCost = self.PanelReplacementCostPV + self.PanelReplacementCostOther

        self.InverterReplacements = np.where(
            self.InverterLifetime < ilr)[0][:-1]
        self.InverterReplacementCost = np.zeros(len(self.Dates))
        self.InverterReplacementCost[self.InverterReplacements] = (self.InstallationCostExcPanels *
                                                                   self.InverterCostAsPercentofCiepPrice) * np.power(
            1 + self.InverterCostInflation, (((i[self.InverterReplacements] / self.InterestDivisor) / 365) - 1))

        return

    # Calculates the reoccurring costs of the project
    def recurring_costs(self):
        self.OAM = np.zeros(len(self.Dates))
        self.LandRental = np.zeros(len(self.Dates))
        self.OAM[0] = (1000 * self.PVSize * 0.01) / self.InterestDivisor
        self.LandRental[0] = self.RentCost * self.NewArea / self.InterestDivisor

        for i in range(1, len(self.Dates)):
            self.OAM[i] = self.OAM[i - 1] * \
                          (1 + (self.OperationCostInflation / self.InterestDivisor))
            self.LandRental[i] = self.LandRental[i - 1] * (1 + (self.OperationCostInflation / self.InterestDivisor))

        return

    # Calculates the finances of the project
    def costs(self):
        self.panel_price()
        self.replacement()
        self.recurring_costs()

        self.TotalCosts = self.PaneReplacementCost + self.InverterReplacementCost + self.OAM + self.LandRental
        return

    # Calculates the LCOE at the end of the project
    def lcoe_calculate(self):
        i = np.linspace(0, len(self.Dates), len(self.Dates))
        tc = self.TotalCosts[:]
        pv = self.PVGen[:]
        ii = i[:] / self.InterestDivisor
        self.LCOE = (self.NewPrice + np.abs(self.xnpv(self.DCR, tc[:], ii[:]))) / self.xnpv(self.DCR, pv[:], ii[:])
        return

    @staticmethod
    def xnpv(dcr, values, date):
        V = np.sum(values[:] / (1.0 + dcr) ** (date[:]))
        return V


class Out:

    # Initialises the Out object
    def __init__(self, job, epc, time, panel, inverter, finance):
        self.Job = job
        self.EPC = epc
        self.Time = time
        self.Panel = panel
        self.Inverter = inverter
        self.Finance = finance

    # Outputs the results as an excel file
    def excel(self):
        df = pd.DataFrame(self.Panel.Dates, columns=['Date'])
        df['Irradiance'] = pd.Series(self.Panel.Irradiance, index=df.index)
        df['Panel Lifetime'] = pd.Series(self.Panel.Lifetime, index=df.index)
        df['Inverter Lifetime'] = pd.Series(
            self.Inverter.Lifetime, index=df.index)
        df['Peak Sun Hours'] = pd.Series(self.Panel.PSH, index=df.index)
        df['Cumilative Sun Hours'] = pd.Series(self.Panel.CPSH, index=df.index)
        df['Burn-in Abs'] = pd.Series(self.Panel.BurnInAbs, index=df.index)
        df['Long Term Degradation'] = pd.Series(
            self.Panel.LongTermDeg, index=df.index)
        df['Long Term Degradation Abs'] = pd.Series(
            self.Panel.LongTermDegAbs, index=df.index)
        df['Panel State of Health'] = pd.Series(
            self.Panel.StateOfHealth, index=df.index)
        df['Peak Capacity'] = pd.Series(self.Panel.Capacity, index=df.index)
        df['Effective Capacity'] = pd.Series(
            self.Panel.EffectiveCapacity, index=df.index)
        df['Monthly Yield'] = pd.Series(self.Panel.Yield, index=df.index)
        df['PV Generation'] = pd.Series(self.Panel.PVGen, index=df.index)
        df['Refurbishment Cost (PV)'] = pd.Series(
            self.Finance.PanelReplacementCostPV, index=df.index)
        df['Refurbishment Cost (Other)'] = pd.Series(
            self.Finance.PanelReplacementCostOther, index=df.index)
        df['Refurbishment Cost (Panels)'] = pd.Series(
            self.Finance.PaneReplacementCost, index=df.index)
        df['Panel Price This Year'] = pd.Series(
            self.Finance.panel_price, index=df.index)
        df['Refurbishment Cost (Inverter)'] = pd.Series(
            self.Finance.InverterReplacementCost, index=df.index)
        df['Annual O&M Cost'] = pd.Series(self.Finance.OAM, index=df.index)
        df['Land Rental'] = pd.Series(self.Finance.LandRental, index=df.index)
        df['Total Cost'] = pd.Series(self.Finance.TotalCosts, index=df.index)
        df['LCOE'] = pd.Series(self.Finance.LCOE, index=df.index)
        df['Enhancment'] = pd.Series(self.Panel.EM, index=df.index)
        CurrentDatetime = dt.datetime.now()
        StringCurrentDatetime = CurrentDatetime.strftime('%Y%m%d%H%M%S')
        df.to_csv(str(self.Job['PrjLoc']) + "-" +
                  StringCurrentDatetime + ".csv")
        return

    # Outputs the results specified at the first line of Results.csv file
    def results(self):

        File = pd.read_csv('Results.csv')
        ResultsRequested = File.columns.values
        ResultsOutput = list()
        for Result in ResultsRequested:
            Result = Result.split('.')
            try:
                if Result[0] == 'Finance':
                    Result = getattr(self.Finance, Result[1])
                    ResultsOutput.append(Result)
                elif Result[0] == 'Panel':
                    Result = getattr(self.Panel, Result[1])
                    ResultsOutput.append(np.average(Result[np.nonzero(Result)]))
                elif Result[0] == 'Inverter':
                    Result = getattr(self.Inverter, Result[1])[-1]
                    ResultsOutput.append(Result)
                elif Result[0] == 'EPC':
                    Result = getattr(self.EPC, Result[1])[-1]
                    ResultsOutput.append(Result)
                else:
                    Result = self.Job[Result[1]]
                    ResultsOutput.append(Result)
            except BaseException:
                ResultsOutput.append('Error')
                continue
        ResultO = pd.DataFrame([ResultsOutput], columns=ResultsRequested)
        File = File.append(ResultO, ignore_index=True)
        File.to_csv('Results.csv', index=False)
        return

    @staticmethod
    def m(a):
        return a.month

    @staticmethod
    def t(a):
        return a.hour


class LCOE:

    # Saves inputs and prerequisites as objects for execution
    def __init__(self, filename, paneldatafile):
        self.filename = filename
        self.paneldatafile = paneldatafile
        self.ResultsLoc = 'Results.csv'
        self.Results = pd.read_csv(self.ResultsLoc)
        self.ResultsCol = self.Results.columns
        self.EmptyResults = pd.DataFrame(columns=self.ResultsCol)
        self.Resutls = self.Results['Finance.LCOE'].to_numpy()

    # Wrapper function to generate a .JBS file based upon a .json file
    def generate_jbs(self):
        self.Q = Que(self.filename, self.paneldatafile)
        self.Q.gen_file()
        self.Q.save_que()
        return

    # Wrapper function to load a preexisting .JBS file
    def load_jbs(self):
        self.Q = JobQue(self.filename + '.json', self.paneldatafile)
        self.Q.re_run(self.filename + '.JBS')
        self.Q.load_pan()
        return

    # Wrapper function to modify the 'Tech' and 'Panel Type' found in a experiment file (.JBS)
    def variations(self, devices, variations):
        for idx, devices in enumerate(devices):
            self.Q.Modify('Tech', variations[idx])
            self.Q.Modify('PanTyp', devices)
        return

    # Wrapper function to calculate LCOE, to be called by itself or as part of multiprocessing
    @staticmethod
    def worker(job):
        E = EPC(job)
        t = TechTime(job)
        P = Panel(job)
        P.simulate(t)
        I = Inverter(job, t)
        I.simulate()
        F = Finance(job, E, t, P, I)
        F.costs()
        F.lcoe_calculate()
        return job, E, t, P, I, F

    # Multiprocessing execution of LCOE calculations
    def run(self):
        with Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            data = list(tqdm.tqdm(pool.imap_unordered(self.worker, self.Q.Jobs)))
            pool.close()
        for idx in data:
            O = Out(idx[0], idx[1], idx[2], idx[3], idx[4], idx[5])
            O.results()
        return