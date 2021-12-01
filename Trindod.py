import pickle
import pandas as pd
import math
from datetime import datetime
import calendar
from dateutil.relativedelta import *
import numpy as np
import time as ttime
from scipy.interpolate import interp1d
import pytz
from pysolar.solar import *
import itertools
import json
import io
import requests
from timezonefinder import TimezoneFinder
import random
import multiprocessing
import tqdm
from Ryfeddod import *
from multiprocessing import Pool
from tzwhere import tzwhere


class JobQue:

    def __init__(self, que, paneldata):
        self.Jobs = que
        self.Locations = r'Data\Location'
        self.Panels = r'Data\Panel'
        self.PanelData = paneldata
        self.Types = r'Data\Type'
        self.num = 0
        self.tf2 = tzwhere.tzwhere(forceTZ=True)
        return

    def ReRun(self, of):
        with open(of, 'rb') as f:
            self.Jobs = pickle.load(f)
        return

    def Modify(self, prop, prop_val):
        d = {prop: prop_val}
        for idx, val in enumerate(self.Jobs):
            self.Jobs[idx].update(d)
        return

    def LoadQue(self):
        self.QueDataset = pd.read_csv(self.Jobs)
        self.Jobs = self.QueDataset.to_dict(orient='records')
        return

    def LoadLoc(self):
        extn = ".json"
        self.Loc = list()
        self.i = 0
        while self.i < len(self.Jobs):
            if self.Jobs[self.i]['PrjLoc'] == 'Random':
                Yield, PSH, Tilt, lat, lon = self.RandomLocGen()
                self.Jobs[self.i]['Yield'] = Yield
                self.Jobs[self.i]['PeakSunHours'] = PSH
                self.Jobs[self.i]['Tilt'] = Tilt
                self.Jobs[self.i]['Latitude'] = lat
                self.Jobs[self.i]['Longitude'] = lon
                self.Jobs[self.i]['IRR'] = 7.5
                tf = TimezoneFinder()
                TZ = pytz.timezone(self.tf2.tzNameAt(latitude=float(self.Jobs[self.i]['Latitude']), longitude=float(self.Jobs[self.i]['Longitude']),forceTZ=True))
                date = datetime.datetime(2015, 12, 21, hour=15, tzinfo=TZ)
                elevation = get_altitude(float(lat), float(lon), date)
                Width = 1.968
                HeightDifference = np.sin(np.radians(Tilt)) * Width
                ModuleRowSpacing = HeightDifference / np.tan((np.radians(elevation)))
                RowWidth = ModuleRowSpacing + (np.cos((np.radians(Tilt))) * Width)
                self.Jobs[self.i]['Elevation'] = elevation
                self.Jobs[self.i]['Spacing'] = RowWidth
            elif "Placeholder" in self.Jobs[self.i]['PrjLoc']:
                print(self.i)
                Yield, PSH, Tilt, = self.Fetch_Yeild(self.Jobs[self.i]['Latitude'], self.Jobs[self.i]['Longitude'])
                if np.sum(Yield) == 0 and np.sum(PSH) == 0 and Tilt == 0:
                    continue
                self.Jobs[self.i]['Yield'] = Yield
                self.Jobs[self.i]['PeakSunHours'] = PSH
                self.Jobs[self.i]['Tilt'] = Tilt
                self.Jobs[self.i]['IRR'] = 7.5
                lat = self.Jobs[self.i]['Latitude']
                lon = self.Jobs[self.i]['Longitude']
                tf = TimezoneFinder()
                TZ = pytz.timezone(self.tf2.tzNameAt(latitude=float(self.Jobs[self.i]['Latitude']), longitude=float(self.Jobs[self.i]['Longitude']),forceTZ=True))
                date = datetime.datetime(2015, 12, 21, hour=15, tzinfo=TZ)
                elevation = get_altitude(float(lat), float(lon), date)
                Width = 1.968
                HeightDifference = np.sin(np.radians(Tilt)) * Width
                ModuleRowSpacing = HeightDifference / np.tan((np.radians(elevation)))
                RowWidth = ModuleRowSpacing + (np.cos((np.radians(Tilt))) * Width)
                self.Jobs[self.i]['Elevation'] = elevation
                self.Jobs[self.i]['Spacing'] = RowWidth
            elif "." in self.Jobs[self.i]['PrjLoc']:
                if int(self.Jobs[self.i]['Latitude']) > 0:
                    Tilt = str(np.abs(int(self.Jobs[self.i]['Latitude']) - 23))
                else:
                    Tilt = str(np.abs(int(self.Jobs[self.i]['Latitude']) + 23))
                self.Jobs[self.i]['Tilt'] = Tilt
                self.Jobs[self.i]['IRR'] = 7.5

                YieldAPSHR = requests.get("https://re.jrc.ec.europa.eu/api/PVcalc?lat=" + str(self.Jobs[self.i]['Latitude']) + "&lon=" + str(self.Jobs[self.i]['Longitude']) + "&peakpower=1&loss=14&aspect=0&angle=" + str(Tilt) + "&pvtechchoice=Unknown&outputformat=csv")
                YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
                YieldAPSHV = YieldAPSH.getvalue()
                if "message" in YieldAPSHV:
                    Pass = 0
                elif "Response [200]" in YieldAPSHV:
                    Pass = 0
                else:
                    Pass = 1
                    YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
                    YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
                    YieldAPSH = pd.read_csv(YieldAPSH, error_bad_lines=False, skipfooter=12, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8], delimiter='\t\t', engine='python')

                Yield = YieldAPSH['E_m'].to_numpy()
                PSH = YieldAPSH['H(i)_m'].to_numpy()

                tf = TimezoneFinder()
                TZ = pytz.timezone(self.tf2.tzNameAt(latitude=float(self.Jobs[i]['Latitude']), longitude=float(self.Jobs[i]['Longitude']),forceTZ=True))
                date = datetime.datetime(2019, 12, 21, hour=15, tzinfo=TZ)
                elevation = get_altitude(float(self.Jobs[self.i]['Latitude']), float(self.Jobs[self.i]['Longitude']), date)
                Width = 1.968
                HeightDifference = np.sin(np.radians(Tilt)) * Width
                ModuleRowSpacing = HeightDifference / np.tan((np.radians(elevation)))
                RowWidth = ModuleRowSpacing + (np.cos((np.radians(Tilt))) * Width)
                self.Jobs[self.i]['Elevation'] = elevation
                self.Jobs[self.i]['Spacing'] = RowWidth
            else:
                with open((self.Locations + "\\" + str(self.Jobs[self.i]['PrjLoc']) + extn)) as f:
                    self.Loc.append(json.load(f))
                X = list(set(self.Loc[self.i].keys()).intersection(self.Jobs[self.i].keys()))
                for dk in X:
                    del self.Loc[self.i][dk]
                self.Jobs[self.i].update(self.Loc[self.i])
            self.i = self.i + 1
        return

    def LoadPan(self):
        P = pd.read_csv(self.PanelData)
        self.EM = list()
        for i in range(len(self.Jobs)):
            if i >= len(P):
                self.Pan = P.iloc[1]
            else:
                try:
                    self.Pan = P[P['PanelID'] == self.Jobs[i]['PanTyp']].to_dict(orient='records')[0]
                except BaseException:
                    self.Pan = P[P['PanelID'] == str(self.Jobs[i]['PanTyp'])].to_dict(orient='records')[0]
                X = list(set(self.Pan.keys()).intersection(self.Jobs[i].keys()))
                for dk in X:
                    del self.Pan[dk]
            self.Jobs[i].update(self.Pan)
        i = 0
        for Job in self.Jobs:
            f = self.Panels + "\\" + str(Job['Tech']) + ".csv"
            self.EM.append(pd.read_csv(f).to_dict(orient='records'))
            X = list(set(self.EM[i][0].keys()).intersection(self.Jobs[i].keys()))
            for dk in X:
                del self.EM[i][0][dk]
            self.Jobs[i].update(self.EM[i][0])
            i = i + 1
        return

    def LoadPan2(self):
        P = pd.read_csv(self.PanelData)
        self.EM = list()
        #self.Jobs =len(self.Jobs))
        for i in range(len(self.Jobs)):
            try:
                self.Pan = P[P['PanelID'] == self.Jobs[i]['PanTyp']].to_dict(orient='records')[0]
            except BaseException:
                self.Pan = P[P['PanelID'] == str(self.Jobs[i]['PanTyp'])].to_dict(orient='records')[0]
            self.Jobs[i].update(self.Pan)
        i = 0
        for Job in self.Jobs:
            f = self.Panels + "\\" + str(Job['Tech']) + ".csv"
            self.EM.append(pd.read_csv(f).to_dict(orient='records'))
            self.Jobs[i].update(self.EM[i][0])
            i = i + 1
        return

    def LoadPan3(self,length,lifetimes):
        P = pd.read_csv(self.PanelData)
        self.EM = list()
        self.Jobs = np.tile(self.Jobs, len(lifetimes))
        for i in range(len(length)):
            try:
                self.Pan = P[P['PanelID'] == self.Jobs[i]['PanTyp']].to_dict(orient='records')[0]
            except BaseException:
                self.Pan = P[P['PanelID'] == str(self.Jobs[i]['PanTyp'])].to_dict(orient='records')[0]
            self.Jobs[i].update(self.Pan)
        i = 0
        for Job in self.Jobs:
            f = self.Panels + "\\" + str(Job['Tech']) + ".csv"
            self.EM.append(pd.read_csv(f).to_dict(orient='records'))
            self.Jobs[i].update(self.EM[i][0])
            i = i + 1
        return

    def LoadTyp(self):
        extn = ".csv"
        self.Typ = list()
        i = 0
        for Job in self.Jobs:
            f = self.Types + "\\" + Job['PrjTyp'] + extn
            self.Typ.append(pd.read_csv(f).to_dict(orient='records'))
            X = list(set(self.Typ[i][0].keys()).intersection(self.Jobs[i].keys()))
            for dk in X:
                del self.Typ[i][0][dk]
            self.Jobs[i].update(self.Typ[i][0])
            i = i + 1
        return

    def get_loc(self):
        Pass = 0
        while Pass == 0:
            lat = str(random.uniform(-35, 60))
            lon = str(random.uniform(-20, 60))
            #lat = str(random.uniform(35, 60))
            #lon = str(random.uniform(-10, 60))
            if float(lat) > 0:
                Tilt = str(np.abs(float(lat) - 23))
            else:
                Tilt = str(np.abs(float(lat) + 23))
            YieldAPSHR = requests.get("https://re.jrc.ec.europa.eu/api/PVcalc?lat=" + lat + "&lon=" + lon + "&peakpower=1&loss=14&aspect=0&angle=" + Tilt + "&pvtechchoice=Unknown&outputformat=csv")
            YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
            YieldAPSHV = YieldAPSH.getvalue()
            if "message" in YieldAPSHV:
                Pass = 0
            elif "Response [200]" in YieldAPSHV:
                Pass = 0
            else:
                Pass = 1
                YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
                # for line in YieldAPSH:
                #    if "Fixed slope of modules (deg.) (optimum at given orientation):" in line:
                #        Tilt = line.split(":")
                #        Tilt = int(Tilt[1])
                #       pass
                YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
                YieldAPSH = pd.read_csv(
                    YieldAPSH, error_bad_lines=False, skipfooter=12, skiprows=[
                        0, 1, 2, 3, 4, 5, 6, 7, 8], delimiter='\t\t', engine='python')
                self.num = self.num + Pass
        return lat, lon, YieldAPSH, float(Tilt)

    def RandomLocGen(self):
        lat, lon, YP, Tilt = self.get_loc()
        Yield = YP['E_m'].to_numpy()
        PSH = YP['H(i)_m'].to_numpy()
        return Yield, PSH, Tilt, lat, lon

    def Fetch_Yeild(self,lat,lon):
        Pass = 0
        while Pass == 0:
            if float(lat) > 0:
                Tilt = str(np.abs(float(lat) - 23))
            else:
                Tilt = str(np.abs(float(lat) + 23))
            YieldAPSHR = requests.get("https://re.jrc.ec.europa.eu/api/PVcalc?lat=" + str(lat) + "&lon=" + str(lon) + "&peakpower=1&loss=14&aspect=0&angle=" + Tilt + "&pvtechchoice=Unknown&outputformat=csv")
            YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
            YieldAPSHV = YieldAPSH.getvalue()
            YieldAPSH = io.StringIO(YieldAPSHR.content.decode('utf-8'))
            if "message" in YieldAPSHV:
                self.Jobs.remove(self.Jobs[self.i])
                return 0, 0, 0
            #    Pass = 0
            elif "Response [200]" in YieldAPSHV:
                return 0, 0, 0
            #    Pass = 0
            Pass = 1
        YieldAPSH = pd.read_csv(YieldAPSH, error_bad_lines=False, skipfooter=12, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8],delimiter='\t\t', engine='python')
        Yield = YieldAPSH['E_m'].to_numpy()
        PSH = YieldAPSH['H(i)_m'].to_numpy()
        return Yield, PSH, float(Tilt)



class Que:

    def __init__(self, filename, paneldatafile):
        self.Tech = "NoEnhancment"
        self.TimStp = "hour"
        self.RenInf = 2.1
        self.RenCos = 0.5
        self.OprCos = 0.01
        self.InvCosInf = 2.1
        self.OprCosInf = 2.1
        self.PrjLoc = "Random"
        self.PrjTyp = "GroundmountPVArray"
        self.ModSta = "01/05/2019"
        self.PrjLif = 20
        self.InvLif = 10
        self.PanFlrPri = 0.245
        self.PanCosInf = -1
        self.PanTyp = '4110'
        self.ProjectName = "0"
        self.filename = filename
        self.paneldatafile = paneldatafile
        with open(self.filename + '.json') as params:
            paramsDict = json.load(params)
            key = list(paramsDict.keys())
            value = list(paramsDict.values())
            for idx, val in enumerate(value):
                for element in value[idx]:
                    if type(element) == str:
                        if "#" in element:
                            element = element.split('#')
                            if "X" in element[0]:
                                value[idx] = [value[idx][0]] * int(element[1])
                            if "CA" in element[0]:
                                value[idx] = np.arange(value[idx][0], value[idx][0] + int(element[1]), 1, dtype='int')
                            if "F" in element[0]:
                                #value[idx] =
                                Fa = pd.read_csv(value[idx][0])
                                keyf = list(Fa.columns.values)
                                valuef = [list(Fa[col].to_numpy().astype('str')) for col in keyf]
                            if "L" in element[0]:
                                value[idx] = np.linspace(value[idx][1],value[idx][2],value[idx][3])

        self.key = key
        self.value = value
        try:
            self.key = self.key + keyf
            self.value = self.value + valuef
        except:
            self.key = key
            self.value = value

    def Declare(self, **kwargs):
        keys, values = kwargs.items()
        for idx in range(len(keys)):
            setattr(self, keys[idx], values[idx])
        return

    def GenFile(self):
        Jobs = list(itertools.product(*self.value))
        Jobs = np.vstack(Jobs)
        Jobs = pd.DataFrame(data=Jobs, index=None, columns=self.key)
        Jobs.to_csv(self.filename + ".csv", index=False)
        return

    def SaveQue(self):
        self.filename = self.filename.split('.')[0]

        JB = JobQue(self.filename + ".csv", self.paneldatafile)  # Initialies job que object
        JB.LoadQue()  # Loads RunQue as job que object
        JB.LoadLoc()  # Loads locations in job que object
        JB.LoadPan()  # Loads panel in job que objec # Loads panel in job que object
        JB.LoadTyp()  # Load panel type in job que object
        with open(self.filename + '.JBS', 'wb') as handle:
            pickle.dump(JB.Jobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


class EPC:
    #  the Initialises the EPC object and calculates all economic factors

    def __init__(self, job):
        self.Design = job['Design'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.Construction = job['Construction'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.Framing = job['Framing'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.DCcabling = job['DCcabling'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.ACcabling = job['ACcabling'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.CivilWorkPanels = job['CivilWork(Panels)'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.CivilWorkGeneral = job['CivilWork(general)'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.PVPanels = job['PVPanels'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.FixedProjectCosts = job['FixedProjectCosts'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.FreightPanels = job['Freight(Panels)'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.FreightOther = job['Freight(other)'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.Inverters = job['Inverters'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.Controls = job['Controls'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.OriginalCost = self.Design + self.Construction + self.Framing + self.DCcabling + self.ACcabling + self.CivilWorkPanels + self.CivilWorkGeneral + self.PVPanels + self.FixedProjectCosts + self.FreightPanels + self.FreightOther + self.Inverters + self.Controls
        #self.OriginalCost = job['Design'] + job['Construction'] + job['Framing'] + job['DCcabling'] + job['ACcabling'] + job['CivilWork(Panels)'] + job['CivilWork(general)'] + job['PVPanels'] + job['FixedProjectCosts'] + job['Freight(Panels)'] + job['Freight(other)'] + job['Inverters'] + job['Controls']
        self.PriceExcludingPanels = self.OriginalCost - (job['PVPanels'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000)))
        self.PanelSize = 410
        self.NumberOfPanels = 1000 * (job['PVSize'] / self.PanelSize)
        self.InstallCostPerPanel = self.PriceExcludingPanels / self.NumberOfPanels
        self.InverterCost = job['Inverters'] * (job['DollarPerWatt'] * (job['PVSize'] * 1000))
        self.OldArea = job['SystemArea']
        self.PanelCost = job['Cost']
        self.EqRatingPanels = job['PowerDensity'] * 1.968 * 0.992
        self.RequiredNumberPanels = 1000 * job['PVSize'] / self.EqRatingPanels
        self.InstallationCostExcPanels = self.RequiredNumberPanels * self.InstallCostPerPanel
        self.PanelCost2 = self.PanelCost * 1000 * job['PVSize']
        self.NewPrice = self.InstallationCostExcPanels + self.PanelCost2
        self.InverterCostAsPercentofCiepPrice = self.InverterCost / self.InstallationCostExcPanels
        self.NewArea = ((((1.92 * math.cos(math.radians(job['Tilt']))) * 2 + job['Spacing']) * 0.99) / 2) * self.RequiredNumberPanels


class TechTime:
    #  Initialises the techtime object
    def __init__(self, job):
        from datetime import datetime
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
        elif self.TimeStepString == "halfhour":
            self.Advance = relativedelta(minutes=30)
            self.GHDivisor = D * 48
            self.AdvanceInt = 1 / 48
            self.InterestDivisor = 17520
            self.Entrants = (self.EndDate - self.StartDate).days * 48

        self.Dates = np.empty(self.Entrants, dtype=datetime)
        I = np.linspace(0, self.Entrants, self.Entrants, dtype=np.int)
        self.Dates[:] = self.StartDate + (self.Advance * I)

    def DateAdvance(self):
        self.CurrentDate = self.CurrentDate + self.Advance
        return


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
        self.PowerDensity = job['PowerDensity']
        try:
            self.ET = job['7']
        except BaseException:
            self.ET = 'R'
        self.HoursInEn = 0

    # Requests irradiance data from PVGIS
    def PVGIS(self, time):
        # Requests and reformats PVGIS data

        self.PVGISData = requests.get('https://re.jrc.ec.europa.eu/api/seriescalc?' + 'lat=' + str(self.Latitude) + '&lon=' + str(self.Longitude) + '&angle=' + str(self.Tilt) + '&startyear=2015&endyear=2015')
        self.PVGISData = io.StringIO(self.PVGISData.content.decode('utf-8'))
        #self.Temp = io.StringIO(self.PVGISData.content.decode('utf-8'))
        #print(pd.read_csv(copy.deepcopy(self.PVGISData), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)']))
        self.Temp = pd.read_csv(copy.deepcopy(self.PVGISData), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'T2m']).to_numpy()
        self.Temp = self.Temp[:,1]
        #print(self.Temp)
        #self.WindSpeed = io.StringIO(self.PVGISData.content.decode('utf-8'))
        self.WindSpeed = pd.read_csv(copy.deepcopy(self.PVGISData), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python',usecols=['time', 'WS10m']).to_numpy()
        self.WindSpeed = self.WindSpeed[:,1]
        self.PVGISData = pd.read_csv(self.PVGISData, skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)'])
        self.PVGISData = self.PVGISData.to_numpy()
        # For loop reformats date
        for i in range(len(self.PVGISData)):
            self.PVGISData[:, 0][i] = datetime.datetime.strptime(self.PVGISData[:, 0][i][:-2], '%Y%m%d:%H')
            #self.PVGISData[:, 0][i] = self.PVGISData[:, 0][i].replace(year=2016)
        Shift = np.where(self.PVGISData[:, 0][:] == time.StartDate)[0][0]  # Identifies index of start date in PVGIS Data
        self.PVGISData = np.roll(self.PVGISData, -Shift * 2)  # Shifts starts date to index = 0
        self.Dates = self.PVGISData[:, 0]
        self.Irradiance = self.PVGISData[:, 1]
        #except BaseException:
         #   ttime.sleep(2)
         #   self.PVGIS(time)
        return

    def PVGIS_HalfHour(self, time):
        # Requests and reformats PVGIS data
        req = 'https://re.jrc.ec.europa.eu/api/seriescalc?' + 'lat=' + str(self.Latitude) + '&lon=' + str(self.Longitude) + '&angle=' + str(self.Tilt) + '&startyear=2015&endyear=2015'
        self.PVGISData = requests.get('https://re.jrc.ec.europa.eu/api/seriescalc?' + 'lat=' + str(self.Latitude) + '&lon=' + str(self.Longitude) + '&angle=' + str(self.Tilt) + '&startyear=2015&endyear=2015')
        self.PVGISData = io.StringIO(self.PVGISData.content.decode('utf-8'))
        self.PVGISData = pd.read_csv(self.PVGISData, skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)'])
        self.PVGISData = self.PVGISData.to_numpy()
        # For loop reformats date
        for i in range(len(self.PVGISData)):
            self.PVGISData[:, 0][i] = datetime.datetime.strptime(self.PVGISData[:, 0][i][:-2], '%Y%m%d:%H')
            self.PVGISData[:, 0][i] = self.PVGISData[:, 0][i].replace(year=2015)
        Shift = np.where(self.PVGISData[:, 0][:] == time.StartDate)[0][0]  # Identifies index of start date in PVGIS Data
        self.PVGISData = np.roll(self.PVGISData, -Shift * 2)  # Shifts starts date to index = 0
        g_half_hours = self.PVGISData[:, 1]
        g_half_hours = np.insert(g_half_hours, -1, 0)
        g_half_hours = (g_half_hours[1:] + g_half_hours[:-1])/2
        t_half_hours = self.PVGISData[:, 0] + relativedelta(minutes=30)
        T = np.stack((t_half_hours,g_half_hours), axis=-1)
        T = np.vstack((self.PVGISData, T))
        T = T[T[:, 0].argsort()]
        self.PVGISData = T
        self.Dates = self.PVGISData[:, 0]
        self.Irradiance = self.PVGISData[:, 1]
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
        if time.TimeStepString == 'hour' or 'halfhour':
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
        elif time.TimeStepString == 'halfhour':
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
            self.EffectiveMultiplier()
            self.EffectiveCapacity = self.Capacity * self.EM
        elif time.TimeStepString == 'month':
            self.EffectiveCapacity = self.Capacity
        elif time.TimeStepString == 'halfhour':
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
        if time.TimeStepString == 'hour':
            self.PVGIS(time)
        elif time.TimeStepString == 'halfhour':
            self.PVGIS_HalfHour(time)
        else:
            self.Dates = time.Dates
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
        elif self.ET == 'MaxLinear':
            PCE0Max = 30
            x = np.linspace(1,1000,1000)
            PCE1 = self.PowerDensity/9.8
            EM0 = PCE0Max/PCE1
            print(PCE1, EM0)
            y = EM0/(1000-x)
            plt.plot(x,y)
            plt.show()


        else:
            A = np.exp(-self.GR * (self.Irradiance - self.X))
            self.EM = self.LA + ((self.UA - self.LA) / (self.C + self.Q * A)) ** (1 / self.MG)
        self.EM[WhereZero] = 0
        return


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


class Finance:

    # Initialise the finance object
    def __init__(self, job, e, t, p, i):
        self.Dates = p.Dates
        self.PVSize = p.PVSize
        self.PanelLifetime = p.Lifetime
        self.InverterLifetime = i.Lifetime
        self.InitialCost = e.PanelCost2 + e.InstallationCostExcPanels
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
        self.PanelReplacementCostPV[self.PanelReplacements] = 1000 * self.PVSize * self.PanelPrice[self.PanelReplacements]

        i = np.linspace(0, len(self.Dates), len(self.Dates))
        self.PanelReplacementCostOther = np.zeros(len(self.Dates))
        self.PanelReplacementCostOther[self.PanelReplacements] = (self.NewPrice * 0.1) * np.power(
            (1 + self.InverterCostInflation), (((i[self.PanelReplacements] / self.InterestDivisor) / 365) - 1))

        self.PaneReplacementCost = self.PanelReplacementCostPV + self.PanelReplacementCostOther

        self.InverterReplacements = np.where(
            self.InverterLifetime < ilr)[0][:-1]
        self.InverterReplacementCost = np.zeros(len(self.Dates))
        self.InverterReplacementCost[self.InverterReplacements] = (self.InstallationCostExcPanels * self.InverterCostAsPercentofCiepPrice) * np.power(1 + self.InverterCostInflation, (((i[self.InverterReplacements] / self.InterestDivisor) / 365) - 1))

        return

    # Caluculates the reocuring costs of the project
    def recurring_costs(self):
        self.OAM = np.zeros(len(self.Dates))
        self.LandRental = np.zeros(len(self.Dates))
        self.OAM[0] = (1000 * self.PVSize * 0.01) / self.InterestDivisor
        self.LandRental[0] = self.RentCost * self.NewArea / self.InterestDivisor

        for i in range(1, len(self.Dates)):
            self.OAM[i] = self.OAM[i - 1] * (1 + (self.OperationCostInflation / self.InterestDivisor))
            self.LandRental[i] = self.LandRental[i - 1] * (1 + (self.OperationCostInflation / self.InterestDivisor))

        return

    # Calculates the finances of the project
    def Costs(self):
        self.panel_price()
        self.replacement()
        self.recurring_costs()

        self.TotalCosts = self.PaneReplacementCost + self.InverterReplacementCost + self.OAM + self.LandRental
        return

    # Calculates the LCOE at the end of the project
    def LCCACalculate(self):

        Data = pd.DataFrame()
        Data['Settlement Date'] = self.Dates
        Data['Settlement Date'] = [x.replace(tzinfo=pytz.UTC) for x in Data['Settlement Date']]
        Data['Generation'] = self.PVGen / 1000
        Data = Data.set_index('Settlement Date')

        Baseline = Setup('Data/2015RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
        Baseline = Scaling(Baseline, 1, 1, 0, 0)
        Baseline = Expand_Generation(Baseline, 20)
        Baseline = Expand_Sacler(Baseline, 20)
        for Asset in Baseline.Mix['Technologies']:
            if Asset['Technology'] == 'SolarBTMNT':
                Baseline.Mix['Technologies'].remove(Asset)
        Baseline = Grid.Demand(Baseline)
        DistBaseline = Dispatch(Baseline)
        CarbonEmissionsBaseline = DistBaseline.CarbonEmissions

        FarmAdded = Setup('Data/2015RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
        FarmAdded = Scaling(FarmAdded, 1, 1, 1, 0)
        FarmAdded = Expand_Generation(FarmAdded, 20)
        FarmAdded = Expand_Sacler(FarmAdded, 20)
        for Asset in FarmAdded.Mix['Technologies']:
            if Asset['Technology'] == 'SolarBTMNT':
                FarmAdded.Mix['Technologies'].remove(Asset)
        FarmAdded.EndDate = FarmAdded.EndDate.replace(year=2015+20)
        FarmAdded = Grid.Demand(FarmAdded)
        FarmAdded = Add_to_SolarNT(FarmAdded, Data)
        FarmAdded = Grid.MatchDates(FarmAdded)
        DistFarmAdded = Dispatch(FarmAdded)
        CarbonEmissions = DistFarmAdded.CarbonEmissions
        CarbonSavings = (CarbonEmissionsBaseline - CarbonEmissions)/2 * (1*10**-3)

        i1 = np.linspace(0, len(self.Dates), len(self.Dates))
        i2 = np.linspace(0, len(CarbonSavings.index.to_numpy()), len(CarbonSavings.index.to_numpy()))
        tc = self.TotalCosts
        pv = CarbonSavings
        ii1 = i1[:] / self.InterestDivisor
        ii2 = i2[:] / self.InterestDivisor
        self.LCCA = (self.NewPrice + np.abs(self.xnpv(self.DCR, tc[:], ii1[:]))) / self.xnpv(self.DCR, pv[:], ii2[:])

        return

    def LCOECalculate(self):
        i = np.linspace(0, len(self.Dates), len(self.Dates))
        tc = self.TotalCosts[:]
        pv = self.PVGen[:]
        ii = i[:] / self.InterestDivisor
        self.LCOE = (self.NewPrice + np.abs(self.xnpv(self.DCR, tc[:], ii[:]))) / self.xnpv(self.DCR, pv[:], ii[:])
        return

    def CostsSums(self):
        self.TotalCostsSum = np.sum(self.TotalCosts)
        self.PaneReplacementCostSum = np.sum(self.PaneReplacementCost)
        self.InverterReplacementCostSum = np.sum(self.InverterReplacementCost)
        self.OAMSum = np.sum(self.OAM)
        self.LandRentalSum = np.sum(self.LandRental)
        return

    def xnpv(self, dcr, values, date):
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
        self.count = 0

    # Outputs the results as an excel file
    def Finance(self):
        df = pd.Dataframe()
        df["Total Costs"] = np.sum(self.Finance.TotalCosts)
        df["Panel Replacement Costs"] = np.sum(self.Finance.PaneReplacementCost)
        df["Inverter Replacment Costs"] = np.sum(self.Finance.InverterReplacementCost)
        df["Operational and Maintenance"] = np.sum(self.Finance.OAM)
        df["Land Rent"] = np.sum(self.Finance.LandRental)
        df.to_csv(str(self.Job['PrjLoc']) + str(self.Job['Tech']) + str(self.Job["PanTyp"]) + ".csv")
        return

    def Excel(self):
        CFC = [
            'Date',
            'ProjectYear',
            'PanelLifetime',
            'InverterLifetime',
            'PeakSunHours',
            'CumilativeSunHours',
            'Burn-inAbsolute',
            'LongTermDegredation',
            'LongTermDegredationAbsolute',
            'PanelStateofHealth',
            'PeakCapacity',
            'EffectiveCapacity',
            'MonthlyYeild',
            'PVGeneration',
            'CapitalCost',
            'RefurbishmentCost(Panels-PV)',
            'RefurbishmentCost(Panels-Other)',
            'RefurbishmentCost(Panels)',
            'PanelPriceThisYear',
            'RefurbishmentCost(Inverter)',
            'AnnualO&MCost',
            'LandRental',
            'TotalCost',
            'CostCheck',
            'LCOE',
            'Enhancment']
        df = pd.DataFrame(self.Panel.Dates, columns=['Date'])
        df['Irradiance'] = pd.Series(self.Panel.Irradiance, index=df.index)
        df['Panel Lifetime'] = pd.Series(self.Panel.Lifetime, index=df.index)
        df['Inverter Lifetime'] = pd.Series(self.Inverter.Lifetime, index=df.index)
        df['Peak Sun Hours'] = pd.Series(self.Panel.PSH, index=df.index)
        df['Cumilative Sun Hours'] = pd.Series(self.Panel.CPSH, index=df.index)
        df['Burn-in Abs'] = pd.Series(self.Panel.BurnInAbs, index=df.index)
        df['Long Term Degradation'] = pd.Series(self.Panel.LongTermDeg, index=df.index)
        df['Long Term Degradation Abs'] = pd.Series(self.Panel.LongTermDegAbs, index=df.index)
        df['Panel State of Health'] = pd.Series(self.Panel.StateOfHealth, index=df.index)
        df['Peak Capacity'] = pd.Series(self.Panel.Capacity, index=df.index)
        df['Effective Capacity'] = pd.Series(self.Panel.EffectiveCapacity, index=df.index)
        df['Monthly Yield'] = pd.Series(self.Panel.Yield, index=df.index)
        df['PV Generation'] = pd.Series(self.Panel.PVGen, index=df.index)
        df['Refurbishment Cost (PV)'] = pd.Series(self.Finance.PanelReplacementCostPV, index=df.index)
        df['Refurbishment Cost (Other)'] = pd.Series(self.Finance.PanelReplacementCostOther, index=df.index)
        df['Refurbishment Cost (Panels)'] = pd.Series(self.Finance.PaneReplacementCost, index=df.index)
        df['Panel Price This Year'] = pd.Series(self.Finance.panel_price, index=df.index)
        df['Refurbishment Cost (Inverter)'] = pd.Series(self.Finance.InverterReplacementCost, index=df.index)
        df['Annual O&M Cost'] = pd.Series(self.Finance.OAM, index=df.index)
        df['Land Rental'] = pd.Series(self.Finance.LandRental, index=df.index)
        df['Total Cost'] = pd.Series(self.Finance.TotalCosts, index=df.index)
        df['LCOE'] = pd.Series(self.Finance.LCOE, index=df.index)
        df['Enhancment'] = pd.Series(self.Panel.EM, index=df.index)
        df.to_csv(str(self.Job['PrjLoc']) + str(self.Job['Tech']) + str(self.Job["PanTyp"]) + ".csv")
        return

    # Outputs the results specified at the first line of Results.csv file
    def Results(self):

        File = pd.read_csv('Results.csv')
        ResultsRequested = File.columns.values
        ResultsOutput = list()
        for Result in ResultsRequested:
            Result = Result.split('.')
            if Result[0] == 'Finance':
                Result = getattr(self.Finance, Result[1])
                ResultsOutput.append(Result)
            elif Result[0] == 'Panel':
                Result = getattr(self.Panel, Result[1])
                ResultsOutput.append(np.average(Result))
            elif Result[0] == 'Inverter':
                Result = getattr(self.Inverter, Result[1])[-1]
                ResultsOutput.append(Result)
            elif Result[0] == 'EPC':
                Result = getattr(self.EPC, Result[1])[-1]
                ResultsOutput.append(Result)
            else:
                Result = self.Job[Result[1]]
                ResultsOutput.append(Result)
        ResultO = pd.DataFrame([ResultsOutput], columns=ResultsRequested)
        File = File.append(ResultO, ignore_index=True)
        File.to_csv('Results.csv', index=False)
        return

    def M(self, a):
        return a.month

    def T(self, a):
        return a.hour


class LCOE:

    def __init__(self, filename, paneldatafile):
        self.filename = filename
        self.paneldatafile = paneldatafile
        self.ResultsLoc = 'Results.csv'
        self.Results = pd.read_csv(self.ResultsLoc)
        self.ResultsCol = self.Results.columns
        self.EmptyResults = pd.DataFrame(columns=self.ResultsCol)
        self.Resutls = self.Results['Finance.LCOE'].to_numpy()

    def GenerateJBS(self):
        self.Q = Que(self.filename, self.paneldatafile)
        self.Q.GenFile()
        self.Q.SaveQue()
        return

    def LoadJBS(self):
        self.Q = JobQue(self.filename + '.json', self.paneldatafile)
        self.Q.ReRun(self.filename + '.JBS')
        self.Q.LoadPan()
        return


    def Variations(self, devices, variations):
        for idx, devices in enumerate(devices):
            self.Q.Modify('Tech', variations[idx])
            self.Q.Modify('PanTyp', devices)
        return

    def init(self, l,):
        global lock
        lock = l

    def WorkerLCCA(self, job):
        E = EPC(job)
        t = TechTime(job)
        P = Panel(job)
        P.Simulate(t)
        I = Inverter(job, t)
        I.Simulate()
        F = Finance(job, E, t, P, I)
        F.Costs()
        F.LCOECalculate()
        F.LCCACalculate()
        F.CostsSums()
        O = Out(job, E, t, P, I, F)
        lock.acquire()
        O.Results()
        #O.Finance()
        #O.Excel()
        lock.release()
        return


    def Worker(self, job):
        E = EPC(job)
        t = TechTime(job)
        P = Panel(job)
        P.Simulate(t)
        I = Inverter(job, t)
        I.Simulate()
        F = Finance(job, E, t, P, I)
        F.Costs()
        F.LCOECalculate()
        O = Out(job, E, t, P, I, F)
        lock.acquire()
        O.Results()
        #O.Excel()
        lock.release()
        return

    def WorkerNonMP(self, job):
        E = EPC(job)
        t = TechTime(job)
        P = Panel(job)
        P.Simulate(t)
        I = Inverter(job, t)
        I.Simulate()
        F = Finance(job, E, t, P, I)
        F.Costs()
        F.LCOECalculate()
        O = Out(job, E, t, P, I, F)
        O.Results()
        return O

    def RunLCCA(self):
        l = multiprocessing.Lock()
        with tqdm.tqdm(total=(len(self.Q.Jobs))) as pbar:
            with Pool(processes=multiprocessing.cpu_count() - 1, initializer=self.init, initargs=(l,)) as pool:
                for i, _ in enumerate(pool.imap_unordered(self.WorkerLCCA, self.Q.Jobs)):
                    pbar.update()
                pool.close()
                pool.join()
        return

    def Run(self):
        l = multiprocessing.Lock()
        with tqdm.tqdm(total=(len(self.Q.Jobs))) as pbar:
            with Pool(processes=multiprocessing.cpu_count() - 1, initializer=self.init, initargs=(l,)) as pool:
                for i, _ in enumerate(pool.imap_unordered(self.Worker, self.Q.Jobs)):
                    pbar.update()
                pool.close()
                pool.join()
        return

    def FetchReults(self):
        self.Results = pd.read_csv(self.ResultsLoc)
        self.Results = self.Results.sort_values(by=['Job.PanTyp'])
        self.Results = self.Results['Finance.LCOE'].to_numpy()
        self.EmptyResults.to_csv(self.ResultsLoc, index=False)
        return self.Results

    def EmptyResults(self):
        self.EmptyResults.to_csv(self.ResultsLoc, index=False)
        return