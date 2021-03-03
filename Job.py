import pandas as pd
import json
import requests
import random
import io
import pickle
from pysolar.solar import *
import datetime
import numpy as np
import pytz
from timezonefinder import TimezoneFinder


class JobQue:
    def __init__(self, que):
        self.QueFile = que
        self.Locations = r'Data\Location'
        self.Panels = r'Data\Panel'
        self.PanelData = r'Data\PanelData.csv'
        self.Types = r'Data\Type'
        self.num = 0
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
        self.QueDataset = pd.read_csv(self.QueFile)
        self.Jobs = self.QueDataset.to_dict(orient='records')
        return

    def LoadLoc(self):
        extn = ".json"
        self.Loc = list()
        i = 0
        for Job in self.Jobs:
            if Job['PrjLoc'] == 'Random':
                Yield, PSH, Tilt, lat, lon = self.RandomLocGen()
                self.Jobs[i]['Yield'] = Yield
                self.Jobs[i]['PeakSunHours'] = PSH
                self.Jobs[i]['Tilt'] = Tilt
                self.Jobs[i]['Latitude'] = lat
                self.Jobs[i]['Longitude'] = lon
                self.Jobs[i]['IRR'] = 7.5
                tf = TimezoneFinder()
                TZ = pytz.timezone(tf.closest_timezone_at(lat=float(lat), lng=float(lon), delta_degree=10))
                date = datetime.datetime(2019, 12, 21, hour=15, tzinfo=TZ)
                elevation = get_altitude(float(lat), float(lon), date)
                Width = 1.968
                HeightDifference = np.sin(np.radians(Tilt)) * Width
                ModuleRowSpacing = HeightDifference / np.tan((np.radians(elevation)))
                RowWidth = ModuleRowSpacing + (np.cos((np.radians(Tilt))) * Width)
                self.Jobs[i]['Elevation'] = elevation
                self.Jobs[i]['Spacing'] = RowWidth

            else:
                with open((self.Locations + "\\" + str(Job['PrjLoc']) + extn)) as f:
                    self.Loc.append(json.load(f))
                self.Jobs[i].update(self.Loc[i])
            i = i + 1
        return

    def LoadPan(self):
        P = pd.read_csv(self.PanelData)
        self.Pan = list()
        self.EM = list()
        for i in range(len(self.Jobs)):
            #print(P.iloc[self.Jobs[i]['PanTyp']].to_dict())
            #print(P[P['PanelID'] == self.Jobs[i]['PanTyp']].T.to_dict()[i])
            self.Pan.append(P[P['PanelID'] == self.Jobs[i]['PanTyp']].T.to_dict()[i])
            self.Jobs[i].update(self.Pan[i])
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
            self.Jobs[i].update(self.Typ[i][0])
            i = i + 1
        return

    def get_loc(self):
        Pass = 0
        while Pass == 0:
            lat = str(random.randint(-35, 60))
            lon = str(random.randint(20, 30))
            if int(lat) > 0:
                Tilt = str(np.abs(int(lat) - 23))
            else:
                Tilt = str(np.abs(int(lat) + 23))
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
        return lat, lon, YieldAPSH, int(Tilt)

    def RandomLocGen(self):
        lat, lon, YP, Tilt = self.get_loc()
        Yield = YP['E_m'].to_numpy()
        PSH = YP['H(i)_m'].to_numpy()
        return Yield, PSH, Tilt, lat, lon
