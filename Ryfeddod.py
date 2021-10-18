import difflib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
import os
import json
import pickle
import pytz
import requests
import io
#from pvlive_api import PVLive
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
import xml.etree.ElementTree as et
from dateutil.relativedelta import *
import calendar

class Grid:

    def __init__(self,MixDir):
        self.BMRSKey = "zz6sqbg3mg0ybyc"
        self.ENTSOEKey = "6f7dd5a8-ca23-4f93-80d8-0c6e27533811"

        with open(MixDir) as Mix_File:
            self.Mix = json.load(Mix_File)

        DataSources = set()
        for Tech in self.Mix["Technologies"]:
            DataSources.add(Tech["Source"])

        self.StartDate = datetime.strptime(self.Mix['StartDate'],'%Y-%m-%d')
        self.EndDate = datetime.strptime(self.Mix['EndDate'], '%Y-%m-%d')
        self.timezone = 'Europe/Prague'

        if self.Mix["Country"] == "UUK":
            if "BMRS" in DataSources:
                self.BMRSFetch()
            if "PVLive" in DataSources:
                self.PVLiveFetch()

            for Tech in self.Mix['Technologies']:
                if Tech['Source'] == "BMRS":
                    Tech['Generation'] = self.BMRSData[str(Tech['Technology'])]
                    Tech['Generation'] = Tech['Generation'].rename('Generation')
                if Tech['Source'] == "PVLive":
                    Tech['Generation'] = self.PVLiveData['generation_mw']
                    Tech['Generation'] = Tech['Generation'].rename('Generation')
        else:
            if "ENTSOE" in DataSources:
                self.domain = self.Mix['Domain']
                self.ENTSOEFetch()

            for Tech in self.Mix['Technologies']:
                if Tech['Source'] == 'ENTSOE':
                    Tech['Generation'] = self.ENTSOEData[str(Tech['Technology'])]
                    Tech['Generation'] = Tech['Generation'].rename('Generation')

    def convert_period_format(self, date_obj, timezone):
        timezone = pytz.timezone(timezone)
        date_obj = timezone.localize(date_obj)
        date_obj = date_obj.astimezone(pytz.utc)
        api_format = date_obj.strftime('%Y%m%d%H%M')
        return api_format

    def ENTSOECodes(self):
        E = pd.read_csv('C:\\Users\Cai Williams\PycharmProjects\Ryfeddod\Data\ENTSOELocations.csv')
        N = ['Name 0', 'Name 1', 'Name 2']

        for X in N:
            EX = E[X].dropna().to_list()
            DN = [string for string in EX if self.domain in string]
            if len(DN) > 0:
                break

        Code = E[E[X].isin([DN[0]])]['Code'].values[0]
        print(Code)
        return Code

    def aggregated_generation(self, start_period, end_period):
        base = 'https://transparency.entsoe.eu/api?'
        security_token = 'securityToken=' + str(self.ENTSOEKey)
        document_type = 'documentType=A75'
        process_type = 'processType=A16'
        in_domain = 'in_domain='+str(self.ENTSOECodes())
        period_start = 'periodStart=' + str(start_period)
        period_end = 'periodEnd=' + str(end_period)
        api_call = base + security_token + '&' + document_type + '&' + process_type + '&' + in_domain + '&' + period_start + '&' + period_end
        api_answer = requests.get(api_call)
        return api_answer

    def time_res_to_delta(self,res):
        res = res.replace('PT', '')
        res = res.replace('M', '')
        res = float(res)
        return timedelta(minutes=res)

    def position_to_time(self, pos, res, start):
        return [start + (res * x) for x in pos]

    def type_code_to_text(self, asset_type):
        codes = pd.read_csv('EUPsrType.csv')
        print(asset_type)
        return codes[codes['Code'] == asset_type]['Meaning'].values[0]

    def match_dates(self, dic):
        index_values = [dic[x][0] for x in dic.keys()]
        common_index_values = list(set.intersection(*map(set, index_values)))
        for x in dic.keys():
            times = np.unique(dic[x][0])
            mask = np.in1d(times, common_index_values)
            mask = np.where(mask)[0]
            dic[x] = dic[x][:, mask]
        return dic

    def ENTSOEFetch(self):
        start_period = self.convert_period_format(self.StartDate, self.timezone)
        end_period = self.convert_period_format(self.EndDate, self.timezone)
        data = self.aggregated_generation(start_period, end_period)
        root = et.fromstring(data.content)
        #root = tree.getroot()
        ns = {'d': 'urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0'}
        entsoe_data = {}
        for GenAsset in root.findall('d:TimeSeries', ns):
            asset_type = GenAsset.find('d:MktPSRType', ns)
            asset_type = asset_type.find('d:psrType', ns).text
            asset_type = self.type_code_to_text(asset_type)
            data = GenAsset.find('d:Period', ns)
            dates = data.find('d:timeInterval', ns)
            start_date = datetime.strptime(dates.find('d:start', ns).text, '%Y-%m-%dT%H:%MZ')
            resolution = data.find('d:resolution', ns).text
            resolution = self.time_res_to_delta(resolution)
            generation = data.findall('d:Point', ns)
            time = [float(x.find('d:position', ns).text) for x in generation]
            time = self.position_to_time(time, resolution, start_date)
            generation = [float(x.find('d:quantity', ns).text) for x in generation]
            tmp = np.vstack((time, generation))
            if asset_type in entsoe_data:
                tmp2 = entsoe_data.get(asset_type)
                tmp2 = np.hstack((tmp2, tmp))
                entsoe_data[asset_type] = tmp2
            else:
                entsoe_data[asset_type] = tmp
        entsoe_data = self.match_dates(entsoe_data)
        entsoe_data_pd = pd.DataFrame()
        for asset in entsoe_data.keys():
            entsoe_data_pd[asset] = entsoe_data[asset][1]
            entsoe_data_pd['Settlement Date'] = entsoe_data[asset][0]
        self.Dates = entsoe_data_pd['Settlement Date']
        entsoe_data_pd = entsoe_data_pd.set_index('Settlement Date')
        entsoe_data_pd = entsoe_data_pd.fillna(0)
        self.ENTSOEData = entsoe_data_pd
        return self.ENTSOEData

    def BMRSFetch(self):
        NumDays = (self.EndDate - self.StartDate).days
        Days = [self.StartDate + timedelta(days=1 * Day) for Day in range(0,NumDays+1)]
        DaysStr = [Day.strftime('%Y-%m-%d') for Day in Days]
        AllAPIRequests = ['https://api.bmreports.com/BMRS/B1620/V1?APIKey='+ self.BMRSKey +'&SettlementDate=' + SettlementDate + '&Period=*&ServiceType=csv'for SettlementDate in DaysStr]
        AllAPIAnswers = [requests.get(APIrequest) for APIrequest in AllAPIRequests]
        ALLAPIDataframes = [pd.read_csv(io.StringIO(Answer.text), skiprows=[0, 1, 2, 3], skipfooter=1, engine='python',index_col=False).sort_values('Settlement Period') for Answer in AllAPIAnswers]
        YearDataframe = pd.concat(ALLAPIDataframes, ignore_index=True)
        YearDataframe = YearDataframe.drop(columns=['*Document Type', 'Business Type', 'Process Type', 'Time Series ID', 'Curve Type', 'Resolution','Active Flag', 'Document ID', 'Document RevNum'])
        YearDataframe = YearDataframe.pivot_table(index=['Settlement Date', 'Settlement Period'], columns='Power System Resource  Type', values='Quantity')
        YearDataframe = YearDataframe.reset_index()
        YearDataframe["Settlement Period"] = [timedelta(minutes=int((Period) * 30)) for Period in YearDataframe['Settlement Period']]
        YearDataframe['Settlement Date'] = pd.to_datetime(YearDataframe['Settlement Date'], format='%Y-%m-%d')
        YearDataframe['Settlement Date'] = YearDataframe['Settlement Date'] + YearDataframe['Settlement Period']
        YearDataframe = YearDataframe.drop(columns=['Settlement Period'])
        timezone = pytz.timezone('Europe/London')
        YearDataframe['Settlement Date'] = [t.replace(tzinfo=timezone) for t in YearDataframe['Settlement Date']]
        YearDataframe['Settlement Date'] = [t.astimezone(pytz.utc) for t in YearDataframe['Settlement Date']]
        self.Dates = YearDataframe['Settlement Date']
        YearDataframe = YearDataframe.set_index("Settlement Date")
        YearDataframe = YearDataframe.fillna(0)
        self.BMRSData = YearDataframe
        return self.BMRSData

    def PVLiveFetch(self):
        pvl = PVLive()
        tz = pytz.timezone('Europe/London')
        self.StartDate = tz.localize(self.StartDate)
        self.EndDate = tz.localize(self.EndDate)
        self.PVLiveData = pvl.between(self.StartDate,self.EndDate,dataframe=True)
        #self.PVLiveData['datetime_gmt'] = [t.replace(tzinfo=None) for t in self.PVLiveData['datetime_gmt']]
        self.PVLiveData = self.PVLiveData.sort_values(by=['datetime_gmt'])
        self.PVLiveData = self.PVLiveData.set_index('datetime_gmt')
        self.PVLiveData = self.PVLiveData.fillna(0)
        self.PVLiveData.index = self.PVLiveData.index.rename('Settlement Date')
        self.PVLiveData.to_csv("BTM.csv")
        return self.PVLiveData

    def Add(self, Name, Tech):
        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset_Copy = Asset.copy()
                Asset_Copy['Technology'] = Name
                self.Mix['Technologies'].append(Asset_Copy)
                return self

    def Modify(self,Tech,**kwags):
        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset.update(kwags)
        return self

    def PVGISFetch(self,EnhancmentDir,Latitude,Longitude):
        Startyear = self.StartDate.year
        EndYear = self.EndDate.year
        PVGISAPICall = "https://re.jrc.ec.europa.eu/api/seriescalc?lat=" + str(Latitude) + "&lon=" + str(Longitude) + "&startyear=" + str(Startyear) + "&endyear=" + str(EndYear) + "&outputformat=csv&optimalinclination=1&optimalangles=1"
        PVGISAnswer = requests.get(PVGISAPICall)
        PVGISData = pd.read_csv(io.StringIO(PVGISAnswer.text), skipfooter=9, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], engine='python', usecols=['time', 'G(i)'])
        PVGISData['time'] = pd.to_datetime(PVGISData['time'], format='%Y%m%d:%H%M')
        PVGISData['time'] = [t.replace(minute=0) for t in PVGISData['time']]

        GHalfHours = PVGISData['G(i)'].to_numpy()
        GHalfHours = np.insert(GHalfHours, -1, 0)
        GHalfHours = (GHalfHours[1:] + GHalfHours[:-1]) / 2


        THalfHours = PVGISData['time'] + timedelta(minutes=30)
        THalfHours = THalfHours.iloc[:]

        HalfHours = pd.DataFrame(THalfHours)
        HalfHours['G(i)'] = GHalfHours


        PVGISData = pd.concat([PVGISData, HalfHours])
        PVGISData = PVGISData.sort_values(by=['time'])
        PVGISData['time'] = [t.replace(year=self.StartDate.year) for t in PVGISData['time']]
        #PVGISData['time'] = [t + timedelta(minutes=60) for t in PVGISData['time']]
        utc = tz.gettz('UTC')
        timezone = tz.gettz('Europe/London')
        PVGISData['time'] = [t.replace(tzinfo=utc) for t in PVGISData['time']]
        PVGISData['time'] = [t.astimezone(timezone) for t in PVGISData['time']]
        #PVGISData['time'] = [t + timedelta(minutes=30) for t in PVGISData['time']]
        PVGISData = PVGISData.set_index(['time'])
        PVGISData.index = PVGISData.index.rename('Settlement Date')

        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))
        PVGISData = PVGISData.loc[PVGISData.index.isin(CommonIndex)]
        self.PVGISData = copy.deepcopy(PVGISData)
        Enhancment = pd.read_csv(EnhancmentDir)
        f = interp1d(Enhancment['Irradiance'].to_numpy(), Enhancment['Enhanced'].to_numpy(),kind='slinear', fill_value="extrapolate")
        self.DynamScale = f(PVGISData['G(i)'])
        self.DynamScalepd = PVGISData
        self.DynamScalepd['G(i)'] = self.DynamScale
        #self.DynamScale = np.roll(self.DynamScale, 2)
        return self

    def DynamScaleFile(self,Dir):
        DynamScaler = pd.read_csv(Dir, parse_dates=['T'], index_col=['T'])

        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))
        D = DynamScaler.index.to_numpy()
        DynamScaler = np.in1d(CommonIndex,D)
        #DynamScaler = DynamScaler[DynamScaler.index.isin(CommonIndex)]

        DynamScaler = DynamScaler['Enhancment'].to_numpy()[1:-1]
        self.DynamScale = DynamScaler
        return self

    def DynamicScalingFromFile(self,Tech, Dir, BaseScale):
        DynamScaler = pd.read_csv(Dir, parse_dates=['T'], index_col=['T'])
        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        CommonIndex = list(set.intersection(*map(set, IndexValues)))#

        DynamScaler = DynamScaler[DynamScaler.index.isin(CommonIndex)]
        DynamScaler = DynamScaler['Enhancment'].to_numpy()[1:-1]
        Scale = DynamScaler * BaseScale

        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset['Scaler'] = Scale[:]

        return self

    def DynamicScaleingPVGIS(self, Tech, DynamScale, BaseScale):

        Scale = DynamScale * BaseScale

        for Asset in self.Mix['Technologies']:
            if Asset['Technology'] == Tech:
                Asset['Scaler'] = Scale[:]

        return self

    def MatchDates(self):

        IndexValues = [Asset['Generation'].index for Asset in self.Mix['Technologies']]
        #x = [print(type(t)) for t in IndexValues[0]]
        CommonIndex = list(set.intersection(*map(set,IndexValues)))
        #print(CommonIndex)

        Lengths = np.zeros(len(self.Mix['Technologies']))
        for idx,Asset in enumerate(self.Mix['Technologies']):
            #x = [print(type(t)) for t in Asset['Generation'].index]
            Asset['Generation'] = Asset['Generation'].loc[Asset['Generation'].index.isin(CommonIndex)]
            Asset['Generation'] = Asset['Generation'][~Asset['Generation'].index.duplicated(keep='first')]
            self.Dates = Asset['Generation'].index
            Lengths[idx] = len(Asset['Generation'].index)
        return self

    def Demand(self):
        self.Demand = pd.DataFrame(index = self.Mix['Technologies'][0]['Generation'].index.copy())
        self.Demand = 0
        for Asset in self.Mix['Technologies']:
            self.Demand = self.Demand + Asset['Generation'][:]
        return self

    def CarbonEmissions(self):
        self.CarbonEmissions = pd.DataFrame(index = self.Mix['Technologies'][0]['Generation'].index.copy())
        self.CarbonEmissions = 0
        #self.CarbonEmissions['Generation'] = self.CarbonEmissions['Generation'].rename('CO2E')

        for Asset in self.Mix['Technologies']:
            self.CarbonEmissions = self.CarbonEmissions + (Asset['Generation'][:] * Asset['CarbonIntensity'])
        return self

    def Save(self,dir,Filename):
        with open(str(dir)+'\\'+str(Filename)+'.NGM','wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def Load(dir):
        with open(dir,'rb') as handle:
            return pickle.load(handle)

class Dispatch:

    def __init__(self,NG):
        self.NG = copy.deepcopy(NG)
        self.Original = copy.deepcopy(NG)
        self.Distributed = copy.deepcopy(NG)
        self.Demand = self.Distributed.Demand
        self.Generation = self.Distributed.Demand
        self.Generation = 0
        self.CarbonEmissions = self.Distributed.Demand
        self.CarbonEmissions = 0
        self.Order()
        self.Distribute(self.DC1)
        self.Distribute(self.DC2)
        self.Distribute(self.DC3)

        #self.Storage()
        self.Distribute(self.DC4)
        self.Undersuply()
        self.Misc()

    def Order(self):

        self.DC1 = np.zeros(0)
        self.DC2 = np.zeros(0)
        self.DC3 = np.zeros(0)
        self.DC4 = np.zeros(0)

        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['DispatchClass'] == 4:
                self.DC4 = np.append(self.DC4, Asset)
            elif Asset['DispatchClass'] == 3:
                self.DC3 = np.append(self.DC3, Asset)
            elif Asset['DispatchClass'] == 2:
                self.DC2 = np.append(self.DC2, Asset)
            elif Asset['DispatchClass'] == 1:
                self.DC1 = np.append(self.DC1, Asset)
        return

    def Distribute(self,DC):
        for Asset in DC:
            DemandRemaining = 0
            MaxGen = Asset['Generation'] * Asset['Scaler']
            DemandRemaining = self.Demand - self.Generation
            Gen = np.minimum(MaxGen, DemandRemaining)
            #print(Asset['Technology'])
            #if np.sum(MaxGen) > np.sum(DemandRemaining):
            #    print("OverGeneration!")
            #    return
            self.Generation = self.Generation + Gen
            self.CarbonEmissions = self.CarbonEmissions + (Gen * Asset['CarbonIntensity'])
            for DissributedAsset in self.Distributed.Mix['Technologies']:
                if Asset['Technology'] == DissributedAsset['Technology']:
                    DissributedAsset['Generation'] = Gen
                    DissributedAsset['CarbonEmissions'] = Gen * Asset['CarbonIntensity']
        return

    def Undersuply(self):
        if np.any((self.Demand-self.Generation)):
            for Asset in self.DC4:
                MaxGen = Asset['Capacity']
                DemandRemaning = self.Demand - self.Generation
                Gen = np.minimum(MaxGen, DemandRemaning)
                self.Generation = self.Generation + Gen
                self.CarbonEmissions = self.CarbonEmissions + (Gen * Asset['CarbonIntensity'])
                for DissributedAsset in self.Distributed.Mix['Technologies']:
                    if Asset['Technology'] == DissributedAsset['Technology']:
                        DissributedAsset['Generation'] = DissributedAsset['Generation'] + Gen
                        DissributedAsset['CarbonEmissions'] = DissributedAsset['CarbonEmissions'] + (Gen * Asset['CarbonIntensity'])
        return

    def Storage(self):

        StorageRTE = 0.92
        StoragePower = 500
        DemandRemaining = self.Demand - self.Generation

        for Asset in self.Distributed.Mix['Technologies']:
            if Asset['Technology'] == "Hydro Pumped Storage":
                StorageCapacity = Asset['Capacity']
                StorageState = np.minimum(Asset['Generation'],DemandRemaining)

        Pre = 0
        Post = 0

        for Asset in self.DC2:
            for AssetPre in self.Original.Mix['Technologies']:
                if Asset['Technology'] == AssetPre['Technology']:
                    Pre = Pre + np.ravel(AssetPre['Generation'].to_numpy(na_value=0))

            for AssetPost in self.Distributed.Mix['Technologies']:
                if Asset['Technology'] == AssetPost['Technology']:
                    Post = Post + np.ravel(AssetPost['Generation'].to_numpy(na_value=0))

        #self.DC2Curtailed = Post[:-1] - Pre

        #if np.sum(self.DC2Curtailed) == 0:
        #    return self
        #else:
        #    self.StorageDischarge = np.minimum(DemandRemaining.to_numpy(na_value=0), (StorageState * StorageRTE).to_numpy(na_value=0))
            #PosCharge = (self.DC2Curtailed * StorageRTE) - (self.StorageDischarge / StorageRTE)
            StorageState = np.minimum(StorageState + PosCharge, StorageCapacity)
            #self.Generation = self.Generation + StorageDischarge
            #print("A")

        return

    def Misc(self):
        self.Oversuply = self.Generation - self.Demand
        self.Error = np.where(self.Oversuply != 0, True, False)

def SweepSolarGen(NG, Start, Stop, Steps):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    GenSum = np.ndarray(shape = (len(NG.Mix['Technologies']),len(Existing)))

    for Asset in NG.Mix['Technologies']:
        Asset['Generation Sum'] = np.zeros(len(Existing))

    for idx in range(len(Existing)):
        NG = Scaling(NG, Existing[idx], Existing[idx], NewTech[idx], NewTech[idx])
        DNG = Dispatch(NG)
        #print(DNG.DC1)

        SolarGen = 0
        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            GenSum[jdx][idx] = np.sum(Asset['Generation']/1000000/2)
            if Asset['Technology'] == 'Solar' or Asset['Technology'] == 'SolarNT': #or Asset['Technology'] == 'SolarNT' or Asset['Technology'] == 'SolarBTMNT':
                SolarGen = SolarGen + np.sum(Asset['Generation']/1000000/2)
        #print("Gas/Coal Generation (TWh) " + str(idx) + " :" + str(SolarGen))

    #Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    #labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.rcParams["figure.figsize"] = (4, 6)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    plt.stackplot(NewTech, GenSum)
    plt.xticks(np.linspace(0,1,5),np.linspace(0,100,5))
    plt.xlim(left=0, right=1)
    plt.ylabel('Energy Generated (TWh)')
    plt.xlabel('Proportion of DSSCs in Grid (%)')
    plt.tight_layout()
    #plt.legend(labels)
    return

def SweepSolarCarbon(NG, Start, Stop, Steps):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)


    GenSum = np.ndarray(shape=(len(NG.Mix['Technologies']), len(Existing)))
    GenSum.fill(0)

    for Asset in NG.Mix['Technologies']:
        Asset['Generation Sum'] = np.zeros(len(Existing))

    for idx in range(len(Existing)):
        NG = Scaling(NG, Existing[idx], Existing[idx], NewTech[idx], NewTech[idx])
        DNG = Dispatch(NG)

        C = 0
        for jdx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
            GenSum[jdx][idx] = np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
            # Asset['Technology'] == 'Fossil Hard coal':
            C = C + np.sum(Asset['CarbonEmissions'] / 2 * (1*10**-9))
        #print("Emissions (Mt) " + str(idx) + " :" + str(C))

        if idx == 0:
            orgco = GenSum[:,0].copy()
        GenSum[:,idx] = (orgco - GenSum[:,idx])
        #GenSum[:][idx] = GenSum[:][idx] - orgco[:]
    GenSum = np.where(GenSum < 0, 0, GenSum)
    # Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    #plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    plt.rcParams["figure.figsize"] = (4, 6)
    plt.rcParams["figure.dpi"] = 300
    plt.stackplot(NewTech * 100, GenSum[:-5,:])
    plt.xticks( np.linspace(0, 100, 5))
    plt.xlim(left=0, right=100)
   #plt.ylim(bottom=0,top=25)
    plt.ylabel('Carbon Equivalent Emissions Savings (Mt)')
    plt.xlabel('Proportion of DSSCs in Grid (%)')
    plt.tight_layout()
    #plt.legend(labels)
    return

def CarbonEmissions(NG, Start, Stop, Steps, Enhancment):
    Existing = np.linspace(Stop, Start, Steps)
    NewTech = np.linspace(Start, Stop, Steps)

    NG = NG.MatchDates()
    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale

    GenSum = np.ndarray(shape=(len(Existing)))

    for Asset in NG.Mix['Technologies']:
        Asset['Generation Sum'] = np.zeros(len(Existing))

    for idx in range(len(Existing)):
        NG = Grid.Load('Data/2016.NGM')
        NG = NG.Modify('Solar', Scaler=Existing[idx])
        NG = NG.Modify('SolarBTM', Scaler=Existing[idx])
        NG = NG.DynamicScaleingPVGIS('SolarNT', DynamScale, NewTech[idx])
        NG = NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, NewTech[idx])
        DNG = Dispatch(NG)
        GenSum[idx] = np.sum(DNG.CarbonEmissions) / 2 * (1*10**-9)

    # Stacks = [Asset['Generation Sum'] for Asset in DNG.Distributed.Mix['Technologies']]
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.plot(NewTech*100, GenSum)
    #plt.legend(labels)
    #plt.show()
    return

def MaxGenOfDay(NG,Tech,Enhancment):

    NG = NG.MatchDates()
    NG = Grid.Load('Data/2016.NGM')
    NG = NG.Modify('Solar',Scaler=0.5)
    NG = NG.Modify('SolarBTM',Scaler=0.5)
    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale
    NG = NG.DynamicScaleingPVGIS('SolarNT', DynamScale, 0.5)
    NG = NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, 0.5)
    NG = NG.MatchDates()
    DNG = Dispatch(NG)
    for Asset in DNG.Distributed.Mix['Technologies']:
        if Asset['Technology'] == Tech:
            #MaxGenTime = list()
            MaxGenTime = [Asset['Generation'][Asset['Generation'].index.dayofyear == i].idxmax().hour for i in range(1,365)]
            MaxGenTimeMins = [Asset['Generation'][Asset['Generation'].index.dayofyear == i].idxmax().minute for i in range(1, 365)]
            MaxGenTimeMins = [Min/60 for Min in MaxGenTimeMins]
            MaxGenTime = [a+b for a, b in zip(MaxGenTime,MaxGenTimeMins)]
            MaxGen = [Asset['Generation'][Asset['Generation'].index.dayofyear == i].max() for i in range(1, 365)]

            MaxSunTime = [NG.PVGISData['G(i)'][NG.PVGISData['G(i)'].index.dayofyear == i].idxmax().hour for i in range(1, 365)]
            MaxSunTimeMins = [NG.PVGISData['G(i)'][NG.PVGISData['G(i)'].index.dayofyear == i].idxmax().minute for i in range(1, 365)]
            MaxSunTimeMins = [Min / 60 for Min in MaxSunTimeMins]
            MaxSunTime = [a + b for a, b in zip(MaxSunTime, MaxSunTimeMins)]
            MaxSun = [NG.PVGISData['G(i)'][NG.PVGISData['G(i)'].index.dayofyear == i].max() for i in range(1, 365)]

    #plt.scatter(range(1,365),MaxGenTime)
    plt.scatter(range(1,365),MaxSunTime)



    return

def DayIrradiance(NG,Enhancment, Month,Day):
    NG = Grid.Load('Data/2016.NGM')
    NG = NG.PVGISFetch(Enhancment, 53.13359, -1.746826)
    DynamScale = NG.DynamScale
    Month = NG.PVGISData.loc[NG.PVGISData.index.month == Month]
    Day = Month.loc[Month.index.day == Day]
    plt.plot(Day.index,Day['G(i)'])
    return Day

def AverageDayTechnologies(DNG,*args):

    for Technology in args:
        for Asset in DNG.NG.Mix['Technologies']:
            if Technology == Asset['Technology']:
                Means = Asset['Generation'].groupby(Asset['Generation'].index.hour).mean()
                plt.plot(Means)
    return

def AverageDayTechnologiesMonth(NG, Month, **kwargs):
    NG = Grid.Load(NG)
    NG.MatchDates()
    NG.Demand()
    NG.CarbonEmissions()
    NG.PVGISFetch(kwargs['Device'], kwargs['lat'], kwargs['lon'])
    NG.Modify('Solar', Scaler=kwargs['Solar'])
    NG.Modify('SolarBTM', Scaler=kwargs['SolarBTM'])
    DynamScale = NG.DynamScale
    NG.DynamicScaleingPVGIS('SolarNT', DynamScale, kwargs['SolarNT'])
    NG.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, kwargs['SolarBTMNT'])
    DNG = Dispatch(NG)
    Means = np.ndarray(shape=(len(DNG.Distributed.Mix['Technologies']),48))
    Means.fill(0)
    #for idx,Technology in enumerate(args):
    for idx, Asset in enumerate(DNG.Distributed.Mix['Technologies']):
        if Asset['Technology'] == 'Nuclear':
            N = Asset['Generation'].loc[Asset['Generation'].index.month == Month]
            N = np.sum(N)

        M = Asset['Generation'].loc[Asset['Generation'].index.month == Month]
        Means[idx] = M.groupby([M.index.hour,M.index.minute]).mean().to_numpy()
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["figure.dpi"] = 300
    plt.stackplot(range(48), Means)
    plt.xlim(left=0,right=47)
    plt.xticks(range(48)[::8],np.arange(0,48,8)*timedelta(minutes=30))
    labels = [Asset['Technology'] for Asset in DNG.Distributed.Mix['Technologies']]
    plt.xlabel('Time of Day')
    plt.ylabel('Generation (MW)')
    plt.tight_layout()
    #plt.legend(labels)
    return

def SameCO2Savings(StartingPoint, Target, Itter, *args):

    Target_CO2 = (np.sum(Target.CarbonEmissions) / 2 * (1 * 10 ** -9))

    a = 0
    b = 100
    if f(a, StartingPoint,Target_CO2) * f(b,StartingPoint,Target_CO2) >= 0:
        print("fail")
        return None

    a_n = a
    b_n = b
    Progress = np.zeros(0)
    Progress = np.append(Progress, (a_n + b_n) / 2)
    for n in range(1, Itter+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n, StartingPoint, Target_CO2)
        if f(a_n, StartingPoint, Target_CO2)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n, StartingPoint, Target_CO2)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Exact Solution")
            print(m_n)
            return m_n
        else:
            print("method fails")
            return None
    return (a_n + b_n)/2

def f(x, Start, Target_CO2):
    Start = Scaling(Start, x, x, 0, 0)
    DNG = Dispatch(Start)
    return (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9)) - Target_CO2

def CapacitySpread(NG, Device, lat, lon, Xmin,Xmax,Ymin,Ymax,Step):
    X = np.arange(Xmin,Xmax,Step)
    X = np.around(X,3)
    Y = np.arange(Ymin,Ymax,Step)
    Y = np.around(Y,3)
    S = np.zeros((len(X),len(Y)))
    C = copy.deepcopy(S)
    T = copy.deepcopy(S)
    NationalGrid = Setup(NG,Device, lat, lon)
    StartBase = copy.deepcopy(NationalGrid)
    for idx, Sx in enumerate(X):
        for jdx, Sy in enumerate(Y):
            NationalGrid = Scaling(NationalGrid, Sx, Sx, Sy, Sy)
            DNG = Dispatch(NationalGrid)
            S[idx][jdx] = SameCO2Savings(StartBase, DNG, 20)
            print(idx, jdx)
            #NationalGrid = Scaling(NationalGrid, S[idx][jdx], S[idx][jdx], 0, 0)
            #DNG = Dispatch(NationalGrid)
            #Current_CO2 = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9))
            #C[idx][jdx] = Current_CO2


    np.savetxt("resutlsS.csv", S, delimiter=",")
    #np.savetxt("resutlsC.csv", C, delimiter=",")
    #np.savetxt("resutlsT.csv", T, delimiter=",")
    plt.pcolor(T[0][0]-T)
    plt.xlabel('DSSC Capacity Scaler')
    plt.ylabel('Silicon Capacity Scaler')
    plt.colorbar(label='Carbon Equivalent Emissions Savings (Mt)')
    plt.xticks(range(len(Y))[::4],Y[::4])
    plt.yticks(range(len(X))[::4],X[::4])
    plt.show()

def RerunToGetMoreResults(Xmin,Xmax,Ymin,Ymax,Step):
    X = np.arange(Xmin, Xmax, Step)
    X = np.around(X, 3)
    Y = np.arange(Ymin, Ymax, Step)
    Y = np.around(Y, 3)
    F = np.genfromtxt('Misc Data/resutlsSNewCastle2.csv', delimiter=',')
    HCS = np.zeros((len(X),len(Y)))
    HCSS = np.zeros((len(X), len(Y)))
    NationalGrid = Grid.Load('Data/2016Raw.NGM')
    NationalGrid = NationalGrid.Demand()
    NationalGrid = NationalGrid.CarbonEmissions()
    NationalGrid = NationalGrid.MatchDates()
    NationalGrid = NationalGrid.PVGISFetch('Data/Devices/NewCastle.csv', 53.13359, -1.746826)
    DynamScale = NationalGrid.DynamScale
    StartBase = copy.deepcopy(NationalGrid)
    for idx, Sx in enumerate(X):
        for jdx, Sy in enumerate(Y):
            NationalGrid = NationalGrid.Modify('Solar', Scaler=Sx)
            NationalGrid = NationalGrid.Modify('SolarBTM', Scaler=Sx)
            NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarNT', DynamScale, Sy)
            NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, Sy)
            DNG = Dispatch(NationalGrid)
            HCS[idx][jdx] = np.sum(DNG.Generation)
            NationalGrid = NationalGrid.Modify('Solar', Scaler=F[idx][jdx])
            NationalGrid = NationalGrid.Modify('SolarBTM', Scaler=F[idx][jdx])
            NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarNT', DynamScale, 0)
            NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarBTMNT', DynamScale,0)
            DNG = Dispatch(NationalGrid)
            HCSS[idx][jdx] = np.sum(DNG.Generation)

    np.savetxt("resultsHCSGen.csv", HCS, delimiter=",")
    np.savetxt("resultsHCSSGen.csv", HCSS, delimiter=",")
    plt.pcolor(HCS/HCSS)
    plt.colorbar()
    plt.xticks(range(len(Y)), Y)
    plt.yticks(range(len(X)), X)
    plt.show()
    return

def Setup(NG,Device,lat,lon):
    NG = Grid.Load(NG)
    NG.MatchDates()
    NG.Demand()
    NG.CarbonEmissions()
    NG.PVGISFetch(Device, lat, lon)
    return NG

def SetupFromFile(NG,Device,Location):
    NG = Grid.Load(NG)
    NG.MatchDates()
    NG.Demand()
    NG.CarbonEmissions()
    NG.DynamScaleFile(Location)
    return NG

def Scaling(NG,Solar,SolarBTM,SolarNT,SolarBTMNT):
    NG.Modify('Solar', Scaler=Solar)
    NG.Modify('SolarBTM', Scaler=SolarBTM)
    NG.DynamicScaleingPVGIS('SolarNT', NG.DynamScale, SolarNT)
    NG.DynamicScaleingPVGIS('SolarBTMNT', NG.DynamScale, SolarBTMNT)
    return NG

def ScalingDynamFromFile(NG,Solar,SolarBTM,SolarNT,SolarBTMNT,EnhancDir):
    NG.Modify('Solar', Scaler=Solar)
    NG.Modify('SolarBTM', Scaler=SolarBTM)
    NG.DynamicScalingFromFile('SolarNT',EnhancDir,SolarNT)
    NG.DynamicScalingFromFile('SolarBTMNT',EnhancDir,SolarBTMNT)
    return NG

def ScaleAndRunGD(NG,Solar,SolarBTM,SolarNT=0,SolarBTMNT=0):
    NG.Modify('Solar', Scaler=Solar)
    NG.Modify('SolarBTM', Scaler=SolarBTM)
    NG.DynamicScaleingPVGIS('SolarNT', NG.DynamScale, SolarNT)
    NG.DynamicScaleingPVGIS('SolarBTMNT', NG.DynamScale, SolarBTMNT)
    DNG = Dispatch(NG)
    C = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9))
    return C

def AverageDayNamedTechnologies(NG,*args,**kwargs):
    NG = Setup(NG,kwargs['Device'],kwargs['lat'],kwargs['lon'])
    NG = Scaling(NG,kwargs['Solar'],kwargs['SolarBTM'],kwargs['SolarNT'],kwargs['SolarBTMNT'])
    DNG = Dispatch(copy.deepcopy(NG))
    for Asset in DNG.Distributed.Mix['Technologies']:
        if Asset['Technology'] in args:
            M = Asset['Generation'].loc[Asset['Generation'].index.month == kwargs['Month']]
            #M = M.loc[M.index.day == kwargs['Day']]
            M = M.groupby([M.index.hour,M.index.minute]).mean().to_numpy()
            if Asset['Technology'] == 'SolarBTM':
                M = Asset['Generation'].loc[Asset['Generation'].index.month == kwargs['Month']]
                M = M.loc[M.index.day == kwargs['Day']].to_numpy()
                #M = M.groupby([M.index.hour, M.index.minute]).mean().to_numpy()
            elif Asset['Technology'] == 'SolarBTMNT':
                N = Asset['Generation'].loc[Asset['Generation'].index.month == kwargs['Month']]
                N = N.loc[N.index.day == kwargs['Day']].to_numpy()
                #N = N.groupby([N.index.hour, N.index.minute]).mean().to_numpy()

            plt.plot(range(len(M)),M,label=Asset['Technology'])
    X = DNG.NG.PVGISData.loc[DNG.NG.PVGISData.index.month == kwargs['Month']]
    X = X.loc[X.index.day == kwargs['Day']]
    X = X.groupby([X.index.hour,X.index.minute]).mean().to_numpy()
    #plt.legend()
    #plt.twinx()
    #plt.plot(range(len(X)),X,label='PVGIS',color='purple')
    #plt.plot(DNG.NG.PVGISData)
    plt.legend()
    return

def AreaConstrained(NG, Device, lat, lon, AreaScalerMin, AreaScalerMax):
    NG = Setup(NG, Device, lat, lon)
    NG = Scaling(NG, 1, 1, 0, 0)

    InitCapacity = 0
    Tilt = 35
    Width = 1.968
    RowWidth = 7 + np.cos(np.radians(Tilt))*Width

    SiEfficiency = 0.20
    EmergingEfficiency = 0.0989

    for Asset in NG.Mix['Technologies']:
        if Asset['Technology'] == 'Solar':
            InitCapacity = InitCapacity + Asset['Generation'].max()
        elif Asset['Technology'] == 'SolarBTM':
            InitCapacity = InitCapacity + Asset['Generation'].max()

    NumberOfPanels = (InitCapacity * 1000)/SiEfficiency
    InitialArea = ((((1.92 * np.cos(np.radians(Tilt))) * 2 + RowWidth)*0.99)/2)*NumberOfPanels

    Areas = np.arange(AreaScalerMin,AreaScalerMax,0.05)
    Areas2 = np.arange(0, AreaScalerMax,0.05)
    Current_CO2 = np.zeros((len(Areas), len(Areas2)))
    #EmA = np.zeros((len(EmergingFractions), len(Areas)))
    #SiA = np.zeros((len(EmergingFractions), len(Areas)))

    for jdx, AreasScaler2 in enumerate(Areas):
        for idx, AreaScaler in enumerate(Areas2):

            #if AreaScaler == 0:
            #    TotalArea = InitialArea
            #    EmergingArea = 0
            #else:
            #    TotalArea = InitialArea * AreaScaler
            #    EmergingArea = (TotalArea - InitialArea) * EmergingFraction
            #if idx == 0:
            #TotalArea = (InitialArea * AreaScaler) + (InitialArea * AreasScaler2-1)
            SiArea = InitialArea * AreaScaler
            EmergingArea = InitialArea * AreasScaler2

            #EmA[jdx][idx] = EmergingArea
            #SiArea = ((TotalArea - InitialArea ))# * (1-EmergingFraction)) + InitialArea
            #SiA[jdx][idx] = SiArea

            EmergingNumberOfPanels = EmergingArea / ((((1.92 * np.cos(np.radians(Tilt))) * 2 + RowWidth)*0.99)/2)
            EmergingCapacity = (EmergingNumberOfPanels * EmergingEfficiency)/1000
            EmergingScaler = EmergingCapacity / (InitCapacity)

            SiNumberOfPanels = SiArea / ((((1.92 * np.cos(np.radians(Tilt))) * 2 + RowWidth) * 0.99) / 2)
            SiCapacity = (SiNumberOfPanels * SiEfficiency) / 1000
            SiScaler = (SiCapacity / (InitCapacity))

            NG = Scaling(NG, SiScaler, SiScaler, EmergingScaler, EmergingScaler)
            DNG = Dispatch(NG)

            Current_CO2[jdx][idx] = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9))

    #Current_CO2 = Current_CO2[:][::-1]
    X, Y = np.meshgrid(Areas2, Areas)
    Z = Current_CO2[0][0] - Current_CO2
    plt.pcolor(X, Y, Z)
    plt.colorbar(label='Carbon Equivalent Emissions Savings (Mt)')
    levels = [1.5, 2, 2.5, 3, 3.5]


    #plt.xticks(np.arange(0,len(Areas2),4), np.around(Areas2[::4],1))
    #plt.yticks(np.arange(0,len(Areas),4), np.around(Areas[::-4],1))
    plt.contour(X, Y, (X + Y), levels=levels, colors='w').clabel(fontsize=9, inline=True)

    plt.ylabel("Silicon Area Scaler")
    plt.xlabel("Emerging PV Area Scaler")
    #plt.clim(0)
    #plt.colorbar(label='Carbon Equivalent Emissions Savings (Mt)')
    return

def Diffrent_Introductions(NG, DSSC, lat, lon,Si,labeltxt):
    NG = Setup(NG, DSSC, lat, lon)
    NG = Scaling(NG, 1, 1, 0, 0)
    DNG = Dispatch(NG)
    ZeroethCO2e = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9))
    DSSCAdded = np.linspace(1,10,100)

    CO2e = np.zeros(len(DSSCAdded))
    CO2eSi = np.zeros(len(DSSCAdded))
    CO2eSi2 = np.zeros(len(DSSCAdded))
    for idx, DSSCs in enumerate(DSSCAdded):
        NG = Scaling(NG, DSSCs, DSSCs, 0, 0)
        DNG = Dispatch(NG)
        CO2eSi[idx] = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9))
        NG = Scaling(NG,DSSCs,DSSCs,0.1,0.1)
        DNG = Dispatch(NG)
        CO2e[idx] = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9))
    plt.plot(DSSCAdded,(CO2eSi-CO2e),label=labeltxt)
    plt.ylabel("Additional Carbon Equivalent Emissions Savings (Mt)")
    plt.xlabel("Si Capacity Scaler")
    return

def Silicon(NG, DSSC, lat, lon,labeltxt,offset):
    NG = Setup(NG, DSSC, lat, lon)
    NG = Scaling(NG, 1, 1, 0, 0)
    SiSweep = np.linspace(1, 10, 100)
    CO2e = np.zeros(len(SiSweep))
    DNG = Dispatch(NG)
    Z = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9))
    for idx,Si in enumerate(SiSweep):
        NG = Scaling(NG,Si,Si,0.1,0.1)
        DNG = Dispatch(NG)
        CO2e[idx] = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9))
    plt.plot(SiSweep,Z-CO2e,label=labeltxt)
    return

def SolarGenFromSeveralDevices(NGdir, lat, lon, *Devices):
    New = np.linspace(0,1,100)
    Existing = np.linspace(1,0,100)
    for Device in Devices:
        print(Device)
        NG = Setup(NGdir, Device, lat, lon)
        Results = np.zeros(0)
        for N,E in zip(New, Existing):
            NG = Scaling(NG, E, E, N, N)
            DNG = Dispatch(NG)
            i = 0
            y = 0
            for Asset in DNG.Distributed.Mix['Technologies']:
                if Asset['Technology'] == 'Solar':
                    y = y + np.sum(Asset['Generation']/1000000/2)
                    i = i + 1
                if Asset['Technology'] == 'SolarBTM':
                    y = y + np.sum(Asset['Generation']/1000000/2)
                    i = i + 1
                if Asset['Technology'] == 'SolarNT':
                    y = y + np.sum(Asset['Generation']/1000000/2)
                    i = i + 1
                if Asset['Technology'] == 'SolarBTMNT':
                    y = y + np.sum(Asset['Generation']/1000000/2)
                    i = i + 1
                if i == 4:
                    Results = np.append(Results,y)
        plt.plot(New*100,Results)
        plt.ylabel("Solar Generation (TWh)")
        plt.xlabel("Proportion of DSSC in Grid (%)")

    return

def CarbonFromSeveralDevices(NGdir, lat, lon, *Devices):
    New = np.linspace(0, 1, 100)
    Existing = np.linspace(1, 0, 100)
    for Device in Devices:
        print(Device)
        NG = Setup(NGdir, Device, lat, lon)
        NG = Scaling(NG, 1, 1, 0, 0)
        DNG = Dispatch(NG)
        Z = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9))
        Results = np.zeros(0)
        for N, E in zip(New, Existing):
            NG = Scaling(NG, E, E, N, N)
            DNG = Dispatch(NG)
            Results = np.append(Results, (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9)))
        plt.plot(New * 100, Z-Results)
        plt.ylabel("Carbon Equivalent Emissions Savings (Mt)")
        plt.xlabel("Proportion of DSSC in Grid (%)")

    return

def SolarGenEndSeveralDevices(NGdir, lat, lon, *Devices):
    New = np.linspace(0,1,5)
    Existing = np.linspace(1,0,5)
    D = [-50,0,50,100,150,200]
    for N,E in zip(New, Existing):
        Results = np.zeros(0)
        for Device in Devices:
            NG = Setup(NGdir, Device, lat, lon)
            NG = Scaling(NG,E,E,N,N)
            DNG = Dispatch(NG)
            Results = np.append(Results,(np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9)))
        plt.plot(D,Results[1]-Results, label=str(N)+":"+str(E))
    plt.xlabel("Curve Fit Shift (Wm$^{-2}$)")
    plt.ylabel("Carbon Equivalent Emissions Savings (Mt)")
    plt.legend()
    return

def Gradient_Descent(initNG, Target, x, y,  MaxItter=400, Tolerance=1e-2, LearningRate=0.2):

    initDNG = Dispatch(initNG)

    X = np.zeros(MaxItter)
    Y = np.zeros(MaxItter)
    G = np.zeros(MaxItter)

    X_0 = 0
    Y_0 = (np.sum(initDNG.CarbonEmissions)/2 * (1 * 10 ** -9)) - Target

    X[0] = X_0 + LearningRate
    NG = Scaling(initNG, x, y, 0, 0)
    DNG = Dispatch(NG)
    Y[0] = (np.sum(DNG.CarbonEmissions)/2 * (1 * 10 ** -9)) - Target
    for i in range(1,MaxItter):
        G[i] = (Y[i]-Y[i-1])/(X[i]-X[i-1])
        X[i] = X[i-1] + LearningRate * G[i]
        NG = Scaling(initNG, X[i], X[i], 0, 0)
        DNG = Dispatch(NG)
        Y[i] = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9)) - Target
        if abs(G[i]) < Tolerance:

            return X[i]
    return X[-1]

def Silicon_Equivilent(DSSC_CO2, NG):
    Target_All = np.genfromtxt(DSSC_CO2, delimiter=',')
    Results = np.zeros(np.shape(Target_All))
    X = np.arange(0, 2, 0.05)
    Y = np.arange(0, 2, 0.05)
    for idx,x in enumerate(Target_All):
        for jdx,Target in enumerate(x):
            print(idx, jdx)
            Results[idx][jdx] = Gradient_Descent(NG,Target,X[idx],Y[jdx])
    np.savetxt('GradDec.csv',Results,delimiter=",")
    return

def Expand_Generation(NG, length_in_years):
    for Asset in NG.Mix['Technologies']:
        New = copy.deepcopy(Asset['Generation'])
        Temp = copy.deepcopy(Asset['Generation'])
        Y = Temp.index[0].year
        for i in range(0, length_in_years-1, 1):
            Current_year = Y + i
            if calendar.isleap(Current_year):
                Temp.index = Temp.index[:] + timedelta(days=366)
            else:
                Temp.index = Temp.index[:] + timedelta(days=365)
            New = New.append(Temp)
        Asset['Generation'] = New
    return NG

def Expand_Sacler(NG, length_in_years):
    for Asset in NG.Mix['Technologies']:
        if type(Asset['Scaler']) != int:
            Asset['Scaler'] = np.tile(Asset['Scaler'], length_in_years)
    return NG

def Add_to_SolarNT(NG,Gen):
    for Asset in NG.Mix['Technologies']:
        if Asset['Technology'] == 'SolarNT':
            Asset['Generation'] = Gen.squeeze()
            Asset['Scaler'] = 1

    return NG

#NG = Setup('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
#NG = SetupFromFile('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', '500LocationEnhancment.csv')
#NG = Scaling(NG, 0, 0, 0, 0)
#DNG = Dispatch(NG)

#Silicon_Equivilent('resutlsCBangor2.csv',NG)

#SolarGenEndSeveralDevices('Data/2016RawT.NGM', 53.13359, -1.746826, 'Data/Devices/DSSC-50.csv','Data/Devices/DSSC-0.csv','Data/Devices/DSSC+50.csv','Data/Devices/DSSC+100.csv','Data/Devices/DSSC+150.csv','Data/Devices/DSSC+200.csv')

#SolarGenFromSeveralDevices('Data/2016RawT.NGM', 53.13359, -1.746826, 'Data/Devices/DSSC.csv', 'Data/Devices/DSSC-50.csv','Data/Devices/DSSC-0.csv','Data/Devices/DSSC+50.csv','Data/Devices/DSSC+100.csv','Data/Devices/DSSC+150.csv','Data/Devices/DSSC+200.csv')

#CarbonFromSeveralDevices('Data/2016RawT.NGM', 53.13359, -1.746826, 'Data/Devices/DSSC.csv', 'Data/Devices/DSSC-50.csv', 'Data/Devices/DSSC-0.csv', 'Data/Devices/DSSC+50.csv', 'Data/Devices/DSSC+100.csv', 'Data/Devices/DSSC+150.csv', 'Data/Devices/DSSC+200.csv')
#Diffrent_Introductions('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826, 1.5, 'Bangor : Si=1.5')
#Diffrent_Introductions('Data/2016RawT.NGM', 'Data/Devices/NewCastle.csv', 53.13359, -1.746826, 1.5, 'Newcastle : Si=1.5')

#Diffrent_Introductions('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826, 5, 'Bangor : Si=5')
#Diffrent_Introductions('Data/2016RawT.NGM', 'Data/Devices/NewCastle.csv', 53.13359, -1.746826, 5, 'Newcastle : Si=5')

#Silicon('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826,'Si',1.5)
#Silicon('Data/2016RawT.NGM', 'Data/Devices/Newcastle.csv', 53.13359, -1.746826,'Si',1.5)
#Silicon('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826,'Si',5)
#Silicon('Data/2016RawT.NGM', 'Data/Devices/Newcastle.csv', 53.13359, -1.746826,'Si',5)

#Diffrent_Introductions('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826,2,'Bangor : Si=2')
#Diffrent_Introductions('Data/2016RawT.NGM', 'Data/Devices/NewCastle.csv', 53.13359, -1.746826,2,'Newcastle : Si=2')
#plt.legend()

#NG = Setup('Data/2016CZ.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
#DNG = Dispatch(NG)
#NG = Scaling(NG, 1, 1, 0, 0)
#SweepSolarGen(NG,0,1,100)
#SweepSolarCarbon(NG,0,1,100)

#AreaConstrained('Data/2016RawT.NGM', 'Data/Devices/NewCastle.csv', 53.13359, -1.746826, 1, 2.05)
#AverageDayTechnologiesMonth('Data/2016RawT.NGM', 7, Device='Data/Devices/Device3.csv', lat=53.13359, lon=-1.746826, Solar=0.5, SolarBTM=0.5, SolarNT=0.5, SolarBTMNT=0.5)
#AverageDayNamedTechnologies('Data/2016RawT.NGM', 'SolarBTM', 'Nuclear', Device='Data/Devices/DSSC.csv', lat=53.13359, lon=-1.746826, Solar=0.5, SolarBTM=1000000000000, SolarNT=0.5, SolarBTMNT=0.5,Month=12,Day=15)

#NationalGrid = Grid("Mix2016GB.json")
#NationalGrid = NationalGrid.Add('SolarNT','Solar')
#NationalGrid = NationalGrid.Add('SolarBTMNT','SolarBTM')
#NationalGrid = NationalGrid.Save('Data','2016GB')

#AverageDayNamedTechnologies('Data/2016RawT.NGM', 'SolarNT','Solar', 'SolarBTM', 'SolarBTMNT', Device='Data/Devices/test.csv', lat=53.13359, lon=-1.746826, Solar=0.5, SolarBTM=0.5, SolarNT=0.5, SolarBTMNT=0.5,Month=12)

#NationalGrid = Grid.Load('Data/2016CZ.NGM')

#NationalGrid = NationalGrid.MatchDates()
#NationalGrid = NationalGrid.Demand()
#NationalGrid = NationalGrid.CarbonEmissions()

#NationalGrid = NationalGrid.PVGISFetch('Data/Devices/DSSC.csv', 53.13359, -1.746826)
#DynamScale = NationalGrid.DynamScale
#NationalGrid = NationalGrid.Modify('Solar', Scaler=1)
#NationalGrid = NationalGrid.Modify('SolarBTM', Scaler=1)
#NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarNT', DynamScale, 0.5)
#NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, 0.5)

#DNG = Dispatch(NationalGrid)
#plt.plot(np.cumsum(DNG.Generation))
#plt.show()

#Current_CO2 = 0
#for Asset in DNG.Distributed.Mix['Technologies']:
#    Current_CO2 = Current_CO2 + (np.sum(Asset['CarbonEmissions'] / 2 * (1 * 10 ** -9)))
#print(Current_CO2)

#NationalGrid = NationalGrid.Modify('Solar', Scaler=0.9)
#NationalGrid = NationalGrid.Modify('SolarBTM', Scaler=0.9)
#DNG = Dispatch(NationalGrid)
#Current_CO2 = 0
#for Asset in DNG.Distributed.Mix['Technologies']:
#    Current_CO2 = Current_CO2 + (np.sum(Asset['CarbonEmissions'] / 2 * (1 * 10 ** -9)))
#print(Current_CO2)

#SameCO2Savings('Data/2016Raw.NGM',DNG,20)

#CapacitySpread('Data/2016RawT.NGM', 'Data/Devices/NewCastle.csv', 53.13359, -1.746826,1,2.05,0,2.05,0.05)
#RerunToGetMoreResults(0,2,0,2,0.05)
#plt.figure(figsize=(8,12))
#plt.rcParams.update({'font.size': 24})
#NationalGrid = NationalGrid.Save('Data','2016')
#SweepSolarCarbon(NationalGrid, 0, 1, 100,'Data/Devices/NewCastle.csv')
#SweepSolarGen(NationalGrid, 0, 2, 1, 'Data/Devices/DSSC.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/PolySi.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/NewCastle.csv')
#CarbonEmissions(NationalGrid, 0, 1, 100,'Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarBTMNT','Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarBTMNT','Data/Devices/DSSC.csv')
#MaxGenOfDay(NationalGrid,'SolarNT','Data/Devices/NewCastle.csv')
#DayIrradiance(NationalGrid, 'Data/Devices/DSSC.csv',6,30).to_csv('20160630Irradiance.csv')


#NationalGrid = NationalGrid.PVGISFetch('Data/Devices/DSSC.csv', 53.13359, -1.746826)
#DynamScale = NationalGrid.DynamScale
#plt.plot(NationalGrid.PVGISData,c='tab:orange')
#plt.twinx()
#plt.plot(NationalGrid.PVGISData.index,DynamScale)

#NationalGrid = NationalGrid.Modify('Solar', Scaler=0.5)
#NationalGrid = NationalGrid.Modify('SolarBTM', Scaler=0.5)
#NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarNT', DynamScale, 0.5)
#NationalGrid = NationalGrid.DynamicScaleingPVGIS('SolarBTMNT', DynamScale, 0.5)
#DNG = Dispatch(NationalGrid)

#AverageDayTechnologiesMonth(DNG,6)
#plt.xlabel("Hour of the day")
#plt.ylabel("Mean Gneeration (MW)")
#plt.twinx()
#A = NationalGrid.PVGISData.loc[NationalGrid.PVGISData.index.month == 1]
#A = A.replace(0, np.NaN)#
#A = A.groupby(A.index.hour).mean()
#plt.plot(A.index, A)
#A = NationalGrid.PVGISData.loc[NationalGrid.PVGISData.index.month == 7]
#A = A.replace(0, np.NaN)
#A = A.groupby(A.index.hour).mean()
#plt.plot(A.index, A)
#plt.ylabel("Mean Irradiance (Wm$^{-2}$)")
#plt.show()