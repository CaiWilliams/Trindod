import pandas as pd
import pickle

class Out:
    def __init__(self,Job,EPC,Time,Panel,Inverter,Finance):
        self.Job = Job
        self.EPC = EPC
        self.Time = Time
        self.Panel = Panel
        self.Inverter = Inverter
        self.Finance = Finance
    
    def Excel(self):
        CFC = ['Date','ProjectYear','PanelLifetime','InverterLifetime','PeakSunHours','CumilativeSunHours','Burn-inAbsolute','LongTermDegredation','LongTermDegredationAbsolute','PanelStateofHealth','PeakCapacity','EffectiveCapacity','MonthlyYeild','PVGeneration','CapitalCost','RefurbishmentCost(Panels-PV)','RefurbishmentCost(Panels-Other)','RefurbishmentCost(Panels)','PanelPriceThisYear','RefurbishmentCost(Inverter)','AnnualO&MCost','LandRental','TotalCost','CostCheck','LCOE']
        df = pd.DataFrame(self.Panel.Dates,columns=['Date'])
        df['Irradiance'] = pd.Series(self.Panel.Irradiance,index=df.index)
        df['Panel Lifetime'] = pd.Series(self.Panel.Lifetime,index=df.index)
        df['Inverter Lifetime'] = pd.Series(self.Inverter.Lifetime,index=df.index)
        df['Peak Sun Hours'] = pd.Series(self.Panel.PSH,index=df.index)
        df['Cumilative Sun Hours'] = pd.Series(self.Panel.CPSH,index=df.index)
        df['Burn-in Abs'] = pd.Series(self.Panel.BurnInAbs,index=df.index)
        df['Long Term Degradation'] = pd.Series(self.Panel.LongTermDeg,index=df.index)
        df['Long Term Degradation Abs'] = pd.Series(self.Panel.LongTermDegAbs,index=df.index)
        df['Panel State of Health'] = pd.Series(self.Panel.StateOfHealth,index=df.index)
        df['Peak Capacity'] = pd.Series(self.Panel.Capacity,index=df.index)
        df['Effective Capacity'] = pd.Series(self.Panel.EffectiveCapacity,index=df.index)
        df['Monthly Yield'] = pd.Series(self.Panel.Yield,index=df.index)
        df['PV Generation'] = pd.Series(self.Panel.PVGen,index=df.index)
        df['Refurbishment Cost (PV)'] = pd.Series(self.Finance.PanelReplacmentCostPV,index=df.index)
        df['Refurbishment Cost (Other)'] = pd.Series(self.Finance.PanelReplacmentCostOther,index=df.index)
        df['Refurbishment Cost (Panels)'] = pd.Series(self.Finance.PaneReplacementCost,index=df.index)
        df['Panel Price This Year'] = pd.Series(self.Finance.PanelPrice,index=df.index)
        df['Refurbishment Cost (Inverter)'] = pd.Series(self.Finance.InverterReplacmentCost,index=df.index)
        df['Annual O&M Cost'] = pd.Series(self.Finance.OAM,index=df.index)
        df['Land Rental'] = pd.Series(self.Finance.LandRental,index=df.index)
        df['Total Cost'] = pd.Series(self.Finance.TotalCosts,index=df.index)
        df['LCOE'] = pd.Series(self.Finance.LCOE,index=df.index)
        df.to_excel(str(self.Job['ProjectName'])+".xlsx")
        return
    
    def Results(self):
        File = pd.read_csv('Results.csv')
        ResultsRequested = File.columns.values
        ResultsOutput = list()
        for Result in ResultsRequested:
            Result = Result.split('.')

            if Result[0] =='Finance':
                Result = getattr(self.Finance,Result[1])
                ResultsOutput.append(Result)
            elif Result[0] == 'Panel':
                Result = getattr(self.Panel,Result[1])
                ResultsOutput.append(Result)
            elif Result[0] == 'Inverter':
                Result = getattr(self.Inverter,Result[1])[-1]
                ResultsOutput.append(Result)
            elif Result[0] =='EPC':
                Result = getattr(self.EPC,Result[1])[-1]
                ResultsOutput.append(Result)
            else:
                Result = self.Job[Result[1]]
                ResultsOutput.append(Result)
        ResultO = pd.DataFrame([ResultsOutput],columns=ResultsRequested)
        File = File.append(ResultO,ignore_index=True)
        File.to_csv('Results.csv',index=False)
        return