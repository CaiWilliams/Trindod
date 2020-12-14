import pandas as pd
import numpy as np


class Out:
    def __init__(self, job, epc, time, panel, inverter, finance):
        self.Job = job
        self.EPC = epc
        self.Time = time
        self.Panel = panel
        self.Inverter = inverter
        self.Finance = finance

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
            'LCOE']
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
        df.to_excel(str(self.Job['ProjectName']) + ".xlsx")
        return

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
                ResultsOutput.append(Result[np.nonzero(Result)].mean())
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

    def PerformanceRatio(self):
        Months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        Hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        MonthlyPR = np.zeros(len(Months))
        HourlyPR = np.zeros((len(Months), len(Hours)))
        A = np.array(list(map(M, self.Panel.Dates)))
        B = np.array(list(map(T, self.Panel.Dates)))
        j = 0
        for month in Months:
            i = 0
            IM = np.where(A == month)[0]
            for hour in Hours:
                IC = np.where((A == month) & (B == hour))[0]
                HourlyPR[j, i] = np.average(self.Panel.PVGen[IC])
                i = i + 1
            j = j + 1
        np.savetxt(self.Job['PrjLoc'] + self.Job['Tech'] + ".csv", HourlyPR, delimiter=",")
        return


def M(a):
    return a.month


def T(a):
    return a.hour
