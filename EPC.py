import math


class EPC:
    def __init__(self,Job):
        self.OriginalCost = Job['Design'] + Job['Construction'] + Job['Framing'] + Job['DCcabling'] + Job['ACcabling'] + Job['CivilWork(Panels)'] + Job['CivilWork(general)'] + Job['PVPanels'] + Job['FixedProjectCosts'] + Job['Freight(Panels)'] + Job['Freight(other)'] + Job['Inverters'] + Job['Controls']
        self.PriceExcludingPanels = self.OriginalCost - Job['PVPanels']
        self.PanelSize = 410
        self.NumberOfPanels = 1000 * (Job['PVSize'] / self.PanelSize)
        self.InstallCostPerPanel = self.PriceExcludingPanels / self.NumberOfPanels
        self.InverterCost = Job['Inverters']
        self.OldArea = Job['SystemArea']
        self.PanelCost = Job['Cost']
        self.EqRatingofPanels = Job['PowerDensity'] * 1.968 * 0.992
        self.RequiredNumberofPanels = 1000 * Job['PVSize']/self.EqRatingofPanels
        self.InstallationCostExcPanels = self.RequiredNumberofPanels * self.InstallCostPerPanel
        self.PanelCost2 = self.PanelCost * 1000 * Job['PVSize']
        self.NewPrice = self.InstallationCostExcPanels + self.PanelCost2
        self.InverterCostAsPercentofCiepPrice = self.InverterCost / self.InstallationCostExcPanels
        self.NewArea = ((((1.92 * math.cos(math.radians(Job['Tilt']))) * 2 + Job['Spacing']) * 0.99)/2) * self.RequiredNumberofPanels
