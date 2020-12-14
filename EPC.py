import math


class EPC:
    def __init__(self, job):
        self.OriginalCost = job['Design'] + job['Construction'] + job['Framing'] + job['DCcabling'] + job['ACcabling'] + job['CivilWork(Panels)'] + job['CivilWork(general)'] + job['PVPanels'] + job['FixedProjectCosts'] + job['Freight(Panels)'] + job['Freight(other)'] + job['Inverters'] + job['Controls']
        self.PriceExcludingPanels = self.OriginalCost - job['PVPanels']
        self.PanelSize = 410
        self.NumberOfPanels = 1000 * (job['PVSize'] / self.PanelSize)
        self.InstallCostPerPanel = self.PriceExcludingPanels / self.NumberOfPanels
        self.InverterCost = job['Inverters']
        self.OldArea = job['SystemArea']
        self.PanelCost = job['Cost']
        self.EqRatingPanels = job['PowerDensity'] * 1.968 * 0.992
        self.RequiredNumberPanels = 1000 * job['PVSize'] / self.EqRatingPanels
        self.InstallationCostExcPanels = self.RequiredNumberPanels * self.InstallCostPerPanel
        self.PanelCost2 = self.PanelCost * 1000 * job['PVSize']
        self.NewPrice = self.InstallationCostExcPanels + self.PanelCost2
        self.InverterCostAsPercentofCiepPrice = self.InverterCost / self.InstallationCostExcPanels
        self.NewArea = ((((1.92 * math.cos(math.radians(job['Tilt']))) * 2 + job['Spacing']) * 0.99) / 2) * self.RequiredNumberPanels
