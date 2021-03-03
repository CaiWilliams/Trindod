import numpy as np


# Class for the finanical calculations
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
    def LCOECalculate(self):
        i = np.linspace(0, len(self.Dates), len(self.Dates))
        tc = self.TotalCosts[:]
        pv = self.PVGen[:]
        ii = i[:] / self.InterestDivisor
        self.Top = (self.NewPrice + np.abs(xnpv(self.DCR, tc[:], ii[:])))
        self.Bottom = xnpv(self.DCR, pv[:], ii[:])
        self.LCOE = (self.NewPrice + np.abs(xnpv(self.DCR, tc[:], ii[:]))) / xnpv(self.DCR, pv[:], ii[:])
        return


# Calculates the net present value
def xnpv(dcr, values, date):
    V = np.sum(values[:] / (1.0 + dcr) ** (date[:]))
    return V
