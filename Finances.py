import numpy as np
import time
from numba import jit

class Finance:

    def __init__(self,Job,E,T,P,I):
        self.Dates = P.Dates
        self.PVSize = P.PVSize
        self.PanelLifetime = P.Lifetime
        self.InverterLifetime = I.Lifetime
        self.InitialCost = E.PanelCost2 + E.InstallationCostExcPanels
        self.InstallationCostExcPanels = E.InstallCostPerPanel
        self.InverterCostAsPercentofCiepPrice = E.InverterCostAsPercentofCiepPrice
        self.NewPrice = E.NewPrice
        self.PanelCost = Job['Cost']
        self.DCR = Job['IRR'] * 0.01
        self.InverterCostInflation = Job['InvCosInf'] * 0.01
        self.OperationCostInflation = Job['OprCosInf'] * 0.01
        self.InterestDevisor = T.InterestDevisor
        self.RentCost = Job['RenCos']
        self.NewArea = E.NewArea
        self.PVGen = P.PVGen

    
    def PanelPrice(self):
        self.PanelPrice = np.zeros(len(self.Dates))
        self.PanelPrice[:] = self.PanelCost
        self.PanelPrice = self.PanelCost + (self.PanelPrice - self.PanelCost) * (1 - self.DCR)/self.InterestDevisor
        return
    
    def Replacments(self):
        PLR = np.roll(self.PanelLifetime,-1)
        ILR = np.roll(self.InverterLifetime,-1)

        self.PanelReplacments = np.where(self.PanelLifetime < PLR)[0][:-1]
        self.PanelReplacmentCostPV = np.zeros(len(self.Dates))
        self.PanelReplacmentCostPV[self.PanelReplacments] = 1000 * self.PVSize * self.PanelPrice[self.PanelReplacments]

        I = np.linspace(0,len(self.Dates),len(self.Dates))
        self.PanelReplacmentCostOther = np.zeros(len(self.Dates))
        self.PanelReplacmentCostOther[self.PanelReplacments] = (self.NewPrice * 0.1) * np.power((1 + self.InverterCostInflation),(((I[self.PanelReplacments]/self.InterestDevisor)/365) -1))
        
        self.PaneReplacementCost = self.PanelReplacmentCostPV + self.PanelReplacmentCostOther

        self.InverterReplacments = np.where(self.InverterLifetime < ILR)[0][:-1]
        self.InverterReplacmentCost = np.zeros(len(self.Dates))
        self.InverterReplacmentCost[self.InverterReplacments] = (self.InstallationCostExcPanels * self.InverterCostAsPercentofCiepPrice) * np.power(1+self.InverterCostInflation,(((I[self.InverterReplacments]/self.InterestDevisor)/365) -1))

        return
    
    def RecurringCosts(self):
        self.OAM = np.zeros(len(self.Dates))
        self.LandRental = np.zeros(len(self.Dates))
        self.OAM[0] = (1000 * self.PVSize * 0.01 )/ self.InterestDevisor
        self.LandRental[0] = self.RentCost * self.NewArea / self.InterestDevisor

        for i in range(1,len(self.Dates)):
            self.OAM[i] = self.OAM[i-1] * (1 + (self.OperationCostInflation / self.InterestDevisor))
            self.LandRental[i] = self.LandRental[i-1] * (1 + (self.OperationCostInflation / self.InterestDevisor))
        
        return
    
    def Costs(self):
        self.PanelPrice()
        self.Replacments()
        self.RecurringCosts()

        self.TotalCosts = self.PaneReplacementCost + self.InverterReplacmentCost + self.OAM + self.LandRental
        return
    
    def LCOECalculate(self):
        I = np.linspace(0,len(self.Dates),len(self.Dates))
        TC = self.TotalCosts[:]
        PV = self.PVGen[:]
        II = I[:]/self.InterestDevisor
        self.Top = (self.NewPrice  + np.abs(xnpv(self.DCR,TC[:],II[:])) )
        self.Bottom = np.abs(xnpv(self.DCR,PV[:],II[:]))
        self.LCOE =  (self.NewPrice  + np.abs(xnpv(self.DCR,TC[:],II[:])) )/ np.abs(xnpv(self.DCR,PV[:],II[:]))
        return
@jit  
def xnpv(DCR,Values,Date):
    V = np.sum(Values[:] / (1.0 + DCR)**(Date[:]))
    return V
