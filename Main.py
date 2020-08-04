from EPCM import *
from CashFlow import * 
from Init import * 
def Run():
    ProName = CreateProject()
    Epcm(ProName)
    Cashflow(ProName)
    return

Run()