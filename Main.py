from EPCM import Epcm
from CashFlow import Cashflow
from Manual import Main

def Run():
    Name = Main()
    Epcm(Name)
    Cashflow(Name)
    return

Run()