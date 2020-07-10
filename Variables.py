import pickle
import os

def GenVars():
    Variables = {
    'PanTyp': 1,
    'PanCosInf': -1,
    'PanFlrPri': 0.245,
    'PanPriDef': 10,
    'InvLif': 10,
    'PrjLif': 20,
    'Irr': 'High Risk',
    'ModSta': '01/01/19',
    'PrjTyp': 'Groundmount PV Array',
    'PrjLoc': 'Fiji',
    'OprCosInf': 2.1,
    'InvCosInf': 2.1,
    'OprCos': 0.5,
    'RenInf': 2.1, 
    }
    VariablesDict = {
    'PanTyp': 'Pannel Type',
    'PanCosInf': 'Pannel Cost Inflation',
    'PanFlrPri': 'Pannel Floor Price',
    'PanPriDef': 'Pannel Price Deflation',
    'InvLif': 'Inverter Life',
    'PrjLif': 'Project Life',
    'Irr': 'IRR Selection',
    'ModSta': 'Model Start',
    'PrjTyp': 'Project Type',
    'PrjLoc': 'Project Location',
    'OprCosInf': 'Operating Cost Inflation',
    'InvCosInf': 'Inverter Cost Inflation',
    'OprCos': 'Operating Cost',
    'RenInf': 'Rental inflation', 
    }
    VariablesUnits = {
    'PanTyp':'',
    'PanCosInf': '%',
    'PanFlrPri': 'USD/Wp',
    'PanPriDef': '%',
    'InvLif': 'Years',
    'PrjLif': 'Years',
    'Irr':'',
    'ModSta':'',
    'PrjTyp':'',
    'PrjLoc':'',
    'OprCosInf': '%',
    'InvCosInf': '%',
    'OprCos': 'USD/m2',
    'RenInf': '%', 
    }
    VariablesIntVals = {
    1:'PanTyp',
    2:'PanCosInf',
    3:'PanFlrPri',
    4:'PanPriDef',
    5:'InvLif',
    6:'PrjLif',
    7:'Irr',
    8:'ModSta',
    9:'PrjTyp',
    10:'PrjLoc',
    11:'OprCosInf',
    12:'InvCosInf',
    13:'OprCos',
    14:'RenInf', 
    }
    pickle.dump(Variables, open("Data/Variables.p","ab"))
    pickle.dump(VariablesDict, open("Data/VariablesDict.p","ab"))
    pickle.dump(VariablesUnits, open("Data/VariablesUnits.p","ab"))
    pickle.dump(VariablesIntVals, open("Data/VariablesIntVals.p","ab"))
    return

def Load(FileName):
    X = pickle.load(open(FileName,"rb"))
    return X

def Save(X,FileName):
    pickle.dump(X,open(FileName,"wb"))
    return

def UserInput():
    Variables = Load("Data/Variables.p")
    VariablesDict = Load("Data/VariablesDict.p")
    VariablesUnits = Load("Data/VariablesUnits.p")
    os.system('cls')
    PrintVars()
    change()
    return

def change():
    Variables = Load("Data/Variables.p")
    VariablesDict = Load("Data/VariablesDict.p")
    VariablesUnits = Load("Data/VariablesUnits.p")
    VariablesIntVals = Load("Data/VariablesIntVals.p")
    print("")
    print("Would you like to change any value? Enter number (1-14) to do so. Otherwise just hit enter")
    x = input()
    if x == '':
        x = 0
    if 15 > int(x)  and int(x) > 0:
        print(VariablesDict[VariablesIntVals[int(x)]] + ": ")
        NewVal = input()
        SrcType = type(Variables[VariablesIntVals[int(x)]])
        print(SrcType)
        if SrcType == str:
            NewVal = str(NewVal)
        elif SrcType == int:
            NewVal = int(NewVal)
        else:
            NewVal = float(NewVal)
        print(type(NewVal))
        Variables[VariablesIntVals[int(x)]] = NewVal
        Save(Variables,"Data/Variables.p")
        PrintVars()
    
    print("Are you done making changes: Y/N")
    Conf = input().lower()
    if Conf == 'n':
        change()
    else:
        Save(Variables,"Data/Variables.p")
        return
        
def PrintVars():
    Variables = Load("Data/Variables.p")
    VariablesDict = Load("Data/VariablesDict.p")
    VariablesUnits = Load("Data/VariablesUnits.p")
    os.system('cls')
    i = 1
    for Key in VariablesDict:
        print(str(i) +"  "+ VariablesDict[Key] +":  "+ str(Variables[Key]) +" "+ VariablesUnits[Key] )
        i = i + 1
    return

GenVars()
UserInput()