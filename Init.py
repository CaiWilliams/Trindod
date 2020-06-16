import numpy as np
import os
import h5py

def CreateProject():
    ProName = ProjectName()
    ProLoc, Lat, Lon = ProjectLoc()
    
def ProjectName():
    os.system('cls')
    print("What is the name of the project?")
    name = input()
    print("Is " + name + "correct? Y/N")
    X = input().lower()
    if X == "y":
        return name
    elif X == "n":
        ProjectName()
    else:
        print("Invalid Entry!")
        ProjectName()
    return

def ProjectLoc():
    os.system('cls')
    print("Location Name:")
    name = input()
    print("Location Latitude:")
    Lat = float(input())
    print("Location Longitude:")
    Lon = float(input())
    os.system('cls')
    print("Location Name: " + name)
    print("Location Latitude: " + Lat)
    print("Location Longitude: "+ Lon)
    print("Location Details Correct? Y/N")
    X = input().lower()
    if X == "y":
        return name,Lat,Lon 
    elif X == "n":
        ProjectLoc()
    else:
        print("Invalid Entry!")
        ProjectLoc()
    return