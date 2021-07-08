import pandas as pd
import matplotlib.pyplot as plt

Results = pd.read_csv('Results.csv')
Results = Results.groupby(Results['Job.Tech'])
Newcastle = Results.get_group("NewCastle")
NoEnhancment = Results.get_group('NoEnhancment')
Newcastle = Newcastle.sort_values(by=['Job.Latitude','Job.Longitude'])
NoEnhancment = NoEnhancment.sort_values(by=['Job.Latitude','Job.Longitude'])

NCLCOE = Newcastle['Finance.LCOE'].to_numpy()
NELCOE = NoEnhancment['Finance.LCOE'].to_numpy()

Reduction = (1 - (NCLCOE / NELCOE)) * 100
#print(Reduction)

plt.scatter(Newcastle['Panel.Irradiance'],Reduction)
plt.xlabel('Average IIrradiance (W/m^2)')
plt.ylabel('LCOE Reduction (%)')
plt.show()