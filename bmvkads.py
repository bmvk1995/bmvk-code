# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:35:28 2023

@author: bmvk1
"""
"Importing the Modules"
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import wbgapi as wb
import pandas as pd
from sklearn.cluster import KMeans

# To Suppress warnings we use import wamrnings ::
import warnings
warnings.filterwarnings("ignore")

# Preporocessing of data and returing original data ::
def data(x, y, z):
    data = wb.data.DataFrame(x, mrv=y)
    data1 = pd.DataFrame(data.sum())
    data1.columns = z
    data2 = data1.rename_axis("year")
    return data, data2

# Converting Year column to Integers
def string(x):
    # print(x)
    z = []
    for i in x.index:
        c = i.split("YR")
        z.append(int(c[1]))
    # print(z)
    x["year"] = z
    return x

indicator = ["EN.ATM.METH.KT.CE", "EN.ATM.CO2E.KT"]  

# Indicators using for this to produce graphs ::
data_ghg_O, data_ghg_R = data(
    indicator[0], 30, ["Methane Emission"])  # Methane Emission data

# Calling the string function to convert year column to integer ::
new_data_GHG = string(data_ghg_R)
data_CO2_O, data_CO2_R = data(indicator[1], 30, ["CO2"])  # CO2 data

# Calling the string function to convert year column to integer ::
new_data_CO2 = string(data_CO2_R)

# Computes exponential function with scale and growth free parameters ::
def exp_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1990))
    return f

popr, pcov = curve_fit(
    exp_growth, data_ghg_R["year"], data_ghg_R["Methane Emission"])
  
# Plotting graph between data and curve_fit ::
data_ghg_R["pop_exp"] = exp_growth(data_ghg_R["year"], *popr)
plt.figure()
plt.plot(
    data_ghg_R["year"],
    data_ghg_R["Methane Emission"],
    label="Methane Emission")
plt.plot(data_ghg_R["year"], data_ghg_R["pop_exp"], label="fit")
plt.legend()
plt.title("Curve Fit and data line of Methane Emission")
plt.xlabel("year")
plt.ylabel("Methane Emission")
plt.show()
print()


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    This routine can be used in assignment programs.
    
    """
    import itertools as iter

# Initiate arrays for lower and upper limits ::

    lower = func(x, *param)
    upper = lower
    uplow = []  # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper

# To plotting graph between confidence ranges and fit data ::
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(data_ghg_R["year"], exp_growth, popr, sigma)
plt.figure()
plt.title("exp_growth function")
plt.plot(
    data_ghg_R["year"],
    data_ghg_R["Methane Emission"],
    label="Methane")
plt.plot(data_ghg_R["year"], data_ghg_R["pop_exp"], label="fit")
plt.fill_between(data_ghg_R["year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("Methane Emission")
plt.show()
popr2, pcov2 = curve_fit(
    exp_growth, data_CO2_R["year"], data_CO2_R['CO2'])  

# Curve_fit for CO2 and Plotting the graph between CO2 data and curve_fit ::
data_CO2_R["pop_exp"] = exp_growth(data_CO2_R["year"], *popr2)
plt.figure()
plt.plot(data_CO2_R["year"], data_CO2_R["CO2"], label="CO2")
plt.plot(data_CO2_R["year"], data_CO2_R["pop_exp"], label="fit")
plt.legend()
plt.title("Curve fit and data line of CO2")
plt.xlabel("year")
plt.ylabel("CO2")
plt.show()
print()

# Plotting graph between confidence ranges and fit data ::
sigma = np.sqrt(np.diag(pcov2))
low, up = err_ranges(data_CO2_R["year"], exp_growth, popr2, sigma)
plt.figure()
plt.title("exp_growth function")
plt.plot(data_CO2_R["year"], data_CO2_R["CO2"], label="CO2")
plt.plot(data_CO2_R["year"], data_CO2_R["pop_exp"], label="fit")
plt.fill_between(data_CO2_R["year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("CO2")
plt.show()

# Prepossing data for clustering ::
data_ghg = pd.DataFrame(data_ghg_O.iloc[:, -1])
data_CO2 = pd.DataFrame(data_CO2_O.iloc[:, -1])
data_ghg.columns = ["Methane Emission"]
data_CO2.columns = ["CO2"]
data_ghg["CO2"] = data_CO2["CO2"]
data_ghg_C = data_ghg.rename_axis("countries")
final_data = data_ghg_C.dropna()

# Plotting the scatter plot for kmeans clustering ::
X = final_data[['CO2', 'Methane Emission']].copy()
kmeanModel = KMeans(n_clusters=3)  # Choosing 3 clusters
identified = kmeanModel.fit_predict(final_data[['CO2', 'Methane Emission']])
cluster_centers = kmeanModel.cluster_centers_  # Getting  the cluster center points

# To Get the unique labels ::
u_labels = np.unique(identified)  # Getting unique cluster labels
clusters_with_data = final_data[['CO2', 'Methane Emission']].copy()
clusters_with_data['Clusters'] = identified  # Add cluster column
fig = plt.figure(figsize=(10, 8))

# To Plot the data points ::
plt.scatter(clusters_with_data['CO2'], clusters_with_data['Methane Emission'],
            c=clusters_with_data['Clusters'], cmap='viridis')

# To Plot the center points ::
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, alpha=0.8);
plt.title("Scatter plot after clusters")
plt.xlabel('CO2')
plt.ylabel('Methane Emission')
plt.show()