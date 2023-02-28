# bmvk-code
python code for showing line plot, bar plot, pie chart using the data from the online sources using numpy and pandas
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:09:54 2023

@author: bmvk1
"""

import pandas as pd
import matplotlib.pyplot as plt
# Read data from an online source
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_1994-2003_CDC_NCHS.csv"
votes = pd.read_csv(url,error_bad_lines=False)

# Create a pie chart of total votes by location
bmvk1 = votes.groupby('year')['births'].sum()
bmvk1.plot(kind='pie', autopct='%1.1f%%', startangle=900)
plt.title('year by births')
plt.show()

# Create a line plot of year by births
bmvk2 = votes.groupby('year')['births'].sum()
bmvk2.plot(kind='line')
plt.title('year by births')
plt.xlabel('births')
plt.ylabel('year')
plt.show()

# Create a bar plot of average total bill by sex
bmvk3 = votes.groupby('day_of_week')['births'].mean()
bmvk3.plot(kind='bar')
plt.title('days_of_week by births')
plt.xlabel('day_of_week')
plt.ylabel('births')
plt.show()
