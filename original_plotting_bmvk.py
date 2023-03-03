# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:11:05 2023

@author: bmvk1
"""
import pandas as pd
import matplotlib.pyplot as plt

def line_plot(url):    
    data = pd.read_csv(url) 
    yearly_births = data.groupby('date_of_month')['births'].max()
    
    # line graph of DATE_OF_MONTH by Births
    plt.plot(yearly_births.index, yearly_births.values)
    plt.title('#~~~Line Graph of Date_of_Month by Births~~~#')
    plt.xlabel('*Date_of_Month*')
    plt.ylabel('*BIRTHS*')
    plt.show()
    
def pie_plot(url):
    data = pd.read_csv(url)
    yearly_births = data.groupby('year')['births'].sum()
    
    #pie chart of yearly births
    plt.pie(yearly_births.values, labels=yearly_births.index, autopct='%1.1f%%')
    plt.title('###~~~~Pie Chart of Yearly Births~~~###')
    plt.show()
    
def bar_plot(url):
    data = pd.read_csv(url)
    yearly_births = data.groupby('day_of_week')['births'].max()
    
    # Plot bar graph of DAY_OF_WEEK by Births
    plt.bar(yearly_births.index, yearly_births.values)
    plt.title('#~~~Bar Graph of day_of_week by births~~~#')
    plt.xlabel('*Day_of_Week*')
    plt.ylabel('*BIRTHS*')
    plt.show()

# URL LINK FROM GITHUB DATA SOURCE
url ="https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_1994-2003_CDC_NCHS.csv" 
line_plot(url)
pie_plot(url)
bar_plot(url)