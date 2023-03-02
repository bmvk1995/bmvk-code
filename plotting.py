# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:11:05 2023

@author: bmvk1
"""
import pandas as pd
import matplotlib.pyplot as plt

def line_plot(url):    
    data = pd.read_csv(url) 
    yearly_births = data.groupby('year')['births'].sum()
    
    # line graph of yearly births
    plt.plot(yearly_births.index, yearly_births.values)
    plt.title('Line Graph of yearly births')
    plt.xlabel('year')
    plt.ylabel('births')
    plt.show()
    
def pie_plot(url):
    data = pd.read_csv(url)
    yearly_births = data.groupby('year')['births'].sum()
    
    #pie chart of yearly births
    plt.pie(yearly_births.values, labels=yearly_births.index, autopct='%1.1f%%')
    plt.title('Pie Chart of yearly births')
    plt.show()
    
def bar_plot(url):
    data = pd.read_csv(url)
    yearly_births = data.groupby('year')['births'].sum()
    
    # Plot bar graph of yearly births
    plt.bar(yearly_births.index, yearly_births.values)
    plt.title('Bar Graph of yearly births')
    plt.xlabel('year')
    plt.ylabel('births')
    plt.show()

# URL LINK FROM GITHUB DATA SOURCE
url ="https://raw.githubusercontent.com/fivethirtyeight/data/master/births/US_births_1994-2003_CDC_NCHS.csv"
line_plot(url)
pie_plot(url)
bar_plot(url)