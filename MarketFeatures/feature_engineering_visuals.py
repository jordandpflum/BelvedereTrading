import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import talib
from Data_Master.read_data_jordan import *
import numpy as np
from numpy.polynomial.polynomial import polyfit




# create interesting groupings for exploration
categories = {'Soybeans': ['ZL','ZM','ZS'],
              'Livestock': ['HE', 'LE', 'GF'],
              'Energy': ['CL', 'HO', 'NG'],
              'Agriculture': ['ZC', 'ZS', 'ZW'],
              'Metals': ['GC', 'PL', 'SI']}


# Sample RSI (ZL) vs Weekly Corr Diff (ZL-ZM) Plot
'''
# Load Data
futures_names = categories['Soybeans']
merged_df = merge_df_reduce(futures_names, 'weekly')
merged_df.dropna(inplace=True)
merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
merged_df = merged_df.set_index('Timestamp')
merged_df = merged_df.resample(rule='W').mean()

# RSI
x = merged_df['ZL_RSI']
y = merged_df['weekly_corr_diff_ZL-ZM']

b, m = polyfit(x, y, 1)

plt.scatter(x,y)
plt.plot(x, b + m * x, 'r-')
plt.xlabel('ZL - RSI')
plt.ylabel('Weekly Correlation Diff (ZL-ZM)')
plt.title('ZL-ZM correlation vs ZL Relative Strength Index')
plt.show()
'''


# Sample plots of features
'''
# RSI
x = merged_df['ZL_RSI']
y = merged_df['weekly_corr_diff_ZL-ZM']

b, m = polyfit(x, y, 1)

plt.scatter(x,y)
plt.plot(x, b + m * x, 'r-')
plt.xlabel('ZL - RSI')
plt.ylabel('Weekly Correlation Diff (ZL-ZM)')
plt.title('ZL-ZM correlation vs ZL Relative Strength Index')

plt.show()


# OBV
x = merged_df['ZL_OBV']
y = merged_df['weekly_corr_diff_ZL-ZM'].abs()

b, m = polyfit(x, y, 1)

plt.scatter(x,y)
plt.plot(x, b + m * x, 'r-')
plt.xlabel('ZL - OBV')
plt.ylabel('Weekly Correlation Diff (ZL-ZM)')
plt.title('ZL-ZM correlation vs ZL On-Balance Volume')

plt.show()

# CHAIKIN
x = merged_df['ZL_CHAIKIN']
y = merged_df['weekly_corr_diff_ZL-ZM'].abs()

b, m = polyfit(x, y, 1)

plt.scatter(x,y)
plt.plot(x, b + m * x, 'r-')
plt.xlabel('ZL - CHAIKIN')
plt.ylabel('Weekly Correlation Diff (ZL-ZM)')
plt.title('ZL-ZM correlation vs ZL Chaikin Oscilator')

plt.show()
'''