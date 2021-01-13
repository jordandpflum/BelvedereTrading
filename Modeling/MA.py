import pandas as pd
from pandas import DataFrame
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt


# Moving Average on ZM-ZS close price

df = pd.read_csv("/Users/fionazh/Desktop/ZS_sub.csv")
#print(df)


#for i in range(10,200):
  # df['MA{}'.format(i)] = df.rolling(window=i).mean()
# df[[f for f in list(df) if "MA" in f]].mean(axis=1)



df_close = pd.DataFrame(df.CLOSE)
df_close['MA_10'] = df_close.CLOSE.rolling(window=10).mean()
df_close['MA_50'] = df_close.CLOSE.rolling(window=50).mean()
df_close['MA_200'] = df_close.CLOSE.rolling(window=200).mean()
print(df_close)

plt.figure(figsize=(15, 10))
#plt.plot(df_close, df.Timestamp)
plt.grid(True)
plt.plot( df_close['CLOSE'], label='ZS')
plt.plot( df_close['MA_10'], label='MA 10 day')
plt.plot( df_close['MA_50'], label='MA 50 day')
plt.plot( df_close['MA_200'], label='MA 200 day')
plt.legend(loc=2)
plt.show()



df2 = pd.read_csv("/Users/fionazh/Desktop/ZM_sub.csv")

df2_close = pd.DataFrame(df2.CLOSE)
df2_close['MA_10'] = df2_close.CLOSE.rolling(window=10).mean()
df2_close['MA_50'] = df2_close.CLOSE.rolling(window=50).mean()
df2_close['MA_200'] = df2_close.CLOSE.rolling(window=200).mean()
print(df2_close)


plt.figure(figsize=(15, 10))
plt.grid(True)
plt.plot( df2_close['CLOSE'], label='ZM')
plt.plot( df2_close['MA_10'], label='MA 10 day')
plt.plot( df2_close['MA_50'], label='MA 50 day')
plt.plot( df2_close['MA_200'], label='MA 200 day')
plt.legend(loc=2)
plt.show()


