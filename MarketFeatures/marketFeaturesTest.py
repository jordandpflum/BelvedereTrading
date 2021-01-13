import pandas as pd
import matplotlib.pyplot as plt
import talib

# Load Data
BZ = pd.read_csv("DataMaster/belvederedata/BZ.csv")
#BZ_values = BZ[['Timestamp', 'CLOSE']]


## MACD
macd, macdsignal, macdhist = talib.MACD(BZ.CLOSE, fastperiod=12, slowperiod=26, signalperiod=9)
print(type(macd))
print(type(macd >= macdsignal))
frame = {'Timestamp': BZ.Timestamp, 'MACD': macd, 'MACD_Signal': macdsignal}
macd2 = pd.DataFrame(frame)
print(macd2)
plt.plot(BZ.Timestamp[:100], macd[:100], label = 'MACD')
plt.plot(BZ.Timestamp[:100], macdsignal[:100], label = 'MACD Signal')
plt.title('MACD')
plt.ylabel('Feature')
#plt.xticks([1,2,3,4])
plt.legend(loc = 'best')
#plt.show()


## Bollinger Bands
upperband, middleband, lowerband = talib.BBANDS(BZ.CLOSE, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
plt.plot(BZ.Timestamp[:100], upperband[:100], label = 'Upper Band')
plt.plot(BZ.Timestamp[:100], middleband[:100], label = 'SMA')
plt.plot(BZ.Timestamp[:100], lowerband[:100], label = 'Lower Band')
plt.title('Bollinger Bands')
plt.ylabel('Feature')
plt.legend(loc = 'best')
#plt.show()


## RSI
real = talib.RSI(BZ.CLOSE, timeperiod=14)
plt.plot(BZ.Timestamp[:100], real[:100], label = 'RSI')
plt.title('Relative Strength Index')
plt.ylabel('RSI')
plt.legend(loc = 'best')
#plt.show()


## Stochastic Oscillator
slowk, slowd = talib.STOCH(BZ.HIGH, BZ.LOW, BZ.CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
plt.plot(BZ.Timestamp[:100], slowk[:100], label = 'Stochastic Indicator')
plt.plot(BZ.Timestamp[:100], slowd[:100], label = '3-Period SMA')
plt.title('Stochastic Oscillator')
plt.ylabel('Feature')
plt.legend(loc = 'best')
#plt.show()



## On-Balance Volume
real = talib.OBV(BZ.CLOSE, BZ.TRADED_VOLUME)
plt.plot(BZ.Timestamp[:100], real[:100], label = 'OBV')
plt.title('On-Balance Volume')
plt.ylabel('OBV')
plt.legend(loc = 'best')
#plt.show()



## Chaikin A/D Oscillator
real = talib.ADOSC(BZ.HIGH, BZ.LOW, BZ.CLOSE, BZ.TRADED_VOLUME, fastperiod=3, slowperiod=10)
plt.plot(BZ.Timestamp[:100], real[:100], label = 'ADOSC')
plt.title('Chaikin A/D Oscillator')
plt.ylabel('ADOSC')
plt.legend(loc = 'best')
#plt.show()



