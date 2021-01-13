import pandas as pd 
import numpy as np 
from HMM_AllFuturePairs import CorrelationPredictor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import talib
import sys
sys.path.insert(0, 'E:\\Belevedere_Spr20-master\\Data_Master')
import feature_engineering
from feature_engineering import engineer_features

def get_x_df(futures_names):
    """
    Function to grab original (fixed) futures data and place it into a dataframe without additional features
    and without target variables (just the "x-data").
    :param: a list of futures contracts names (e.g. "BZ")
    :return: pandas df with a timestamp and 5*28 columns (open,high,low,close,volume) for each contract.
    Should be merged with a y-dataframe
    """
    x_df = pd.DataFrame()

    count = 0
    for future_name in futures_names:
        file = 'E:\\Belevedere_Spr20-master\\Data_Master\\fixed_belvedere_data\\' + future_name + '.csv'
        if(x_df.empty):
            x_df = pd.read_csv(file, sep='\t')
            x_df.columns = ['Timestamp', future_name + '_OPEN', future_name + '_HIGH', future_name + '_LOW', future_name + '_CLOSE', future_name + '_TRADED_VOLUME']
        else:
            more_data = pd.read_csv(file, sep='\t')
            more_data.columns = ['Timestamp', future_name + '_OPEN', future_name + '_HIGH', future_name + '_LOW', future_name + '_CLOSE', future_name + '_TRADED_VOLUME']
            x_df = x_df.merge(more_data.drop_duplicates(subset=['Timestamp']), on='Timestamp')
        count += 1
        # print(future_name)
        # print(x_df.shape)
    return x_df

all_future_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC','LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
features_names = ['ADX', 'ADXR', 'APO', 'AROON_UP', 'AROON_DOWN', 'AROONOSC',
'BOP', 'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX', 'MFI', 'MINUS_DI',
'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI',
'STOCH_k', 'STOCH_d', 'STOCH_Fast_k', 'STOCH_Fast_d', 'STOCHRSI_k', 'STOCHRSI_d',
'TRIX', 'ULTOSC', 'WILLR', 'OBV', 'CHAIKIN']

df = engineer_features(get_x_df(all_future_names), all_future_names, features_names)
print(df.head())

def model2_dataframe(future,week,diff):
    if diff == "True":
        all_future_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC','LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        price_data = get_x_df(all_future_names)

        price_diff1 = price_data.iloc[:,1:price_data.shape[1]]
        price_diff1 = price_diff1.diff()
        price_diff1 = price_diff1.drop(price_diff1.index[0])

        price_diff0 = price_data.iloc[:,0]
        price_diff0 = price_diff0.drop(price_diff0.index[0])

        price_diff = pd.concat([price_diff0,price_diff1],axis = 1)


        price_diff['Timestamp'] = pd.to_datetime(price_diff['Timestamp'])
        price_diff['Week'] = price_diff['Timestamp'].dt.week
        price_diff['Year'] = price_diff['Timestamp'].dt.year

        # print(price_data.head())
        # print(price_diff.head())
        # print(price_diff.shape)

        price_diff_week = price_diff.loc[(price_diff['Week'] == week) & (price_diff['Year'] == 2017)]
        # print(price_diff_week.head())
        model = CorrelationPredictor()
        model.fit()
        correlation_df = model.predict_future_correlations(week)
        correlations = correlation_df.iloc[week - 1,:]
        correlations = model.ES_corr_test.iloc[week - 1,:]
        # print(correlations)

        pair_list = []
        for i in range(len(correlations)):
            if correlations[i] > 0.2:
                pair_list.append(model.ES_corr_train.columns[i])
        # print(pair_list)
        correlated_futures = []
        for value in pair_list:
            if value[12:14] == future:
                correlated_futures.append(value[15:17])

        # print(correlated_futures)

        final_features = [future + "_OPEN",future + "_HIGH",future + "_LOW",future + "_CLOSE"]
        for value in correlated_futures:
            final_features.extend([value + "_OPEN",value + "_HIGH",value + "_LOW",value + "_CLOSE"])
        final_dataframe = pd.DataFrame()
        for value in final_features:
            final_dataframe[value] = price_diff_week[value]
        return final_dataframe
    else:
        all_future_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC','LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        price_data_model2 = get_x_df(all_future_names)
        price_data_model2['Timestamp'] = pd.to_datetime(price_data_model2['Timestamp'])
        price_data_model2['Week'] = price_data_model2['Timestamp'].dt.week
        price_data_model2['Year'] = price_data_model2['Timestamp'].dt.year

        price_data_week = price_data_model2.loc[(price_data_model2['Week'] == week) & (price_data_model2['Year'] == 2017)]
        model = CorrelationPredictor()
        model.fit()
        correlation_df = model.predict_future_correlations(week)
        correlations = correlation_df.iloc[week - 1,:]
        
        pair_list = []
        for i in range(len(correlations)):
            if correlations[i] > 0.2:
                pair_list.append(model.ES_corr_train.columns[i])
        
        correlated_futures = []
        for value in pair_list:
            if value[12:14] == future:
                correlated_futures.append(value[15:17])

        final_features = [future + "_OPEN",future + "_HIGH",future + "_LOW",future + "_CLOSE"]
        for value in correlated_futures:
            final_features.extend([value + "_OPEN",value + "_HIGH",value + "_LOW",value + "_CLOSE"])
        final_dataframe = pd.DataFrame()
        for value in final_features:
            final_dataframe[value] = price_data_week[value]
        return final_dataframe

# df = model2_dataframe("ES",52,diff = "False")
# print(df)

def model1_dataframe(future,week,diff):
    if diff == "True":
        all_future_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC','LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        price_data_model1 = get_x_df(all_future_names)
        price_data_model1 = price_data_model1[['Timestamp',future + "_OPEN",future + "_HIGH",future + "_LOW",future + "_CLOSE"]]
        price_diff_model1 = price_data_model1.iloc[:,1:price_data_model1.shape[1]]
        price_diff_model1 = price_diff_model1.diff()
        price_diff_model1 = price_diff_model1.drop(price_diff_model1.index[0])

        price_diff0_model1 = price_data_model1.iloc[:,0]
        price_diff0_model1 = price_diff0_model1.drop(price_diff0_model1.index[0])

        price_diff_model1 = pd.concat([price_diff0_model1,price_diff_model1],axis = 1)


        price_diff_model1['Timestamp'] = pd.to_datetime(price_diff_model1['Timestamp'])
        price_diff_model1['Week'] = price_diff_model1['Timestamp'].dt.week
        price_diff_model1['Year'] = price_diff_model1['Timestamp'].dt.year

        # print(price_data.head())
        # print(price_diff.head())
        # print(price_diff.shape)

        price_diff_week_model1 = price_diff_model1.loc[(price_diff_model1['Week'] == week) & (price_diff_model1['Year'] == 2017)]
        return price_diff_week_model1
    else:
        all_future_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC','LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        price_data_model1 = get_x_df(all_future_names)
        price_data_model1 = price_data_model1[['Timestamp',future + "_OPEN",future + "_HIGH",future + "_LOW",future + "_CLOSE"]]
        price_data_model1['Timestamp'] = pd.to_datetime(price_data_model1['Timestamp'])
        price_data_model1['Week'] = price_data_model1['Timestamp'].dt.week
        price_data_model1['Year'] = price_data_model1['Timestamp'].dt.year

        price_week_model1 = price_data_model1.loc[(price_data_model1['Week'] == week) & (price_data_model1['Year'] == 2017)]
        price_week_model1 = price_week_model1.iloc[:,1:price_week_model1.shape[1] - 2]
        return price_week_model1

# df = model1_dataframe("ES",52,"False")
# print(df)


def model2_xy(future,week,lag):
    x_df = pd.DataFrame()
    y_df = pd.DataFrame()
    df = model2_dataframe(future,week,diff = "False")
    df1 = df.copy()
    count = 0
    for i in range(lag,df.shape[0],lag):
        if i + count <= df.shape[0]:
            y_df = y_df.append(pd.Series(df.iloc[i + count,3]),ignore_index = True)
            df1 = df1.drop(df.index[i + count])
            count += 1
    for j in range(0,df1.shape[0],lag):
        if j + lag <= df1.shape[0]:
            arr1 = df1.iloc[j:j+lag,:].values
            x_df = x_df.append(pd.Series(arr1.reshape(df1.shape[1]*lag,)),ignore_index = True)
    return x_df,y_df

# x_df,y_df = model2_xy("ES",52,3)
# print(x_df)
# print(y_df)
# print(y_df.iloc[:,0].unique())

def model1_xy(future,week,lag):
    x_df = pd.DataFrame()
    y_df = pd.DataFrame()
    df = model1_dataframe(future,week,diff = "False")
    df1 = df.copy()
    count = 0
    for i in range(lag,df.shape[0],lag):
        if i + count <= df.shape[0]:
            y_df = y_df.append(pd.Series(df.iloc[i + count,3]),ignore_index = True)
            df1 = df1.drop(df.index[i + count])
            count += 1
    for j in range(0,df1.shape[0],lag):
        if j + lag <= df1.shape[0]:
            arr1 = df1.iloc[j:j+lag,:].values
            x_df = x_df.append(pd.Series(arr1.reshape(df1.shape[1]*lag,)),ignore_index = True)
    return x_df,y_df

# x_df,y_df = model1_xy("ES",52,3)
# print(x_df)
# print(y_df)
# print(y_df.iloc[:,0].unique())

# x1,y1 = model1_xy("ES",52,3)
# x2,y2 = model2_xy("ES",52,3)

# X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.33, random_state=42)
# X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.33, random_state=42)
# model1 = linear_model.Lasso(alpha=0)
# model2 = linear_model.Lasso(alpha=0)
# model1.fit(X1_train,y1_train)
# model2.fit(X2_train,y2_train)
# score1 = model1.score(X1_test,y1_test)
# score2 = model2.score(X2_test,y2_test)
# print(score1)
# print(score2)