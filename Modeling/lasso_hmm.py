import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import matplotlib.pyplot as plt
from datetime import datetime
from Data_Master.read_data import *
import random
from Modeling.regimehmm import *

random.seed(0)

def create_regime_dataframes(no_of_states,futures,features,time_interval):
    '''
    Function which grabs the date ranges from the output of our classifcation models and generates training and testing dataframes
    for different regimes.
    :param no_of_states: Specifies the total number of regimes
    :param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
    :param features: Specifies features to be included
    :time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
    '''
    futures_and_features = ['Timestamp']
    x_df = get_x_df(futures)
    x_df = engineer_features(x_df, futures, features)
    x_df = x_df.dropna()
    x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
    features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
    date_dict = create_correlation_predictor(no_of_states)
    #date_dict = get_regime_labels_hmm()
    # print(date_dict)

    for future in futures:
        for feature in features1:
            futures_and_features.append(future + '_' + feature)

    train_state_dataframes = ["x_train" + str(i + 1) for i in np.arange(no_of_states)]

    for i in range(no_of_states):
        train_state_dataframes[i] = pd.DataFrame(columns = futures_and_features)


    train_dict = date_dict['actual']
    Labels = train_dict.keys()

    i = 0
    for label in Labels:
        for j in range(len(train_dict[label])):
            mask = (x_df['Timestamp'] >= pd.Timestamp(train_dict[label][j][0]))  & (x_df['Timestamp'] <= pd.Timestamp(train_dict[label][j][1]))
            train_state_dataframes[i] = train_state_dataframes[i].append(x_df.loc[mask])
        i += 1

    test_state_dataframes = ["x_test" + str(i + 1) for i in np.arange(no_of_states)]

    for i in range(no_of_states):
        test_state_dataframes[i] = pd.DataFrame(columns = futures_and_features)

    test_dict = date_dict['predicted']
    Labels = test_dict.keys()

    i = 0
    for label in Labels:
        for j in range(len(test_dict[label])):
            mask = (x_df['Timestamp'] >= pd.Timestamp(test_dict[label][j][0]))  & (x_df['Timestamp'] <= pd.Timestamp(test_dict[label][j][1]))
            test_state_dataframes[i] = test_state_dataframes[i].append(x_df.loc[mask])
        i += 1
    return train_state_dataframes,test_state_dataframes



def all_data_df(no_of_states,futures,regime,features,time_interval,subset = "False"):
    ''''
    If subset = "False", returns a dataframe consisting of the entire training data (all the regime dataframes combined).
    If subset = "True", returns a dataframe consisting of data points from all the regimes but with a size equal to regime dataframe
    corresponding to the regime specified by the 'regime' parameter.
    :param no_of_states: Specifies the total number of regimes
    :param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
    :param regime: Specifies the regime number in order for the size to be matched
    :param features: Specifies features to be included
    :time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
    '''
    futures_and_features = ['Timestamp']
    train_state_dataframes,test_set_dataframes = create_regime_dataframes(no_of_states,futures,features,time_interval)
    rows,columns = train_state_dataframes[regime].shape
    part_rows = math.floor(rows/no_of_states)
    x_df = get_x_df(futures)
    x_df = engineer_features(x_df, futures, features)
    x_df = x_df.dropna()
    x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
    features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
    date_dict = create_correlation_predictor(no_of_states)
    #date_dict = get_regime_labels_hmm()
    # print(date_dict)
    for future in futures:
        for feature in features1:
            futures_and_features.append(future + '_' + feature)
    all_data_df = pd.DataFrame(columns = futures_and_features)
    if subset == "True":
        for i in range(len(train_state_dataframes)):
            df = train_state_dataframes[i]
            rows1,columns1 = df.shape
            rand_start = random.randint(0,rows1 - part_rows)
            temp__df = df.iloc[rand_start:rand_start + part_rows,:]
            all_data_df = all_data_df.append(temp__df)
        return all_data_df
    else:

        train_dict = date_dict['actual']
        Labels = train_dict.keys()

        for label in Labels:
            for j in range(len(train_dict[label])):
                mask = (x_df['Timestamp'] >= pd.Timestamp(train_dict[label][j][0]))  & (x_df['Timestamp'] <= pd.Timestamp(train_dict[label][j][1]))
                all_data_df = all_data_df.append(x_df.loc[mask])
        return all_data_df

def get_x_y(df,lag):
    '''
    Generates an x_dataframe and a y_dataframe from the dataframe given using a rolling window of lagged features.
    Essentially, converts time series dataframe to a supervised learning framework in order to run Lasso Regression.
    :param df: Input dataframe
    :param lag: Specifies the lag for the rolling window.
    '''
    rows,columns = df.shape
    x_df = np.zeros((rows - lag,columns*lag))
    y_df = np.zeros((rows - lag,))
    count = 0
    for i in range(lag,rows):
        if i <= rows - 1:
            y_df[count] = df.iloc[i, 3] # Column 3 is the ZM Close price
            arr1 = df.iloc[i-lag:i,:].values
            x_df[count] = arr1.reshape(columns*lag,)
            count += 1
    x_df = pd.DataFrame(x_df)
    y_df = pd.DataFrame(y_df)
    return x_df,y_df

def Lasso_HMM(futures,features,no_of_states,regime,reg,time_interval):
    '''
    For a specific regime, fits two Lasso Regression models: 1 - Model trained on the entire training data (dataframe returned
    by all_data_df()) and 2 - Model trained on just the dataframe corresponding to that regime. Returns the Mean Absolute Error,
    Mean Squared Error and R2 Score for each of them (R2 score is just for interpretability).
    :param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
    :param features: Specifies features to be included
    :param no_of_states: Specifies the total number of regimes
    param regime: Specifies the regime number
    : reg: Regularization Parameter
    :time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
    '''

    features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
    futures_and_features = []
    for future in futures:
        for feature in features1:
            futures_and_features.append(future + '_' + feature)
    train_state_dataframes,test_state_dataframes = create_regime_dataframes(no_of_states,futures,features,time_interval)
    if train_state_dataframes[regime].empty or test_state_dataframes[regime].empty:
        return "No Data for this Regime"

    min_max_scaler = preprocessing.MinMaxScaler()

    df_model1 = all_data_df(no_of_states,futures,regime,features,time_interval,subset = "True")
    df_model1 = df_model1.drop(['Timestamp'],axis = 1)
    df_model1 = pd.DataFrame(min_max_scaler.fit_transform(df_model1))
    df_model1.columns = futures_and_features

    df_model2 = train_state_dataframes[regime]
    df_model2 = df_model2.drop(['Timestamp'],axis = 1)
    df_model2 = pd.DataFrame(min_max_scaler.fit_transform(df_model2))
    df_model2.columns = futures_and_features

    df_test = test_state_dataframes[regime]
    df_test = df_test.drop(['Timestamp'],axis = 1)
    df_test = pd.DataFrame(min_max_scaler.fit_transform(df_test))
    df_test.columns = futures_and_features

    x_train_model1,y_train_model1 = get_x_y(df_model1,3)
    x_train_model2,y_train_model2 = get_x_y(df_model2,3)
    x_test,y_test = get_x_y(df_test,3)

    model1 = linear_model.Lasso(alpha=reg)
    model2 = linear_model.Lasso(alpha=reg)
    model1.fit(x_train_model1,y_train_model1)
    model2.fit(x_train_model2,y_train_model2)

    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)

    mse1 = mean_squared_error(y_test, y_pred1)
    mse2 = mean_squared_error(y_test, y_pred2)

    mae1 = mean_absolute_error(y_test,y_pred1)
    mae2 = mean_absolute_error(y_test,y_pred2)

    score1 = model1.score(x_test,y_test)
    score2 = model2.score(x_test,y_test)

    return mae1,mae2,mse1,mse2,score1,score2


