from AltModels import *
import numpy as numpy
import pandas as pd 
sys.path.insert(0, 'E:\\Belvedere_Spr20\\Data_Master')
from read_data import *
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
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
        file = 'E:/Belvedere_Spr20/Data_Master/fixed_belvedere_data/' + future_name + '.csv'
        if(x_df.empty):
            x_df = pd.read_csv(file, sep='\t')
            x_df.columns = ['Timestamp', future_name + '_OPEN', future_name + '_HIGH', future_name + '_LOW', future_name + '_CLOSE', future_name + '_TRADED_VOLUME']
        else:
            more_data = pd.read_csv(file, sep='\t')
            more_data.columns = ['Timestamp', future_name + '_OPEN', future_name + '_HIGH', future_name + '_LOW', future_name + '_CLOSE', future_name + '_TRADED_VOLUME']
            x_df = x_df.merge(more_data.drop_duplicates(subset=['Timestamp']), on='Timestamp')
        count += 1
    return x_df

def get_df_with_features(df,futures,features):
    x_df = get_x_df(futures)
    x_df = engineer_features(x_df, futures, features)
    x_df = x_df.dropna()
    x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'])
    return x_df

def get_date_dict(futures,time_interval,features,switch_date,no_of_states,model):
    df = merge_df_reduce(futures[0] + futures[1], time_interval, features)
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
    date_dict = regime_and_switch(df, futures, features, time_interval, switch_date, target=['correlations', 'this'], model_type=model, components=no_of_states, partition='match_distribution')
    return date_dict

def create_regime_dataframes(no_of_states,futures,features):
    futures_and_features = ['Timestamp']
    x_df = get_x_df(futures)
    x_df = get_df_with_features(x_df,futures,features)
    date_dict = get_date_dict([['ZS', 'ZM'],[]],'weekly',features,'2017-09-04',no_of_states,'GradientBoost')
    features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
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
    

def all_data_df(start_date,switch_date,futures,features,regime,no_of_states):
    train_state_dataframes,test_set_dataframes = create_regime_dataframes(no_of_states,futures,features)
    rows,columns = train_state_dataframes[regime].shape
    features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
    futures_and_features = ['Timestamp']
    for future in futures:
        for feature in features1:
            futures_and_features.append(future + '_' + feature)
    all_data_df = pd.DataFrame(columns = futures_and_features)
    x_df = get_x_df(futures)
    x_df = get_df_with_features(x_df,futures,features)
    mask = (x_df['Timestamp'] >= pd.Timestamp(start_date))  & (x_df['Timestamp'] < pd.Timestamp(switch_date))  
    all_data_df = x_df.loc[mask]
    all_data_df.columns = futures_and_features
    return all_data_df


def get_x_y(df,lag):
    rows,columns = df.shape
    x_df = np.zeros((rows - lag,columns*lag))
    y_df = np.zeros((rows - lag,))
    count = 0
    for i in range(lag,rows):
        if i <= rows - 1:
            y_df[count] = df.ix[i, 'ZM_CLOSE']
            arr1 = df.ix[i-lag:i-1,:].values
            x_df[count] = arr1.reshape(columns*lag,)
            count += 1
    x_df = pd.DataFrame(x_df)
    y_df = pd.DataFrame(y_df)
    return x_df,y_df

def Lasso(futures,features,no_of_states,regime):
    start_date = '2015-01-02'
    switch_date = '2017-09-04'
    features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
    futures_and_features = []
    for future in futures:
        for feature in features1:
            futures_and_features.append(future + '_' + feature)
    train_state_dataframes,test_state_dataframes = create_regime_dataframes(no_of_states,futures,features)
    if train_state_dataframes[regime].empty or test_state_dataframes[regime].empty:
        return "No Data for this Regime"

    min_max_scaler = preprocessing.MinMaxScaler()

    df_model1 = all_data_df(start_date,switch_date,futures,features,regime,no_of_states)
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

    model1 = linear_model.Lasso(alpha=0.01)
    model2 = linear_model.Lasso(alpha=0.01)
    model1.fit(x_train_model1,y_train_model1)
    model2.fit(x_train_model2,y_train_model2)

    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)

    score1 = model1.score(x_test,y_test)
    score2 = model2.score(x_test,y_test)

    mse1 = mean_squared_error(y_test, y_pred1)
    mse2 = mean_squared_error(y_test, y_pred2)

    return score1,score2,mse1,mse2


