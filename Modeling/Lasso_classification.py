from Modeling.AltModels import *
import numpy as numpy
import pandas as pd 
from Data_Master.read_data import *
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import matplotlib.pyplot as plt
import random
random.seed(0)
from datetime import datetime

def create_regime_dataframes(no_of_states,futures,pair_plus_futures,features,model,time_interval,switch_date,partition):
	"""
	Function which grabs the date ranges from the output of our classifcation models and generates training and testing dataframes
	for different regimes.
	:param no_of_states: Specifies the total number of regimes
	:param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
	:param pair_plus_futures: Specifies if additional futures(other than ZM,ZS) to be present in the dataframes
	:param features: Specifies features to be included
	:param model: Specifies the classification model used to identify regimes
	:time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
	:switch_date: All data before switch date is used for training and all data after switch date is used for testing
	"""
	futures1 = futures + pair_plus_futures
	futures_and_features = ['Timestamp']
	x_df = get_x_df(futures1)
	x_df = engineer_features(x_df, futures1, features)
	x_df = x_df.dropna()
	x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
	df = merge_df_reduce(futures + pair_plus_futures, time_interval, features)
	features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
	date_dict = regime_and_switch(df, [futures,pair_plus_futures], features1, [],time_interval, switch_date, target=['correlations', 'this'], model_type=model, components=no_of_states, partition=partition)
	
	for future in futures1:
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
			start = train_dict[label][j][0]
			end = train_dict[label][j][1]
			mask = (x_df['Timestamp'] >= pd.Timestamp(start))  & (x_df['Timestamp'] < pd.Timestamp(end))   
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
			start = test_dict[label][j][0]
			end = test_dict[label][j][1]
			mask = (x_df['Timestamp'] >= pd.Timestamp(start))  & (x_df['Timestamp'] < pd.Timestamp(end))   
			test_state_dataframes[i] = test_state_dataframes[i].append(x_df.loc[mask])
		i += 1
	return train_state_dataframes,test_state_dataframes



def all_data_df(no_of_states,futures,pair_plus_futures,regime,features,model,time_interval,switch_date,partition,subset = "False"):
	''''
	If subset = "False", returns a dataframe consisting of the entire training data (all the regime dataframes combined).
	If subset = "True", returns a dataframe consisting of data points from all the regimes but with a size equal to regime dataframe 
	corresponding to the regime specified by the 'regime' parameter.
	:param no_of_states: Specifies the total number of regimes
	:param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
	:param pair_plus_futures: Specifies if additional futures(other than ZM,ZS) to be present in the dataframes
	:param regime: Specifies the regime number in order for the size to be matched
	:param features: Specifies features to be included
	:param model: Specifies the classification model used to identify regimes
	:time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
	:switch_date: All data before switch date is used for training and all data after switch date is used for testing
	'''
	futures1 = futures + pair_plus_futures
	futures_and_features = ['Timestamp']
	train_state_dataframes,test_set_dataframes = create_regime_dataframes(no_of_states,futures,pair_plus_futures,features,model,time_interval,switch_date,partition)
	rows,columns = train_state_dataframes[regime].shape
	part_rows = math.floor(rows/no_of_states)
	x_df = get_x_df(futures1)
	x_df = engineer_features(x_df, futures1, features)
	x_df = x_df.dropna()
	x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
	df = merge_df_reduce(futures + pair_plus_futures, time_interval, features)
	features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
	date_dict = regime_and_switch(df, [futures,pair_plus_futures], features1, [],time_interval, switch_date, target=['correlations', 'this'], model_type=model, components=no_of_states, partition=partition)
	
	for future in futures1:
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
				start = train_dict[label][j][0]
				end = train_dict[label][j][1]
				mask = (x_df['Timestamp'] >= pd.Timestamp(start))  & (x_df['Timestamp'] < pd.Timestamp(end))   
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

def Lasso_Classification(futures,pair_plus_futures,features,no_of_states,regime,model,reg,time_interval,switch_date,partition):
	'''
	For a specific regime, fits two Lasso Regression models: 1 - Model trained on the entire training data (dataframe returned
	by all_data_df()) and 2 - Model trained on just the dataframe corresponding to that regime. Returns the Mean Absolute Error,
	Mean Squared Error and R2 Score for each of them (R2 score is just for interpretability).
	:param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
	:param pair_plus_futures: Specifies if additional futures(other than ZM,ZS) to be present in the dataframes
	:param features: Specifies features to be included
	:param no_of_states: Specifies the total number of regimes
	param regime: Specifies the regime number
	:param model: Specifies the classification model used to identify regimes
	: reg: Regularization Parameter
	:time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
	:switch_date: All data before switch date is used for training and all data after switch date is used for testing
	'''
	features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
	futures1 = futures + pair_plus_futures
	futures_and_features = []
	for future in futures1:
		for feature in features1:
			futures_and_features.append(future + '_' + feature)
	train_state_dataframes,test_state_dataframes = create_regime_dataframes(no_of_states,futures,pair_plus_futures,features,model,time_interval,switch_date,partition)
	
	if train_state_dataframes[regime].empty or test_state_dataframes[regime].empty:
		return "No Data for this Regime"

	min_max_scaler = preprocessing.MinMaxScaler()

	df_model1 = all_data_df(no_of_states,futures,pair_plus_futures,regime,features,model,time_interval,switch_date,partition,subset = "True")
	
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

def compare_future_add(no_of_states,futures,pair_plus_futures,features,model,regime,time_interval,switch_date,reg,partition):
	'''
	A function which compares the affect of adding features corresponding to correlated futures (ZL,ZC,ZW) vs having just the features
	of ZM,ZS for a specific regime. Returns MAE,MSE and R2 Score for both the models.
	:param no_of_states: Specifies the total number of regimes
	:param futures: Specifies the futures to be present in the dataframes (ZM,ZS)
	:param pair_plus_futures: Specifies if additional futures(other than ZM,ZS) to be present in the dataframes
	:param features: Specifies features to be included
	:param model: Specifies the classification model used to identify regimes
	:param regime: Specifies the regime number
	:time_interval: Specifies time interval used for correlation prediction(weekly or monthly)
	:switch_date: All data before switch date is used for training and all data after switch date is used for testing
	: reg: Regularization Parameter
	'''
	min_max_scaler = preprocessing.MinMaxScaler()
	features1 = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + features
	futures1 = futures + pair_plus_futures
	futures_and_features = []
	futures_and_features1 = []
	for future in futures1:
		for feature in features1:
			futures_and_features.append(future + '_' + feature)
	for future in futures:
		for feature in features1:
			futures_and_features1.append(future + '_' + feature)
	empty_pair_plus = []
	train_dataframes1,test_dataframes1 = create_regime_dataframes(no_of_states,futures,empty_pair_plus,features,model,time_interval,switch_date,partition)
	train_dataframes2,test_dataframes2 = create_regime_dataframes(no_of_states,futures,pair_plus_futures,features,model,time_interval,switch_date,partition)
	if train_dataframes1[regime].empty or test_dataframes1[regime].empty or train_dataframes2[regime].empty or test_dataframes2[regime].empty :
		return "No Data for this Regime"

	df_model1 = train_dataframes1[regime]
	df_model1 = df_model1.drop(['Timestamp'],axis = 1)
	df_model1 = pd.DataFrame(min_max_scaler.fit_transform(df_model1))
	df_model1.columns = futures_and_features1

	df_model2 = train_dataframes2[regime]
	df_model2 = df_model2.drop(['Timestamp'],axis = 1)
	df_model2 = pd.DataFrame(min_max_scaler.fit_transform(df_model2))
	df_model2.columns = futures_and_features

	df_test1 = test_dataframes1[regime]
	df_test1 = df_test1.drop(['Timestamp'],axis = 1)
	df_test1 = pd.DataFrame(min_max_scaler.fit_transform(df_test1))
	df_test1.columns = futures_and_features1

	df_test2 = test_dataframes2[regime]
	df_test2 = df_test2.drop(['Timestamp'],axis = 1)
	df_test2 = pd.DataFrame(min_max_scaler.fit_transform(df_test2))
	df_test2.columns = futures_and_features

	x_train_model1,y_train_model1 = get_x_y(df_model1,3)
	x_train_model2,y_train_model2 = get_x_y(df_model2,3)
	x_test1,y_test1 = get_x_y(df_test1,3)
	x_test2,y_test2 = get_x_y(df_test2,3)

	model1 = linear_model.Lasso(alpha=reg)
	model2 = linear_model.Lasso(alpha=reg)
	model1.fit(x_train_model1,y_train_model1)
	model2.fit(x_train_model2,y_train_model2)

	y_pred1 = model1.predict(x_test1)
	y_pred2 = model2.predict(x_test2)

	score1 = model1.score(x_test1,y_test1)
	score2 = model2.score(x_test2,y_test2)

	mse1 = mean_squared_error(y_test1, y_pred1)
	mse2 = mean_squared_error(y_test2, y_pred2)

	mae1 = mean_absolute_error(y_test1,y_pred1)
	mae2 = mean_absolute_error(y_test2,y_pred2)

	return score1,score2,mse1,mse2,mae1,mae2
