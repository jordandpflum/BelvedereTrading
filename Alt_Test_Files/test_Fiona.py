
import pandas as pd
import numpy as np
from sklearn.multioutput import *
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM
from sklearn import preprocessing
from sklearn import linear_model
from Data_Master.data_split import *
from Data_Master.feature_engineering import *
from Data_Master.read_data import *
#from sklearn import linear_model
#from feature_hmm import *
#from Modeling.feature_hmm import *
#from Modeling.Lasso import *
#from Modeling.AltModels import *

"""
Test Data_Master/data_split.py (split into training and test set with 10% of data in test)
"""


#splitting test and training set with rolling prediction approach

futures = ['ZS', 'ZM']
#all_features = list(marketFeaturefunctions.keys())
time_interval = 'weekly'
df = merge_df_reduce(futures, time_interval)
#print(df.head())

df = target_generator(df, 'weekly_corr_diff_ZS-ZM', [5, 10])
#print(df)
df.tail()
#df.to_csv('df1.csv', index = False, header = True)
x = df.loc[:, ['weekly_corr_diff_ZS-ZM']]
y = df.loc[:, ['weekly_corr_diff_ZS-ZM + 5 weeks', 'weekly_corr_diff_ZS-ZM + 10 weeks']]
# reserve 10% of data for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)


#no_of_weeks = 20
#model = CorrelationPredictor()
#model.fit()
#reg = model.predict_future_correlations(no_of_weeks)
#prediction_weeks = y.isnull().sum()
#calculate_rolling_prediction(reg, x_train, x_test, y_train, y_test, prediction_weeks)
