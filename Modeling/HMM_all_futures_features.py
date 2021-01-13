# HMM correlation change prediction implimentation by sampling from next predicted state
import os
import datetime
import calendar
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from Data_Master.feature_engineering import *
from Data_Master.read_data import *

class CorrelationPredictor():
    def __init__(self, test_size=0.33,
                 n_hidden_states=4, n_latency_weeks= 3, feature_size = 5):
        """
        :param test_size: Fraction of the dataset to kept aside for the test set
        :param n_hidden_states: Number of hidden states
        :param n_latency_weeks: Number of weeks to be used as history for prediction
        """

        self.n_latency_weeks = n_latency_weeks

        self.hmm = GaussianHMM(n_components=n_hidden_states, n_iter=1000)

        self._split_train_test_data(test_size,feature_size)

    def _split_train_test_data(self, test_size, feature_size):
        """
        :params future 1 and future2: Future pairs whose correlations we want to predict. For example if we want to predict correlations between ZS and ZM, plug in future1 as "ZS" and future2 as "ZM"
        """
        all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO',
                             'RB', 'ZN', 'ZB', 'ZF', 'CL',
                             'BZ', 'ZT', 'HE', 'ZC', 'LE',
                             'ZW', 'ZS', 'KE', 'GF', 'ZM',
                             'ZL', 'GE', 'GC', 'HG', 'SI',
                             'PL']
        print('top'+'_'+str(feature_size)+"_"+"feature_selected:")
        all_features_names = ['ULTOSC','AROONOSC', 'PPO', 'STOCH_k','AROON_UP',
                              'ADX', 'DX', 'APO', 'STOCH_d', 'MACD',
                              'ADXR', 'STOCHRSI_k', 'MACDEXT', 'CCI', 'ROCP',
                              'RSI', 'ROCR100', 'STOCH_Fast_k', 'MFI', 'PLUS_DI',
                              'STOCH_Fast_d', 'AROON_DOWN', 'STOCHRSI_d', 'MACDFIX', 'MINUS_DI',
                              'PLUS_DM', 'CMO', 'TRIX', 'CHAIKIN', 'WILLR',
                              'ROC', 'BOP', 'ROCR', 'MOM', 'OBV']
        feature_name = all_features_names[:feature_size]
        print(feature_name)
        x_df = merge_df_reduce(all_futures_names, 'weekly')
        x_df = engineer_features(x_df, all_futures_names, feature_name)
        x_df.drop(x_df.iloc[:, 0:131], inplace=True, axis=1)
        x_df = x_df.dropna()
        for i in feature_name :
            for future in all_futures_names:
                x_df[future+'_'+i] = (x_df[future+'_'+i] - x_df[future+'_'+i].mean()+0.00000001) / (0.00000001+x_df[future+'_'+i].max() - x_df[future+'_'+i].min())
        corr_with_stamp = x_df
        _train_data_with_stamp, test_data_with_stamp = train_test_split(corr_with_stamp, test_size=test_size, shuffle=False)
        self.ES_corr_train = _train_data_with_stamp
        self.ES_corr_test = test_data_with_stamp

    def fit(self):
        # Estimates HMM parameters
        self.hmm.fit(self.ES_corr_train)

    def decode_train(self):
        decoded_train = self.hmm.decode(self.ES_corr_train)
        return decoded_train

    def decode_test(self):
        decoded_test = self.hmm.decode(self.ES_corr_test)

        return decoded_test

    def predict(self,df):
        # Finds the most likely state sequence (Vitterbi Algorithm)
        pred = self.hmm.predict(df)
        return pred

    def predict_future_correlations(self,weeks):
        """
        Function which predicts future correlation changes.
        :param weeks: Number of weeks we want to predict
        """
        hmm = self.hmm
        predicted_next_correlations = pd.DataFrame()
        next_state = []
        for day_index in range(self.n_latency_weeks,self.n_latency_weeks + weeks):
            state_sequence = self.predict(self.ES_corr_test.iloc[0:day_index,:])
            next_state.append(state_sequence[-1])
            next_state_probs = np.array(hmm.transmat_[state_sequence[-1], :])
            state_vector = np.arange(len(next_state_probs))
            mean_matrix = np.zeros((self.ES_corr_train.shape[1],len(next_state_probs)))
            for j in range(len(next_state_probs)):
                mean_matrix[:,j] = hmm.means_[state_vector[j]]
            prediction = np.dot(mean_matrix,next_state_probs)
            predicted_next_correlations = predicted_next_correlations.append(pd.Series(prediction),ignore_index = True)
        predicted_next_correlations = predicted_next_correlations.iloc[:, 0:325]
        predicted_next_correlations['next_state'] = next_state
        return predicted_next_correlations


