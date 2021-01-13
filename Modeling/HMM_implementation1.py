#HMM price prediction using Maximum Likelihood Estimation. This is the initial code and is still yet to be made more efficient and will be extended to predict correlation changes.


import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
 

 # Supress warning in hmmlearn
warnings.filterwarnings("ignore")

 
class StockPredictor():
    def __init__(self, test_size=0.33,n_hidden_states=4, n_latency_days= 100,n_steps_open=3, n_steps_high=3,n_steps_low=3, n_steps_close=3, n_steps_tv=5):
        """
        :param test_size: Fraction of the dataset to kept aside for the test set
        :param n_hidden_states: Number of hidden states
        :param n_latency_days: Number of weeks to be used as history for prediction 
        :param n_steps_open: Number of bins for open_price
        :param n_steps_high: Number of bins for high_price
        :param n_steps_low: Number of bins for low_price
        :param n_steps_close: Number of bins for close_price
        :param n_steps_tv: Number of bins for traded volume
        """
        self._init_logger()
 
        
        self.n_latency_days = n_latency_days
 
        self.hmm = GaussianHMM(n_components=n_hidden_states)
 
        self._split_train_test_data(test_size)
 
        self._compute_all_possible_outcomes(
            n_steps_open, n_steps_high, n_steps_low,n_steps_close,n_steps_tv)
 
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
 
    def _split_train_test_data(self, test_size):
        BZ = pd.read_csv("BZ.csv") # Read BZ prices (can enter any future). Price CSV files are in the Data Master\Belvedere Data Directory.
        self.price_table = BZ.drop(columns = ['Timestamp'])
        _train_data, test_data = train_test_split(
            self.price_table, test_size=test_size, shuffle=False)
 
        self._train_data = _train_data
        self._test_data = test_data
        max_values = np.array(self._train_data.max(axis = 0))
        min_values = np.array(self._train_data.min(axis = 0))
        self.open_min = min_values[0]
        self.close_min = min_values[1]
        self.high_min = min_values[2]
        self.low_min = min_values[3]
        self.tv_min = min_values[4]
        self.open_max = max_values[0]
        self.close_max = max_values[1]
        self.high_max = max_values[2]
        self.low_max = max_values[3]
        self.tv_max = max_values[4]
    #@staticmethod
    def _extract_features(data,price_table):
    	# Function which can be used to run the model on engineered features. For now, we just have prices.
        open_price = np.array(price_table['OPEN'])
        close_price = np.array(price_table['CLOSE'])
        high_price = np.array(price_table['HIGH'])
        low_price = np.array(price_table['LOW'])
        traded_volume = np.array(price_table['TRADED_VOLUME'])
        return np.column_stack((open_price, close_price, high_price, low_price, traded_volume))
 
    def fit(self):
        self._logger.info('>>> Extracting Features')
        feature_vector = StockPredictor._extract_features(self._train_data,self.price_table)
        self._logger.info('Features extraction Completed <<<')
 
        self.hmm.fit(feature_vector)
 
    def _compute_all_possible_outcomes(self, n_steps_open,
                                       n_steps_high, n_steps_low,n_steps_close,n_steps_tv):
    	# Function which computes all possible outcomes for the next observation.
        self.open_range = np.linspace(self.open_min, self.open_max, n_steps_open)
        self.high_range = np.linspace(self.high_min, self.high_max, n_steps_high)
        self.low_range = np.linspace(self.low_min, self.low_max, n_steps_low)
        self.close_range = np.linspace(self.close_min, self.close_max, n_steps_close)
        self.tv_range = np.linspace(self.tv_min, self.tv_max, n_steps_tv)
 
        self._possible_outcomes = np.array(list(itertools.product(
            self.open_range, self.high_range, self.low_range,self.close_range,self.tv_range)))
 
    def _get_most_probable_outcome(self, day_index):
    	# Function which calculates the outcome which maximizes the likelihood.
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data.iloc[previous_data_end_index: previous_data_start_index]
        previous_data_features = StockPredictor._extract_features(
            previous_data,self.price_table)
 
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]
 
        return most_probable_outcome
 
    def predict_next_price(self, day_index):
        # Function which uses the most probable outcome to make prediction.
        predicted_open_price, predicted_high_price, predicted_low_price,predicted_close_price,predicted_traded_volume = self._get_most_probable_outcome(
            day_index)
        x = pd.Series([predicted_open_price, predicted_high_price, predicted_low_price,predicted_close_price,predicted_traded_volume])
        return x
    def predict_next_prices_for_minutes(self, minutes):
    	# Function which predicts prices for the future minutes. Predicted prices will be the upper end of the binned ranges.
        predicted_next_prices = pd.DataFrame()
        for day_index in tqdm(range(self.n_latency_days,self.n_latency_days + minutes)):
            predicted_next_prices = predicted_next_prices.append(self.predict_next_price(day_index),ignore_index = True)
        return predicted_next_prices
 
minutes = 20
stock_predictor = StockPredictor()
stock_predictor.fit()
next_prices = stock_predictor.predict_next_prices_for_minutes(minutes)

# Bin the testing data according to the bin ranges obtained from training set.
test_data_discrete = pd.DataFrame(columns = ['OPEN','HIGH','LOW','CLOSE','TV','OPEN_PRED','HIGH_PRED','LOW_PRED','CLOSE_PRED','TV_PRED'])
test_data_discrete['OPEN'],bins_open = pd.cut(stock_predictor._test_data.iloc[stock_predictor.n_latency_days:stock_predictor.n_latency_days + minutes,0],stock_predictor.open_range,retbins = True)
test_data_discrete['HIGH'],bins_high = pd.cut(stock_predictor._test_data.iloc[stock_predictor.n_latency_days:stock_predictor.n_latency_days + minutes,1],stock_predictor.high_range,retbins = True)
test_data_discrete['LOW'],bins_low = pd.cut(stock_predictor._test_data.iloc[stock_predictor.n_latency_days:stock_predictor.n_latency_days + minutes,2],stock_predictor.low_range,retbins = True)
test_data_discrete['CLOSE'],bins_close = pd.cut(stock_predictor._test_data.iloc[stock_predictor.n_latency_days:stock_predictor.n_latency_days + minutes,3],stock_predictor.close_range,retbins = True)
test_data_discrete['TV'],bins_tv = pd.cut(stock_predictor._test_data.iloc[stock_predictor.n_latency_days:stock_predictor.n_latency_days + minutes,4],stock_predictor.tv_range,retbins = True)
next_prices = next_prices.set_index(test_data_discrete.index)

#
test_data_discrete['OPEN_PRED'] = next_prices.iloc[:,0]
test_data_discrete['HIGH_PRED'] = next_prices.iloc[:,1]
test_data_discrete['LOW_PRED'] = next_prices.iloc[:,2]
test_data_discrete['CLOSE_PRED'] = next_prices.iloc[:,3]
test_data_discrete['TV_PRED'] = next_prices.iloc[:,4]

# Calculate the accuracy of each feature. Accuracy = Number of samples which fall in the correct bin/ total number of samples
open_acc = ((pd.cut(test_data_discrete['OPEN_PRED'], bins=bins_open) == test_data_discrete['OPEN']).values.sum())/minutes
close_acc = ((pd.cut(test_data_discrete['CLOSE_PRED'], bins=bins_close) == test_data_discrete['CLOSE']).values.sum())/minutes
high_acc = ((pd.cut(test_data_discrete['HIGH_PRED'], bins=bins_high) == test_data_discrete['HIGH']).values.sum())/minutes
low_acc = ((pd.cut(test_data_discrete['LOW_PRED'], bins=bins_low) == test_data_discrete['LOW']).values.sum())/minutes
tv_acc = ((pd.cut(test_data_discrete['TV_PRED'], bins=bins_tv) == test_data_discrete['TV']).values.sum())/minutes

print("True Price data:")
print(stock_predictor._test_data[stock_predictor.n_latency_days:stock_predictor.n_latency_days + minutes])

print("True Binned prices with predicted prices:")
print(test_data_discrete)

print("Predicted Price Data:")
print(next_prices)

print("Open price accuracy is: " + str(open_acc*100) + "%")
print("Close price accuracy is: " + str(close_acc*100) + "%")
print("High price accuracy is: " + str(high_acc*100) + "%")
print("Low price accuracy is: " + str(low_acc*100) + "%")
print("Total Volume accuracy is: " + str(tv_acc*100) + "%")


	

            
 
 
