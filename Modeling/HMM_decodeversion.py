# HMM correlation change prediction implimentation by sampling from next predicted state
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
        all_futures_names = ['ZM', 'ZS']
        feature_name = ['AROONOSC', 'PPO', 'STOCH_k','AROON_UP']
        x_df = merge_df_reduce(all_futures_names, 'weekly')
        x_df = engineer_features(x_df, all_futures_names, feature_name)
        x_df = x_df.dropna()
        self.timestamp_col = x_df.iloc[:, 0:1]
        x_df = x_df.iloc[:, 11:]
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
