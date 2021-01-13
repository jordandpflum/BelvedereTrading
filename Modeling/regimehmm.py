from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import numpy as np
from Data_Master.read_data import *

class CorrelationPredictor():
    def __init__(self, test_size=0.33,
                 n_hidden_states=5, n_latency_weeks= 3, feature_size = 5):
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

def create_correlation_predictor(n_hidden_states):
    '''
    :param n_hidden_states: Number of hidden states
    :returns A nested dictionary with actual and predicted label assignments, and a the dates which receive this
        assignment. For example, if components=3, the dictionary might look like:
            regime_labels['actual'] =
        1: [[start_date1, end_date1], ...]
        2: [[start_date2, end_date2],...]
        3: [[start_date3, end_date3], ...]
        4: [[start_date4, end_date4], ...]
    regime_labels['predicted'] =
    1: [[start_date6, end_date6], ...]
    2: [[start_date7, end_date7],...]
    3: [[start_date8, end_date8], ...]
    4: [[start_date9, end_date9], ... ]
    '''
    no_of_weeks = 20
    model = CorrelationPredictor(n_hidden_states=n_hidden_states)

    model.fit()

    Timestamp = model.timestamp_col
    test_stamp = Timestamp.iloc[130:]
    np_Timestamp = test_stamp.to_numpy()
    np_Timestamp = np_Timestamp.ravel()
    new_df = pd.DataFrame()
    new_df['Timestamp'] = pd.Series(np_Timestamp)

    test_state = model.decode_test()
    test_state = test_state[1]
    train_state = model.decode_train()
    train_state = train_state[1]

    train_dict = {}
    test_dict = {}
    Timestamp_col_train = Timestamp.iloc[:130]
    np_Timestamp_train = Timestamp_col_train.to_numpy()
    np_Timestamp_train = np_Timestamp_train.ravel()

    for i in range(0, 129):
        timestamp = np_Timestamp_train[i]
        dates = pd.to_datetime(timestamp).date()
        dates = datetime.datetime(dates.year, dates.month, dates.day)
        if train_state[i] not in train_dict.keys():
            train_dict[train_state[i]] = [[dates, dates + datetime.timedelta(days=4)]]
        else:
            train_dict[train_state[i]].append([dates, dates + datetime.timedelta(days=4)])

    for i in range(0, 19):
        timestamp2 = np_Timestamp[i]
        date2 = pd.to_datetime(timestamp2).date()
        date2 = datetime.datetime(date2.year, date2.month, date2.day)
        if test_state[i] not in test_dict.keys():
            test_dict[test_state[i]] = [[date2, date2 + datetime.timedelta(days=4)]]
        else:
            test_dict[test_state[i]].append([date2, date2 + datetime.timedelta(days=4)])

    train_dict = dict(sorted(train_dict.items()))
    test_dict = dict(sorted(test_dict.items()))

    regime_labels = {'actual': train_dict,
                     'predicted': test_dict}

    return regime_labels

# def get_regime_labels_hmm():
#     """
#     this function return the regime labels of hmm model
#     :return:
#     """
#     return regime_labels

'''
print('regime labels')
np.save('regime_labels.npy', regime_labels)
read_dictionary = np.load('regime_labels.npy',allow_pickle='TRUE').item()
print(read_dictionary)
print(regime_labels)
'''

