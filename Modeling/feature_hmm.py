# HMM correlation change prediction implimentation by sampling from next predicted state
import os
import datetime
import calendar
from Data_Master.feature_engineering import *
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing

#Code for read data
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
        file = '../Data_Master/fixed_belvedere_data/' + future_name + '.csv'
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
def get_y_df(futures_names, time_intervals, difference_columns=True):
    """
    Get the "y-data" - finding the correlations between all pairs of futures in futures_names for the given time interval.
    :param futures_names: A list of future contract names (e.g. "BZ"),
    time_intervals: a time interval length for the correlations (options: weekly, biweekly, monthly, seasonal)
    :return: A pandas data frame with the number of columns being the number of pairs of contracts (e.g. for 5 futures contracts, there are 5*(5-1) = 20 pairs) * number of time intervals for each pair
    Should be merged with an x-dataframe prior to modeling
    """

    pairs = []
    for future_name1 in futures_names:
        for future_name2 in futures_names:
            if future_name1 + '-' + future_name2 not in pairs and future_name2 + '-' + future_name1 not in pairs and future_name1 != future_name2:
                pairs.append(future_name1 + '-' + future_name2)

    if time_intervals == 'weekly':
        # get all the week files and sort the dates
        weeks = []

        for filename in os.listdir('../Data_Master/correlations/cor_res_weekly/'):
            if filename != '.DS_Store':
                a,b,c,d,e = filename.split('_')
                week_number,f = e.split('.')
                weeks.append(week_number)
        sorted_weeks = sorted(weeks)

        out_df = pd.DataFrame(columns=['Timestamp'] + ['weekly_corr_' + pair for pair in pairs])

        row_count = 0
        # for each week, append a row for each day with the correlation value
        for week_number in sorted_weeks:
            year,week = week_number.split('-')
            if(int(year) in [2015,2016,2017]):
                d = year + '-W' + week
                start_date = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
                end_date = start_date + datetime.timedelta(days=4)

                file = '../Data_Master/correlations/cor_res_weekly/All_oc_cor_week_' + week_number + '.csv'
                corr_matrix = pd.read_csv(file)
                corr_matrix.set_index('Unnamed: 0', inplace=True)

                future_pair_corrs = []
                for future_pair in pairs:
                    future_name1,future_name2 = future_pair.split('-')
                    future_pair_corrs.append(corr_matrix.loc[future_name1 + '_oc', future_name2 + '_oc'])

                date = start_date + datetime.timedelta(days=1)
                corrs = [future_pair_corr for future_pair_corr in future_pair_corrs]
                out_df.loc[row_count] = [pd.Timestamp(date)] + corrs
                row_count += 1

        # if we want to include the differences, compute the correlation difference from the last week in new columns
        if(difference_columns):
            for pair in pairs:
                out_df['weekly_corr_diff_' + pair] = out_df['weekly_corr_' + pair].diff()

        out_df.fillna(method='bfill', inplace = True)
        out_df.fillna(method='ffill', inplace = True)
        return out_df
    elif time_intervals == 'biweekly':
        # get all the biweekly files and sort the dates
        weeks = []

        for filename in os.listdir('../Data_Master/correlations/cor_res_biweek/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                week_number, f = e.split('.')
                weeks.append(week_number)
        sorted_weeks = sorted(weeks)
        out_df = pd.DataFrame(columns=['Timestamp'] + ['biweekly_corr_' + pair for pair in pairs])

        row_count = 0
        # fill in a row for each day in the two weeks with the correlation value
        for week_number in sorted_weeks:
            first_week_number = week_number[0:7]
            year, week = first_week_number.split('-')
            if (int(year) in [2015, 2016, 2017]):
                d = year + '-W' + week
                start_date = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
                end_date = start_date + datetime.timedelta(days=11) # one week from starting week's Friday

                file = '../Data_Master/correlations/cor_res_biweek/All_oc_cor_week_' + week_number + '.csv'
                corr_matrix = pd.read_csv(file)
                corr_matrix.set_index('Unnamed: 0', inplace=True)

                future_pair_corrs = []
                for future_pair in pairs:
                    future_name1, future_name2 = future_pair.split('-')
                    future_pair_corrs.append(corr_matrix.loc[future_name1 + '_oc', future_name2 + '_oc'])

                for i in range(12):
                    if(i != 5 and i != 6): #skip saturday and Sunday
                        date = start_date + datetime.timedelta(days=i)
                        corrs = [future_pair_corr for future_pair_corr in future_pair_corrs]
                        out_df.loc[row_count] = [pd.Timestamp(date)] + corrs
                        row_count += 1

        # if we want to include differences, compute correlation difference from the last two week period in new columns
        if (difference_columns):
            for pair in pairs:
                out_df['biweekly_corr_diff_' + pair] = out_df['biweekly_corr_' + pair].diff(periods=10)

        out_df.fillna(method='bfill', inplace = True)
        out_df.fillna(method='ffill', inplace = True)
        return out_df
    elif time_intervals == 'monthly':
        # get all the month files and sort the dates
        months = []

        for filename in os.listdir('../Data_Master/correlations/cor_res_month/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                month, f = e.split('.')
                months.append(month)
        sorted_months = sorted(months)

        out_df = pd.DataFrame(columns=['Timestamp'] + ['monthly_corr_' + pair for pair in pairs])

        row_count = 0
        # fill in a value for each day in each month with the correct correlation value
        for year_month in sorted_months:
            year, month = year_month.split('-')
            year = int(year)
            month = int(month.lstrip("0"))
            if (int(year) in [2015, 2016, 2017]):
                a, num_days = calendar.monthrange(year, month)
                start_date = datetime.date(year, month, 1)
                end_date = datetime.date(year, month, num_days)

                file = '../Data_Master/correlations/cor_res_month/All_oc_cor_month_' + year_month + '.csv'
                corr_matrix = pd.read_csv(file)
                corr_matrix.set_index('Unnamed: 0', inplace=True)

                future_pair_corrs = []
                for future_pair in pairs:
                    future_name1, future_name2 = future_pair.split('-')
                    future_pair_corrs.append(corr_matrix.loc[future_name1 + '_oc', future_name2 + '_oc'])

                for i in range(num_days):
                    date = start_date + datetime.timedelta(days=i)
                    if (date.weekday() < 5):  # skip saturdays and Sundays
                        corrs = [future_pair_corr for future_pair_corr in future_pair_corrs]
                        out_df.loc[row_count] = [pd.Timestamp(date)] + corrs
                        row_count += 1

        # if we want to include the differences, compute the correlation difference from the last month in new columns
        # (somewhat more complicated since there are a different number of days in each month)
        if(difference_columns):
            months_pairs = {}
            for pair in pairs:
                out_df['monthly_corr_diff_' + pair] = ''
                months_pairs[pair] = {}
                for year in [2015,2016,2017]:
                    months_pairs[pair][year] = {}
                    for month in range(1,13):
                        months_pairs[pair][year][month] = 'N/A'
            for i in range(out_df.shape[0]):
                timestamp = out_df.loc[i]['Timestamp'].to_pydatetime()
                year,month = int(timestamp.year),int(timestamp.month)
                for pair in pairs:
                    corr = out_df.loc[i]['monthly_corr_' + pair]
                    if(months_pairs[pair][year][month] == 'N/A'):
                        months_pairs[pair][year][month] = corr
                    if(year == 2015 and month == 1):
                        continue
                    else:
                        if(month > 1):
                            out_df.loc[(i,'monthly_corr_diff_' + pair)] = months_pairs[pair][year][month] - months_pairs[pair][year][month - 1]
                        else:
                            out_df.loc[(i,'monthly_corr_diff_' + pair)] = months_pairs[pair][year][month] - months_pairs[pair][year - 1][12]
        out_df.fillna(method='bfill', inplace = True)
        out_df.fillna(method='ffill', inplace = True)
        return out_df
    elif time_intervals == 'seasonal':
        # get all the seasonal files and sort the dates
        quarters = []

        for filename in os.listdir('../Data_Master/correlations/cor_res_season/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                quarter, f = e.split('.')
                quarters.append(quarter)
        sorted_quarters = sorted(quarters)

        out_df = pd.DataFrame(columns=['Timestamp'] + ['seasonal_corr_' + pair for pair in pairs])

        row_count = 0
        # fill in a value for each day in each quarter with the correct correlation value
        for quarter in sorted_quarters:
            first_year_month = quarter[0:7]
            year, month = first_year_month.split('-')
            year = int(year)
            month = int(month.lstrip("0"))
            second_month,third_month = month + 1, month + 2
            if (int(year) in [2015, 2016, 2017]):
                a, num_days_first = calendar.monthrange(year, month)
                a, num_days_second = calendar.monthrange(year, second_month)
                a, num_days_third = calendar.monthrange(year, third_month)
                total_num_days = num_days_first + num_days_second + num_days_third
                start_date = datetime.date(year, month, 1)
                end_date = datetime.date(year, third_month, num_days_third)

                file = '../Data_Master/correlations/cor_res_season/All_oc_cor_month_' + quarter + '.csv'
                corr_matrix = pd.read_csv(file)
                corr_matrix.set_index('Unnamed: 0', inplace=True)

                future_pair_corrs = []
                for future_pair in pairs:
                    future_name1, future_name2 = future_pair.split('-')
                    future_pair_corrs.append(corr_matrix.loc[future_name1 + '_oc', future_name2 + '_oc'])

                for i in range(total_num_days):
                    date = start_date + datetime.timedelta(days=i)
                    if (date.weekday() < 5):  # skip saturdays and Sundays
                        corrs = [future_pair_corr for future_pair_corr in future_pair_corrs]
                        out_df.loc[row_count] = [pd.Timestamp(date)] + corrs
                        row_count += 1

        # if we want to include the differences, compute the correlation difference from the last quarter in new columns
        # (somewhat more complicated since there are a different number of days in each quarter)
        if (difference_columns):
            months_pairs = {}
            for pair in pairs:
                out_df['seasonal_corr_diff_' + pair] = ''
                months_pairs[pair] = {}
                for year in [2015, 2016, 2017]:
                    months_pairs[pair][year] = {}
                    for month in range(1, 13):
                        months_pairs[pair][year][month] = 'N/A'
            for i in range(out_df.shape[0]):
                timestamp = out_df.loc[i]['Timestamp'].to_pydatetime()
                year, month = int(timestamp.year), int(timestamp.month)
                for pair in pairs:
                    corr = out_df.loc[i]['seasonal_corr_' + pair]
                    if (months_pairs[pair][year][month] == 'N/A'):
                        months_pairs[pair][year][month] = corr
                    if (year == 2015 and month < 4):
                        continue
                    else:
                        if (month > 3):
                            out_df.loc[(i, 'seasonal_corr_diff_' + pair)] = months_pairs[pair][year][month] - \
                                                                            months_pairs[pair][year][month - 3]
                        else:
                            if(year > 2015):
                                if(month == 1):
                                    prev_month = 10
                                elif(month == 2):
                                    prev_month = 11
                                elif(month == 3):
                                    prev_month = 12
                                out_df.loc[(i, 'seasonal_corr_diff_' + pair)] = months_pairs[pair][year][month] - \
                                                                                months_pairs[pair][year - 1][prev_month]
        out_df.fillna(method='bfill', inplace = True)
        out_df.fillna(method='ffill', inplace = True)
        return out_df
def helper_merge_reduce_x(futures_names):
    """
    Function to grab original futures data and place it into a dataframe without additional features
    and without target variables (just the "x-data").
    :param: a list of futures contracts names (e.g. "BZ")
    :return: pandas df with a timestamp and 5*28 columns (open,high,low,close,volume) for each contract.
    Should be merged with a y-dataframe
    """
    x_df = pd.DataFrame()

    count = 0
    for future_name in futures_names:
        file = '../Data_Master/fixed_belvedere_data/' + future_name + '.csv'
        if x_df.empty:
            x_df = pd.read_csv(file, sep='\t')
            x_df.columns = ['Timestamp', future_name + '_OPEN', future_name + '_HIGH', future_name + '_LOW', future_name + '_CLOSE', future_name + '_TRADED_VOLUME']
        else:
            more_data = pd.read_csv(file, sep='\t')
            more_data.columns = ['Timestamp', future_name + '_OPEN', future_name + '_HIGH', future_name + '_LOW', future_name + '_CLOSE', future_name + '_TRADED_VOLUME']
            x_df = x_df.merge(more_data.drop_duplicates(subset=['Timestamp']), on='Timestamp')
        count += 1
    x_df.Timestamp.apply(str)
    x_df.Timestamp = x_df.Timestamp.str.slice(0,10)
    x_df = x_df.drop_duplicates(subset='Timestamp', keep='first', inplace=False)
    return x_df
def merge_df_reduce(futures, time_intervals):
    """

    :param futures:
    :param time_intervals:
    :return:
    """
    x = helper_merge_reduce_x(futures)
    y = get_y_df(futures, time_intervals)
    x_header = list(x.columns.values)
    y_header = list(y.columns.values)
    joint_df = pd.DataFrame(columns=x_header)
    joint_df = joint_df.append(x, ignore_index=True, sort=False)
    for i in y_header[1:]:
        name = str(i)
        y_timestamp = []
        for times in y[y.columns[0]]:
            y_timestamp.append((str(times))[:10])
        corr = {}
        for item in joint_df[joint_df.columns[0]]:
            corr[item] = None
            if item[:10] in y_timestamp:
                corr[item] = (y.loc[y['Timestamp'] == str(item[:10])]).iloc[0][name]
        joint_df[str(name)] = joint_df['Timestamp'].map(corr)
    return joint_df

class CorrelationPredictor():
    def __init__(self, test_size=0.33,
                 n_hidden_states=4, n_latency_weeks= 30):
        """
        :param test_size: Fraction of the dataset to kept aside for the test set
        :param n_hidden_states: Number of hidden states
        :param n_latency_weeks: Number of weeks to be used as history for prediction
        """

        self.n_latency_weeks = n_latency_weeks

        self.hmm = GaussianHMM(n_components=n_hidden_states,n_iter=2000)
        features_names = ['OBV','MACDEXT','MACDFIX']
        self._split_train_test_data(test_size, 'ZM', 'ZS', features_names) # Enter correlation pairs here

    def _split_train_test_data(self, test_size, future1, future2, features_names):
        """
        :params future 1 and future2: Future pairs whose correlations we want to predict. For example if we want to predict correlations between ZS and ZM, plug in future1 as "ZS" and future2 as "ZM"
        """
        x_df = merge_df_reduce([future1, future2], 'weekly')
        x_df = x_df.dropna()
        x_df = engineer_features(x_df, [future1, future2], features_names)
        x_df.drop(x_df.iloc[:, 0:12], inplace=True, axis=1)
        x_df = x_df.fillna(0)
        # normalized the features
        for i in features_names:
            x_df[future1+'_'+i] = (x_df[future1+'_'+i] - x_df[future1+'_'+i].mean()) / (x_df[future1+'_'+i].max() - x_df[future1+'_'+i].min())
            x_df[future2+'_'+i] = (x_df[future2+'_'+i] - x_df[future2+'_'+i].mean()) / (x_df[future2+'_'+i].max() - x_df[future2+'_'+i].min())
        corr_with_stamp = x_df
        _train_data_with_stamp, test_data_with_stamp = train_test_split(corr_with_stamp, test_size=test_size, shuffle=False)
        _train_data = _train_data_with_stamp
        test_data = test_data_with_stamp
        self._train_data = _train_data
        self.test_data = test_data
        self.ES_corr_train = self._train_data
        self.ES_corr_test = self.test_data

    def fit(self):
        # Estimates HMM parameters
        vec = np.array(self.ES_corr_train.values.reshape(-1,7), dtype=np.float64)
        self.hmm.fit(vec)

    def predict(self, df):
        # Finds the most likely state sequence (Vitterbi Algorithm)
        pred = self.hmm.predict(df.values.reshape(-1,7))
        return pred

    def predict_future_correlations(self, weeks):
        """
        Function which predicts future correlation changes.
        :param weeks: Number of weeks we want to predict
        """
        hmm = self.hmm
        predicted_next_correlations = pd.DataFrame()
        for day_index in range(self.n_latency_weeks,self.n_latency_weeks + weeks):
            state_sequence = self.predict(self.ES_corr_test.iloc[0:day_index])
            next_state_probs = np.array(hmm.transmat_[state_sequence[-1], :])
            state_vector = np.arange(len(next_state_probs))
            mean_vector = np.zeros((len(next_state_probs), 7))
            for j in range(len(next_state_probs)):
                mean_vector[j] = hmm.means_[state_vector[j]]
            sum = np.zeros(7)
            for i in range(0,len(mean_vector)):
                temp = np.dot(mean_vector[i],next_state_probs[i])
                for j in range(len(temp)):
                    sum[j] += temp[j]
            prediction = pd.Series(sum)
            predicted_next_correlations = predicted_next_correlations.append(pd.Series(prediction),ignore_index = True)
        return predicted_next_correlations

future1 = "ZM"
future2 = "ZS"
no_of_weeks = 20
model = CorrelationPredictor()
model.fit()
predictions = model.predict_future_correlations(no_of_weeks)
print("True Correlation changes are:")
data = model.ES_corr_test.iloc[model.n_latency_weeks:model.n_latency_weeks + no_of_weeks]
print(data["weekly_corr_diff_" + future1 + "-" + future2])
print("Predicted Correlation changes are:")
print(predictions)

true_changes = np.array(model.ES_corr_test.iloc[model.n_latency_weeks:model.n_latency_weeks + no_of_weeks])
binary_true_changes = np.copy(true_changes)
binary_true_changes[binary_true_changes < 0] = 0
binary_true_changes[binary_true_changes != 0] = 1
predictions = np.array(predictions)
binary_predicted_changes = np.copy(predictions)
binary_predicted_changes[binary_predicted_changes < 0] = 0
binary_predicted_changes[binary_predicted_changes != 0] = 1
max = np.amax(true_changes)
min = np.amin(true_changes)
epsilon = ((max - min)/2) *0.2
direction_accuracy = np.mean(binary_true_changes == binary_predicted_changes) * 100
average_squared_error = (1/len(binary_true_changes))*np.sum(np.square(true_changes - predictions))
proximity_accuracy = np.mean(np.abs(true_changes - predictions) <= epsilon) * 100
print("Direction Accuracy is:")
print(str(direction_accuracy) + "%")
print("Average Squared Error is:")
print(average_squared_error)
print("Proximity Accuracy is:")
print(str(proximity_accuracy) + "%")

#Plotting the true correlation changes with the predicted correlation changes
weeks = np.linspace(0, no_of_weeks, no_of_weeks)
plt.plot(weeks, data["weekly_corr_diff_" + future1 + "-" + future2], color = "blue", label = "True correlation changes")
plt.plot(weeks, predictions[:,0], color = "red", label = "Predicted correlation changes")
plt.legend()
plt.xlabel("Weeks")
plt.ylabel("Correlation changes")
plt.title(future1 + "-" + future2)
plt.show()
