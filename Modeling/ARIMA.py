## Decided not proceed
# Just for project records

from AltModels import *
import numpy as numpy
import pandas as pd
#from read_data import *
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.stattools as ts
from pyramid.arima import auto_arima
import math
#from feature_hmm import *

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


futures = ["ZS","ZM"]
time_intervals = 'weekly'

df = merge_df_reduce(futures, time_intervals)
#print(df)

data = df[['Timestamp', 'weekly_corr_ZS-ZM']]
train = data.dropna(axis = 0, how ='any')
print(train)

#Dickey-Fuller Test
result = float(ts.adfuller(train))
result
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# ARIMA model
# use auto_arima from the pyramid library
Arima_model=auto_arima(train, start_p=0, start_q=0, max_p=10, max_q=10, start_P=0, start_Q=0, max_P=10, max_Q=10, m=52, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=30)

# Validation
def mda(actual: np.ndarray, predicted: np.ndarray):
    """
    mean directional accuracy, indicates the degree to which the model accurately forecasts the directional changes in cancellation frequency from week to week
    parameter:
    actual: np.ndarray
    predicted: np.ndarray
    """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

#print(mda(val, predictions))


#calculate mse and rmse
mse = mean_squared_error(val, predictions)
rmse = math.sqrt(mse)
print('RMSE: %f' % rmse)


#Test stationary
df = pd.read_csv("/Users/fionazh/Desktop/ZS_sub.csv")
close = df['CLOSE']
close

# transform log close price
# visualise time series
# generate autocorrelation and partial autocorrelation plots
lnclose = np.log(close)
lnclose
plt.plot(lnclose)
plt.show()

acf_1 = acf(lnclose)[1:20]
test_df = pd.DataFrame([acf_1]).T
test_df.columns = ['Autocorrelation']
test_df.index += 1
test_df.plot(kind = 'bar')
plt.show()

pacf_1 = pacf(lnclose)[1:20]
test_df = pd.DataFrame([pacf_1]).T
test_df.columns = ['Partial Autocorrelation']
test_df.index += 1
test_df.plot(kind = 'bar')
plt.show()

# Dickey Fuller test
## up and down pattern is what to expect in a stationary time series
result = ts.adfuller(lnclose, 1)
result
lnclose_diff = lnclose - lnclose.shift()
diff = lnclose_diff.dropna()
acf_1_diff = acf(diff)[1:20]
test_df = pd.DataFrame([acf_1_diff]).T
test_df.columns = ['First Difference Autocorrelation']
test_df.index += 1
test_df.plot(kind ='bar')
pacf_1_diff = pacf(diff)[1:20]
plt.plot(pacf_1_diff)
plt.show()

##  pattern for Partial Autocorrelation
test_df = pd.DataFrame([pacf_1_diff]).T
test_df.columns = ['First Difference Partial Autocorrelation']
test_df.index += 1
test_df.plot(kind = 'bar')
plt.show()
