from Data_Master.read_data import *
from Data_Master.feature_engineering import *
from Data_Exploration.polar_correlation_generator import *
from Modeling.AltModels import *
import csv
import ast
import operator

"""
TEST Data_Master/read_data.py
"""

# exclude "RTY", "PA"
all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC', 'LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
futures = all_futures_names[0:2]

x_df = get_x_df(futures)
print(x_df)
print(x_df.shape)
print(x_df.head)
# x_df.to_csv("~/Desktop/example_x_dataframe.csv", sep='\t')

x_df = engineer_features(x_df, futures, ['MACD', 'RSI', 'OBV', 'CHAIKIN'])

y_df = get_y_df(futures, time_intervals='biweekly', difference_columns=True)
print(y_df)
# y_df.to_csv("example_y_seasonal_dataframe.csv", sep='\t')

merged_df_reduced = merge_df_reduce(futures, time_intervals='weekly')
print(merged_df_reduced)
# merged_df_expanded = merge_df_expand(futures, time_intervals='monthly')
# print(merged_df_expanded)

fixed_data = fill_missing_data(["BZ"])
print(fixed_data)
fixed_data2 = fill_missing_data(['ES'])
print(fixed_data2)
#
# """
# TEST Data_Exploration/polar_correlation_generator.py
# """
#
all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC', 'LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
futures = all_futures_names[0:2]

x_df = get_x_df(futures)
x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'], utc=True).dt.tz_localize(tz=None) # timezone de-localization is necessary for interpreting timestamps
y_df = get_y_df(futures, time_intervals='weekly')
generate_polar_vis(y_df, futures, 'correlations', 'weekly', connect_first_and_last=True)


# GENERATES AND SAVES ALL VISUALIZATIONS

# create interesting groupings for exploration
# categories = {'Soybeans': ['ZL','ZM','ZS'],
#               'Livestock': ['HE', 'LE', 'GF'],
#               'Energy': ['CL', 'HO', 'NG'],
#               'Agriculture': ['ZC', 'ZS', 'ZW'],
#               'Metals': ['GC', 'PL', 'SI']}
#
#
# time_intervals = ['weekly','biweekly','monthly','seasonal']
# features = ['correlations','correlation_differences']
# connect_first_and_last = True
# for save_directory, futures_names in categories.items():
#     for time_interval in time_intervals:
#         for feature in features:
#             if(feature in ['open_price']):
#                 x_df = get_x_df(futures_names)
#                 x_df['Timestamp'] = pd.to_datetime(x_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
#                 for future in futures_names:
#                     generate_polar_vis(x_df, future, feature, time_interval, connect_first_and_last)
#             elif(feature in ['correlations', 'correlation_differences']):
#                 y_df = get_y_df(futures_names, time_intervals=time_interval)
#                 for pair in set(itertools.combinations(futures_names, 2)):
#                     generate_polar_vis(y_df, pair, feature, time_interval, connect_first_and_last, save_directory)
#                     print(pair)

""" 
TEST Data_Master/feature_engineering.py
"""
all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC', 'LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
futures = all_futures_names[0:2]

x_df = get_x_df(futures)

all_features = list(marketFeaturefunctions.keys())

all_features.remove('STOCHRSI') # not working

x_df = engineer_features(x_df, futures, all_features)
print(x_df)

"""
TEST Modeling/AltModels.py
"""

# futures[0] is the pair of interest to predict correlations, futures[1] is a list of additional futures to generate
# features from
futures = [['ZS', 'ZM'],[]]
# futures = [['ZS', 'ZM'],['ZL', 'ZC', 'ZW']]
all_features = list(marketFeaturefunctions.keys())

time_interval = 'weekly'
lags=[]
df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features, lags)

# use raw data in modeling
all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features

generate_mixture_model(df, futures, all_features, lags, time_interval, target="correlations", components=3)

switch_date = '2017-09-04'
regimes = regime_and_switch(df, futures, all_features, lags, time_interval, switch_date, target=['correlations', 'this'], model_type='GradientBoost', components=8, partition='match_distribution')

accuracy, features = generate_classification_model(df, futures, all_features, lags, time_interval, target=['correlations', 'this'], model_type='GradientBoost', components=3, partition='match_distribution')

# for time_interval in ['weekly', 'monthly']:
#     for n_components in [3, 5, 8]:
#         for partition in ['uniform', 'uniform_min_max', 'match_distribution']:
#             with open('classification_models_time_interval=' + time_interval + '_n_components=' + str(n_components) + '.csv', 'w') as csvfile:

# lags = []
# for additional_futures in [True, False]:
#     if(additional_futures):
#         with open('classification_model_tests_additional_futures.csv', 'w') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Partition Type', 'Correlations This Interval', 'Correlations Next Interval'])
#     else:
#         with open('classification_model_tests.csv', 'w') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Partition Type', 'Correlations This Interval', 'Correlations Next Interval'])
#     for model in ['NaiveBayes', 'RadialBasisSVM', 'PolynomialSVM', 'SigmoidSVM', 'RandomForest', 'GradientBoost', 'DecisionTree']:
#         for time_interval in ['weekly', 'monthly']:
#             futures = [['ZS', 'ZM'], ['ZL', 'ZC', 'ZW']]
#             all_features = list(marketFeaturefunctions.keys())
#             df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features)
#             all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features
#             for n_components in [3, 5, 8]:
#                 for partition in ['uniform', 'uniform_min_max', 'match_distribution']:
#                     row = []
#                     row.append(model)
#                     row.append(time_interval)
#                     row.append(n_components)
#                     row.append(partition)
#                     for corr_target in ['correlations']:
#                         for time_target in ['this', 'next']:
#                             print('Task: Classify ' + corr_target + ' for ' + time_target + ' time interval. \n')
#                             accuracy, features = generate_classification_model(df, futures, all_features, lags, time_interval, target=[corr_target, time_target], model_type=model, components=n_components, partition=partition)
#                             row.append(accuracy + '\n' + "Features: " + str(features))
#                     # with open('classification_models_time_interval=' + time_interval + '_n_components=' + str(n_components) + '.csv', 'a') as csvfile:
#                     if(additional_futures):
#                         with open('classification_model_tests_additional_futures.csv', 'a') as csvfile:
#                             writer = csv.writer(csvfile)
#                             writer.writerow(row)
#                     else:
#                         with open('classification_model_tests.csv', 'a') as csvfile:
#                             writer = csv.writer(csvfile)
#                             writer.writerow(row)

# for type in ['pair_only', 'pair_plus']:
#     for time_interval in ['weekly', 'monthly']:
#         if(type == 'pair_only'):
#             futures = [['ZS', 'ZM'],[]]
#         elif(type == 'pair_plus'):
#             futures = [['ZS', 'ZM'], ['ZL', 'ZC', 'ZW']]
#
#         all_features = list(marketFeaturefunctions.keys())
#         df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features)
#
#         # use raw data in modeling
#         all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features
#
#         targets = [['correlations'], ['this', 'next']]
#         model_types = ['RandomForest', 'GradientBoost', 'DecisionTree']
#         n_components_list = [3,5,8]
#         partitions = ['uniform', 'uniform_min_max', 'match_distribution']
#         features_dict = ensemble_feature_selection(df, futures, all_features, time_interval, targets, model_types, n_components_list, partitions)
#
#         f = open('feature_selection_' + time_interval + '_' + type + '.txt', 'w')
#         f.write(str(features_dict))
#         f.close()

# for fname in ['feature_selection_weekly_pair_only.txt', 'feature_selection_weekly_pair_plus.txt', 'feature_selection_monthly_pair_only.txt', 'feature_selection_monthly_pair_plus.txt']:
#     f = open(fname, "r")
#     contents = f.read()
#     f.close()
#     features = ast.literal_eval(contents)
#     sorted_features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)
#     name,txt = fname.split('.')
#     newname = name + '_sorted' + '.txt'
#     f = open(newname, 'w')
#     f.write(str(sorted_features))
#     f.close()

# for lagged_features in [False, True]:
#     if(lagged_features):
#         with open('classification_model_tests_lagged_features.csv', 'w') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Accuracy', 'Features'])
#         lags = [1,3,5]
#     else:
#         with open('classification_model_tests_no_lagged_features.csv', 'w') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Accuracy', 'Features'])
#         lags = []
#     for time_interval in ['weekly', 'monthly']:
#         futures = [['ZS', 'ZM'], []]
#         all_features = list(marketFeaturefunctions.keys())
#         df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features, lags)
#         all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features
#         for model_type in ['NaiveBayes', 'RadialBasisSVM', 'GradientBoost']:
#             for components in [3,5,8]:
#                 accuracy, features = generate_classification_model(df, futures, all_features, lags, time_interval, target=['correlations', 'this'], model_type=model_type, components=components, partition='match_distribution')
#                 row = [model_type, time_interval, components, accuracy, features]
#                 if(lagged_features):
#                     with open('classification_model_tests_lagged_features.csv', 'a') as csvfile:
#                         writer = csv.writer(csvfile)
#                         writer.writerow(row)
#                 else:
#                     with open('classification_model_tests_no_lagged_features.csv', 'a') as csvfile:
#                         writer = csv.writer(csvfile)
#                         writer.writerow(row)