from Data_Master.read_data import *
from Data_Master.feature_engineering import *
from Data_Exploration.polar_correlation_generator import *
from Modeling.AltModels import *
import csv
import ast
import operator

print('Run classification model tests with pair-only and pair-plus settings.')

lags = []
for additional_futures in [True, False]:
    if(additional_futures):
        with open('classification_model_tests_additional_futures.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Partition Type', 'Correlations This Interval', 'Correlations Next Interval'])
    else:
        with open('classification_model_tests.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Partition Type', 'Correlations This Interval', 'Correlations Next Interval'])
    for model in ['NaiveBayes', 'RadialBasisSVM', 'PolynomialSVM', 'SigmoidSVM', 'RandomForest', 'GradientBoost', 'DecisionTree']:
        for time_interval in ['weekly', 'monthly']:
            futures = [['ZS', 'ZM'], ['ZL', 'ZC', 'ZW']]
            all_features = list(marketFeaturefunctions.keys())
            df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features)
            all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features
            for n_components in [3, 5, 8]:
                for partition in ['uniform', 'uniform_min_max', 'match_distribution']:
                    row = []
                    row.append(model)
                    row.append(time_interval)
                    row.append(n_components)
                    row.append(partition)
                    for corr_target in ['correlations']:
                        for time_target in ['this', 'next']:
                            print('Task: Classify ' + corr_target + ' for ' + time_target + ' time interval. \n')
                            accuracy, features = generate_classification_model(df, futures, all_features, lags, time_interval, target=[corr_target, time_target], model_type=model, components=n_components, partition=partition)
                            row.append(accuracy + '\n' + "Features: " + str(features))
                    # with open('classification_models_time_interval=' + time_interval + '_n_components=' + str(n_components) + '.csv', 'a') as csvfile:
                    if(additional_futures):
                        with open('classification_model_tests_additional_futures.csv', 'a') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(row)
                    else:
                        with open('classification_model_tests.csv', 'a') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(row)

print("Test ensemble feature selection and write to txt.")

for type in ['pair_only', 'pair_plus']:
    for time_interval in ['weekly', 'monthly']:
        if(type == 'pair_only'):
            futures = [['ZS', 'ZM'],[]]
        elif(type == 'pair_plus'):
            futures = [['ZS', 'ZM'], ['ZL', 'ZC', 'ZW']]

        all_features = list(marketFeaturefunctions.keys())
        df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features)

        # use raw data in modeling
        all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features

        targets = [['correlations'], ['this', 'next']]
        model_types = ['RandomForest', 'GradientBoost', 'DecisionTree']
        n_components_list = [3,5,8]
        partitions = ['uniform', 'uniform_min_max', 'match_distribution']
        features_dict = ensemble_feature_selection(df, futures, all_features, time_interval, targets, model_types, n_components_list, partitions)

        f = open('feature_selection_' + time_interval + '_' + type + '.txt', 'w')
        f.write(str(features_dict))
        f.close()


print('Test lagged features and write to csv.')

for lagged_features in [False, True]:
    if(lagged_features):
        with open('classification_model_tests_lagged_features.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Accuracy', 'Features'])
        lags = [1,3,5]
    else:
        with open('classification_model_tests_no_lagged_features.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model Type', 'Time Interval', 'Number of Components', 'Accuracy', 'Features'])
        lags = []
    for time_interval in ['weekly', 'monthly']:
        futures = [['ZS', 'ZM'], []]
        all_features = list(marketFeaturefunctions.keys())
        df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features, lags)
        all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features
        for model_type in ['NaiveBayes', 'RadialBasisSVM', 'GradientBoost']:
            for components in [3,5,8]:
                accuracy, features = generate_classification_model(df, futures, all_features, lags, time_interval, target=['correlations', 'this'], model_type=model_type, components=components, partition='match_distribution')
                row = [model_type, time_interval, components, accuracy, features]
                if(lagged_features):
                    with open('classification_model_tests_lagged_features.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(row)
                else:
                    with open('classification_model_tests_no_lagged_features.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(row)