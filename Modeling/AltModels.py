import pandas as pd
import numpy as np
from scipy import stats
import sklearn.mixture as mix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import calendar
import os
import datetime
from isoweek import Week


def prepare_data(df,futures,features,lags,time_interval,target):
    """
    Function which grabs the relevant data from the df based on futures and features.
    :param df: Dataframe to pull from.
    :param futures: futures[0] is the pair of futures we want to predict on. futures[1] is a list of additional futures
    to generate features from.
    :param features: List of features.
    :param lags: a list of lags for the features
    :param time_interval: Time interval
    :param target: target[0] is correlations or correlation_differences.
    :return: A reduced dataframe containing only relevant features and target columns with non-numerical values dropped.
    """
    futures_and_features = ['Timestamp']

    for future in futures[0]:
        for feature in features:
            futures_and_features.append(future + '_' + feature)
            for lag in lags:
                futures_and_features.append(future + '_' + feature + '_' + str(lag) + '-lag')


    # if there are "outside" futures in futures[1], this will add their features
    for future in futures[1]:
        for feature in features:
            futures_and_features.append(future + '_' + feature)
            for lag in lags:
                futures_and_features.append(future + '_' + feature + '_' + str(lag) + '-lag')

    if (target[0] == "correlations"):
        target_name = time_interval + '_' + 'corr_' + futures[0][0] + '-' + futures[0][1]
    elif (target[0] == 'correlation_differences'):
        target_name = time_interval + '_' + 'corr_diff_' + futures[0][0] + '-' + futures[0][1]
    futures_and_features.append(target_name)

    features_and_target_df = df[futures_and_features].dropna()
    return features_and_target_df

def generate_mixture_model(df, futures, features, lags, time_interval, target, components):
    """
    Generates and evaluates a Gaussian Mixture model given the parameters.
    :param df: Starting dataframe to read from.
    :param futures: Pair of futures.
    :param features: List of features to include in the model.
    :param lags: a list of lags for the features (if applicable)
    :param time_interval: Time interval
    :param target: target[0] is correlations or correlation_differences. target[1] specs 'this' or 'next' time interval.
    :param components: Number of components to break the target range into.
    :return: Nothing
    """

    data = prepare_data(df, futures, features, lags, time_interval, target)
    X = data.drop(columns=['Timestamp']).values
    model = mix.GaussianMixture(n_components=components, covariance_type='full', n_init=100, random_state=7).fit(X)

    hidden_states = model.predict(X)
    print(hidden_states)

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i][-1])
        print("var = ", np.diag(model.covariances_[i])[-1])
        print()

    score = model.score(X)
    print("Model Score: " + str(score))


def range_helper(value, target, components, partition='uniform', min=None, max=None, range_list=None):
    """
    Returns a label based on which section of the correlation range (-1,1) or (-2,2) the value falls into. For example,
    3 components breaks the uniformly partitioned range (-1,1) into:
    0) -1.0 to -0.333
    1) -0.333 to 0.333
    2) 0.0333 to 1.0
    :param value: The number to determine which part of the range it is in.
    :param target: If "correlations" range is (-1,1). If "correlation_differences", range is (-2,2).
    :param components: The number of components to break the range (-1,1) into.
    :param partition: Method for partitioning the range of values. If 'uniform', range is (-1,1) for correlations
    and (-2,2) for correlation changes. If 'uniform_min_max', range is (min(correlations), max(correlations)) broken up
    into |max(correlations) - min(correlations)|/components intervals. If 'match_distribution', range is partitioned to
    try to assign roughly the same number of labels to each group.
    :param min: minimum value to use if partition option 'uniform_min_max' is chosen.
    :param max: maximum value to use if partition option 'uniform_min_max' is chosen.
    :param range_list: the predefined range list to use if the partition option 'match_distribution' is chosen.
    :return: The part of the range the value falls into. Values can be 0 to components - 1.
    """

    if (partition == 'uniform'):
        if(target[0] == 'correlations'):
            val_range = np.arange(-1, 1, 2 / components)
        elif(target[0] == 'correlation_differences'):
            val_range = np.arange(-2, 2, 4 / components)
    elif (partition == 'uniform_min_max'):
        val_range = np.arange(min, max, abs(max-min) / components)
    elif (partition == 'match_distribution'):
        val_range = range_list
    for i in range(len(val_range)):
        if(i < len(val_range) - 1):
            if(val_range[i] <= value and value < val_range[i+1]):
                return i
        else:
            return i

def num_days_helper(timestamp, time_interval):
    """
    Get the number of days to add to the time interval the option "next" interval was selected.
    :param timestamp: Timestamp to reference from if "monthly" or "seasonal" time_interval is selected.
    :param time_interval: 'weekly', 'biweekly', 'monthly', 'seasonal'
    :return: The number of days to reach the next time interval from the current timestamp.
    """
    if(time_interval == 'weekly'):
        return 5
    elif(time_interval == 'biweekly'):
        return 10
    elif(time_interval == 'monthly'):
        year,month = timestamp[0:4], timestamp[5:7]
        a, num_days = calendar.monthrange(int(year), int(month))
        return num_days
    elif(time_interval == 'seasonal'):
        year, month = timestamp[0:4], timestamp[5:7]
        a, num_days_m1 = calendar.monthrange(int(year), int(month))
        if(int(month) <= 10):
            a, num_days_m2 = calendar.monthrange(int(year), int(month) + 1)
            a, num_days_m3 = calendar.monthrange(int(year), int(month) + 2)
        elif(int(month) == 11):
            a, num_days_m2 = calendar.monthrange(int(year), int(month) + 1)
            a, num_days_m3 = calendar.monthrange(int(year) + 1, 1)
        elif(int(month) == 12):
            a, num_days_m2 = calendar.monthrange(int(year) + 1, 1)
            a, num_days_m3 = calendar.monthrange(int(year) + 1, 2)

        return num_days_m1 + num_days_m2 + num_days_m3


def generate_classification_model(df,futures,features,lags,time_interval,target,model_type,components,partition):
    """
    :param df: Starting Dataframe to read from.
    :param futures: futures[0] is the pair of futures we want to predict on. futures[1] is a list of additional futures
    to generate features from.
    :param features: List of features to include in the model.
    :param lags: a list of lags for the futures and features pairs
    :param time_interval: Time interval
    :param target: target[0] is correlations or correlation_differences. target[1] specs 'this' or 'next' time interval.
    :param model_type: Type of classification model. Current acceptable values are 'NaiveBayes', 'RadialBasisSVM',
    'PolynomialSVM', 'SigmoidSVM', 'RandomForest', 'GradidentBoost', 'DecisionTree'
    :param components: Number of components to break the target range into.
    :param partition: Method for partitioning the range of values. If 'uniform', range is (-1,1) for correlations
    and (-2,2) for correlation changes. If 'uniform_min_max', range is (min(correlations), max(correlations)) broken up
    into |max(correlations) - min(correlations)|/components intervals. If 'match', range is partitioned to try to assign
    roughly the same number of labels to each group.
    :return: The 10-fold cross-validation accuracy mean and standard deviation, and the 10 most important features as
    determined by RFE, if applicable.
    """

    data = prepare_data(df, futures, features, lags, time_interval, target)

    if (target[0] == "correlations"):
        target_name = time_interval + '_' + 'corr_' + futures[0][0] + '-' + futures[0][1]
    elif(target[0] == 'correlation_differences'):
        target_name = time_interval + '_' + 'corr_diff_' + futures[0][0] + '-' + futures[0][1]

    if(partition == 'uniform'):
        min_target_val = None
        max_target_val = None
        range_list = None
    elif(partition == 'uniform_min_max'):
        min_target_val = np.min(data[target_name].values)
        max_target_val = np.max(data[target_name].values)
        range_list = None
    elif(partition == 'match_distribution'):
        min_target_val = None
        max_target_val = None

        target_items = data[target_name].values
        num_target_items = len(data[target_name].values)
        num_items_per_partition = round(num_target_items/components)
        count = 0
        range_list = []
        for i in range(len(np.sort(target_items))):
            if(count == 0 and len(range_list) < components):
                range_list.append(np.sort(target_items)[i])
            count += 1
            if (count == num_items_per_partition):
                count = 0

    if(target[1] == 'this'):
        data['class_labels'] = data[target_name].apply(range_helper, target=target, components=components, partition=partition, min=min_target_val, max=max_target_val, range_list=range_list)
    elif(target[1] == 'next'):
        data['class_labels'] = np.nan
        for i in range(data.shape[0]):
            num_days = num_days_helper(data.iloc[i]['Timestamp'], time_interval)
            if(i + num_days < data.shape[0]):
                value = data.iloc[i + num_days][target_name]
                data.iloc[i, data.columns.get_loc('class_labels')] = range_helper(value, target, components, partition=partition, min=min_target_val, max=max_target_val, range_list=range_list)

    data = data.dropna()
    X = data.drop(columns=['Timestamp', target_name, 'class_labels']).values
    y = data['class_labels']

    if(model_type == 'NaiveBayes'):
        model = GaussianNB()
    elif(model_type == 'RadialBasisSVM'):
        model = SVC(kernel='rbf')
    elif(model_type == 'PolynomialSVM'):
        model = SVC(kernel='poly')
    elif (model_type == 'SigmoidSVM'):
        model = SVC(kernel='sigmoid')
    elif(model_type == 'RandomForest'):
        model = RandomForestClassifier()
    elif(model_type == 'GradientBoost'):
        model = GradientBoostingClassifier()
    elif(model_type == 'DecisionTree'):
        model = DecisionTreeClassifier()

    print('10-fold Cross Validation score for ' + model_type + ' model: ')
    scores = cross_val_score(model, X, y, cv=10)
    print("\t Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return_accuracy = "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    return_features = []

    if(model_type in ['RandomForest', 'GradientBoost', 'DecisionTree']):
        selector = RFE(model, 10, step=1)
        selector.fit(X, y)
        print("This model supports recursive feature elimination. The most important features for this task were: ")
        columns = data.drop(columns=['Timestamp', target_name, 'class_labels']).columns
        for i in range(len(columns)):
            if(selector.support_[i] == True):
                print('\t' + columns[i])
                return_features.append(columns[i])

    print('\n')

    return return_accuracy,return_features


def get_date_ranges(time_interval):
    """
    Gets all start and end date pairs for the selected time interval.
    :param time_interval: 'weekly', 'biweekly', 'monthly', or 'seasonal'
    :return: A list of start/end date pairs.
    """
    date_ranges = []
    if time_interval == 'weekly':
        # get all the week files and sort the dates
        weeks = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_weekly/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                week_number, f = e.split('.')
                weeks.append(week_number)
        sorted_weeks = sorted(weeks)

        # for each week, append a row for each day with the correlation value
        for week_number in sorted_weeks:
            year, week = week_number.split('-')
            if (int(year) in [2015, 2016, 2017]):
                # d = year + '-W' + week
                # start_date = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
                w = Week(int(year), int(week))
                start_date = w.monday()
                end_date = start_date + datetime.timedelta(days=4)
                date_ranges.append([start_date, end_date])
    elif (time_interval == 'biweekly'):
        weeks = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_biweek/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                week_number, f = e.split('.')
                weeks.append(week_number)
        sorted_weeks = sorted(weeks)

        for week_number in sorted_weeks:
            first_week_number = week_number[0:7]
            second_week_number = week_number[7:14]
            first_week_number_year, first_week_number_week, = first_week_number.split('-')
            second_week_number_year, second_week_number_week = second_week_number.split('-')
            if (first_week_number_week.lstrip("0") in [str(i) for i in range(1, 53, 2)]):
                year, week = first_week_number_year, first_week_number_week
            else:
                year, week = second_week_number_year, second_week_number_week

            if (int(year) in [2015, 2016, 2017]):
                # d = year + '-W' + week
                # start_date = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
                w = Week(int(year), int(week))
                start_date = w.monday()
                end_date = start_date + datetime.timedelta(days=11)
                date_ranges.append([start_date, end_date])
    elif (time_interval == 'monthly'):
        # get all the month files and sort the dates
        months = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_month/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                month, f = e.split('.')
                months.append(month)
        sorted_months = sorted(months)

        row_count = 0
        for year_month in sorted_months:
            year, month = year_month.split('-')
            year = int(year)
            month = int(month.lstrip("0"))
            if (int(year) in [2015, 2016, 2017]):
                a, num_days = calendar.monthrange(year, month)
                start_date = datetime.date(year, month, 1)
                end_date = datetime.date(year, month, num_days)
                date_ranges.append([start_date, end_date])

    elif (time_interval == 'seasonal'):
        # get all the seasonal files and sort the dates
        quarters = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_season/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                quarter, f = e.split('.')
                quarters.append(quarter)
        sorted_quarters = sorted(quarters)

        for quarter in sorted_quarters:
            first_year_month = quarter[0:7]
            year, month = first_year_month.split('-')
            year = int(year)
            month = int(month.lstrip("0"))
            second_month, third_month = month + 1, month + 2
            if (int(year) in [2015, 2016, 2017]):
                a, num_days_first = calendar.monthrange(year, month)
                a, num_days_second = calendar.monthrange(year, second_month)
                a, num_days_third = calendar.monthrange(year, third_month)
                total_num_days = num_days_first + num_days_second + num_days_third
                start_date = datetime.date(year, month, 1)
                end_date = datetime.date(year, third_month, num_days_third)
                date_ranges.append([start_date, end_date])


    return date_ranges

def regime_and_switch(df,futures,features,lags,time_interval,switch_date,target,model_type,components,partition):
    """
    Function to grab date ranges associated with label assignments. Returns actual label values for dates prior to date
    provided in switch_date parameter, and predicted labels after that.
    :param df: Starting Dataframe to read from.
    :param futures: futures[0] is the pair of futures we want to predict on. futures[1] is a list of additional futures
    to generate features from.
    :param features: List of features to include in the model.
    :param lags: A list of lags for the features and futures pairs.
    :param time_interval: Time interval
    :param switch_date: Date to change from actual to predicted values. All dates prior will be used for training, all
    dates after will be predicted.
    :param target: target[0] is correlations or correlation_differences. target[1] specs 'this' or 'next' time interval.
    :param model_type: Type of classification model. Current acceptable values are 'NaiveBayes', 'RadialBasisSVM',
    'RandomForest', 'GradidentBoost', 'DecisionTree'
    :param components: Number of components to break the target range into.
    :param partition: Method for partitioning the range of values. If 'uniform', range is (-1,1) for correlations
    and (-2,2) for correlation changes. If 'uniform_min_max', range is (min(correlations), max(correlations)) broken up
    into |max(correlations) - min(correlations)|/components intervals. If 'match', range is partitioned to try to assign
    roughly the same number of labels to each group.
    :return: A nested dictionary with actual and predicted label assignments, and a the dates which receive this
    assignment. For example, if components=3, the dictionary might look like:

    regime_labels['actual'] =
        0: [[start_date1, end_date1], [start_date2, end_date2], ...]
        1: [[start_date3, end_date3], ...]
        2: [[start_date4, end_date4], [start_date5, end_date5], ...]
    regime_labels['predicted'] =
        0: [[start_date6, end_date6], [start_date7, end_date7], ...]
        1: []
        2: [[start_date8, end_date8], [start_date9, end_date9], ... ]

    Note: Some values may be empty if no dates contain that assignment!
    """

    data = prepare_data(df, futures, features, lags, time_interval, target)

    if (target[0] == "correlations"):
        target_name = time_interval + '_' + 'corr_' + futures[0][0] + '-' + futures[0][1]
    elif (target[0] == 'correlation_differences'):
        target_name = time_interval + '_' + 'corr_diff_' + futures[0][0] + '-' + futures[0][1]

    if (partition == 'uniform'):
        min_target_val = None
        max_target_val = None
        range_list = None
    elif (partition == 'uniform_min_max'):
        min_target_val = np.min(data[target_name].values)
        max_target_val = np.max(data[target_name].values)
        range_list = None
    elif (partition == 'match_distribution'):
        min_target_val = None
        max_target_val = None

        target_items = data[target_name].values
        num_target_items = len(data[target_name].values)
        num_items_per_partition = round(num_target_items / components)
        count = 0
        range_list = []
        for i in range(len(np.sort(target_items))):
            if (count == 0 and len(range_list) < components):
                range_list.append(np.sort(target_items)[i])
            count += 1
            if (count == num_items_per_partition):
                count = 0

    if (target[1] == 'this'):
        data['class_labels'] = data[target_name].apply(range_helper, target=target, components=components, partition=partition, min=min_target_val, max=max_target_val, range_list=range_list)
    elif (target[1] == 'next'):
        data['class_labels'] = np.nan
        for i in range(data.shape[0]):
            num_days = num_days_helper(data.iloc[i]['Timestamp'], time_interval)
            if (i + num_days < data.shape[0]):
                value = data.iloc[i + num_days][target_name]
                data.iloc[i, data.columns.get_loc('class_labels')] = range_helper(value, target, components, partition=partition, min=min_target_val, max=max_target_val, range_list=range_list)

    # we also want to apply these label assignments to the unmodified dataframe to get all the dates
    if (target[1] == 'this'):
        df['class_labels'] = df[target_name].apply(range_helper, target=target, components=components, partition=partition, min=min_target_val, max=max_target_val, range_list=range_list)
    elif (target[1] == 'next'):
        df['class_labels'] = np.nan
        for i in range(df.shape[0]):
            num_days = num_days_helper(df.iloc[i]['Timestamp'], time_interval)
            if (i + num_days < df.shape[0]):
                value = df.iloc[i + num_days][target_name]
                df.iloc[i, df.columns.get_loc('class_labels')] = range_helper(value, target, components, partition=partition, min=min_target_val, max=max_target_val, range_list=range_list)

    data = data.dropna()

    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True).dt.tz_localize(tz=None)  # timezone de-localization is necessary for interpreting timestamps

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True).dt.tz_localize(tz=None)  # timezone de-localization is necessary for interpreting timestamps

    train_dates = (data['Timestamp'] < pd.Timestamp(switch_date))
    test_dates = (data['Timestamp'] >= pd.Timestamp(switch_date))

    train_data = data.loc[train_dates]
    test_data = data.loc[test_dates]

    X = train_data.drop(columns=['Timestamp', target_name, 'class_labels']).values
    y = train_data['class_labels']

    if (model_type == 'NaiveBayes'):
        model = GaussianNB()
    elif (model_type == 'RadialBasisSVM'):
        model = SVC(kernel='rbf')
    elif (model_type == 'PolynomialSVM'):
        model = SVC(kernel='poly')
    elif (model_type == 'SigmoidSVM'):
        model = SVC(kernel='sigmoid')
    elif (model_type == 'RandomForest'):
        model = RandomForestClassifier(random_state=0)
    elif (model_type == 'GradientBoost'):
        model = GradientBoostingClassifier(random_state=0)
    elif (model_type == 'DecisionTree'):
        model = DecisionTreeClassifier()

    model.fit(X, y)

    test_data_with_timestamps = test_data.copy(deep=True) # need a copy with timestamps

    test_data = test_data.drop(columns=['Timestamp', target_name, 'class_labels']).values
    predictions = model.predict(test_data)

    test_data_with_timestamps['pred_label'] = predictions

    date_ranges = get_date_ranges(time_interval)

    regime_labels = {'actual': {},
                     'predicted': {}}

    for i in range(components):
        regime_labels['actual'][i] = []
        regime_labels['predicted'][i] = []

    for start_date,end_date in date_ranges:
        if(pd.Timestamp(end_date) <= pd.Timestamp(switch_date)):
            mask = (df['Timestamp'] >= pd.Timestamp(start_date)) & (df['Timestamp'] <= pd.Timestamp(end_date))
            labels = df.loc[mask]['class_labels'].values
            most_common_label = stats.mode(labels)[0][0]
            regime_labels['actual'][most_common_label].append([start_date,end_date])
        else:
            mask = (test_data_with_timestamps['Timestamp'] >= pd.Timestamp(start_date)) & (test_data_with_timestamps['Timestamp'] <= pd.Timestamp(end_date))
            labels = test_data_with_timestamps.loc[mask]['pred_label'].values
            most_common_label = stats.mode(labels)[0][0]
            regime_labels['predicted'][most_common_label].append([start_date, end_date])

    return regime_labels

def ensemble_feature_selection(df, futures, all_features, time_interval, targets, model_types, n_components_list, partitions):
    """
    Uses multiple classification models and hyperparameter variations to select top features using scikit-learn RFE.
    :param df: Starting dataframe generated using the relevant hyperparameters.
    :param futures: futures[0] is the pair of futures we want to predict on. futures[1] is a list of additional futures
    to generate features from.
    :param all_features: List of features.
    :param time_interval: Relevant time interval.
    :param targets: target[0] is correlations or correlation_differences. target[1] specs 'this' or 'next' time interval.
    :param model_types: List of model types to use for feature selection.
    :param n_components_list: List of component numbers to vary and test.
    :param partitions: List of partitioning methods to vary and test.
    :return: A dictionary of features and the number of times they were included as a top-10 feature across the runs.
    """

    features_dict = {}
    for model_type in model_types:
        for corr_target in targets[0]:
            for time_target in targets[1]:
                for n_components in n_components_list:
                    for partition in partitions:
                        accuracy, features = generate_classification_model(df, futures, all_features, time_interval, [corr_target, time_target], model_type, n_components, partition)
                        for feature in features:
                            if(feature not in features_dict):
                                features_dict[feature] = 1
                            else:
                                features_dict[feature] += 1
    return features_dict
