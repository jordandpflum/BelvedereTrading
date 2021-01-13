import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics

def test_file(file):
    """
    Test the functions in file supplied by the user.
    :param file: The file to test.
    :return: N/A
    """
    if(file == 'read_data'):
        print("TEST: Data_Master/read_data.py")

        all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC', 'LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        futures = all_futures_names[0:2]

        print("Test: get_x_df")
        x_df = get_x_df(futures)
        print(x_df)

        print("Test: get_y_df")
        y_df = get_y_df(futures, time_intervals='biweekly', difference_columns=True)
        print(y_df)

        print("Test: merged_df_reduced")
        merged_df_reduced = merge_df_reduce(futures, time_intervals='weekly')
        print(merged_df_reduced)

        # Long run time, works but leaving commented
        # print("Test: merged_df_expanded")
        # merged_df_expanded = merge_df_expand(futures, time_intervals='monthly')
        # print(merged_df_expanded)
    elif(file == 'feature_engineering'):
        print("TEST: Data_Master/engineer_features.py")

        all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC', 'LE',
                             'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        futures = all_futures_names[0:2]

        x_df = get_x_df(futures)

        features = list(marketFeaturefunctions.keys())

        print('Test: engineer_features')
        x_df = engineer_features(x_df, futures, features)

        lags = [1,3,5]

        futures_and_features = []
        for future in futures:
            for feature in features:
                futures_and_features.append(future + '_' + feature)

        print('Test: engineer_lagged_features')
        x_df, new_features = engineer_lagged_features(x_df, futures_and_features, lags)

        print(x_df)
        print(new_features)

    elif(file == 'polar_correlation_generator'):
        print("TEST: Data_Exploration/polar_correlation_generator.py")

        all_futures_names = ['ES', 'YM', 'NQ', 'NG', 'HO', 'RB', 'ZN', 'ZB', 'ZF', 'CL', 'BZ', 'ZT', 'HE', 'ZC', 'LE', 'ZW', 'ZS', 'KE', 'GF', 'ZM', 'ZL', 'GE', 'GC', 'HG', 'SI', 'PL']
        futures = all_futures_names[0:2]
        y_df = get_y_df(futures, time_intervals='weekly')

        print("Test: generate_polar_vis")
        generate_polar_vis(y_df, futures, 'correlations', 'weekly', connect_first_and_last=True)
    elif(file == 'feature_selection_exploration'):
        print("TEST: MarketFeatures/feature_selection_exploration")

        categories = {'Soybeans': ['ZL', 'ZM', 'ZS'],
                      'Livestock': ['HE', 'LE', 'GF'],
                      'Energy': ['CL', 'HO', 'NG'],
                      'Agriculture': ['ZC', 'ZS', 'ZW'],
                      'Metals': ['GC', 'PL', 'SI']}

        # Features w/o MACD, MACDEXT, and MACDFIX (T/F)
        features_names = ['ADX', 'ADXR', 'APO', 'AROON_UP', 'AROON_DOWN', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
                          'MFI', 'MINUS_DI', 'MOM', 'PLUS_DI', 'PLUS_DM',
                          'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH_k', 'STOCH_d', 'STOCH_Fast_k',
                          'STOCH_Fast_d', 'STOCHRSI_k', 'STOCHRSI_d', 'TRIX', 'ULTOSC', 'WILLR', 'OBV', 'CHAIKIN'
                          ]

        futures_names = categories['Soybeans']
        merged_df = merge_df_reduce(futures_names, 'weekly')

        merged_df.dropna(inplace=True)
        merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
        merged_df = merged_df.set_index('Timestamp')

        print("Test: feature_selection_slopes")
        slope_df = feature_selection_slopes(merged_df, futures_names, features_names, normalize=True)

        print("Test: feature_selection_slopes_visual_heatmap")
        heatmap, r = feature_selection_slopes_visual_heatmap(slope_df)

        print('Test: feature_selection_slopes_expansive')
        slope_df = feature_selection_slopes_expansive(merged_df, futures_names, features_names, lagged_features=[],
                                                      lagged_features_incl=False, normalize=True)

        heatmap, r = feature_selection_slopes_visual_heatmap(slope_df)
    elif(file == 'select_top_features'):
        print("TEST: MarketFeatures/select_top_features")

        categories = {'Soybeans': ['ZL', 'ZM', 'ZS'],
                      'Livestock': ['HE', 'LE', 'GF'],
                      'Energy': ['CL', 'HO', 'NG'],
                      'Agriculture': ['ZC', 'ZS', 'ZW'],
                      'Metals': ['GC', 'PL', 'SI']}

        # Features w/o MACD, MACDEXT, and MACDFIX (T/F)
        features_names = ['ADX', 'ADXR', 'APO', 'AROON_UP', 'AROON_DOWN', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
                          'MFI', 'MINUS_DI', 'MOM', 'PLUS_DI', 'PLUS_DM',
                          'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH_k', 'STOCH_d', 'STOCH_Fast_k',
                          'STOCH_Fast_d', 'STOCHRSI_k', 'STOCHRSI_d', 'TRIX', 'ULTOSC', 'WILLR', 'OBV', 'CHAIKIN'
                          ]

        futures_names = categories['Soybeans']
        merged_df = merge_df_reduce(futures_names, 'weekly')

        merged_df.dropna(inplace=True)
        merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
        merged_df = merged_df.set_index('Timestamp')

        print("Test: feature_selection_slopes")
        slope_df = feature_selection_slopes(merged_df, futures_names, features_names, normalize=True)

        num_features = 10
        future_pair_to_analyze = 'ZL-ZM'
        print("Test: select_top_features")
        slope_df = feature_selection_slopes_expansive(merged_df, futures_names, features_names, lagged_features=[],
                                                      lagged_features_incl=False, normalize=True)
        top_features_dictionary = select_top_features(slope_df, future_pair_to_analyze, num_features)

        # Print out top n features with corresponding slopes
        print('Dictionary with ' + str(num_features) + ' highest values:')
        print("Keys: Values")
        for i in top_features_dictionary:
            print(i[0], " :", i[1], " ")

        slope_df = feature_selection_slopes_expansive(merged_df, futures_names, features_names, lagged_features=[],
                                                      lagged_features_incl=False, normalize=True)
        # Top n features for ZL-ZM
        num_features = 10
        future_pair_to_analyze = 'ZL-ZM'
        top_features_dictionary = select_top_features(slope_df, future_pair_to_analyze, num_features)

        # Print out top n features with corresponding slopes
        print('Dictionary with ' + str(num_features) + ' highest values:')
        print("Keys: Values")
        for i in top_features_dictionary:
            print(i[0], " :", i[1], " ")
        # Engineer_lagged_features
        # Get Features to Lag
        features_to_lag_names = []
        for i in top_features_dictionary:
            features_to_lag_names.append(i[0])

        print("Test: select_top_features with lags")

        slope_df = feature_selection_slopes_expansive(merged_df, futures_names, features_names, lagged_features=[],
                                                      lagged_features_incl=False, normalize=True)
        # Top n features for ZL-ZM
        num_features = 10
        future_pair_to_analyze = 'ZL-ZM'
        top_features_dictionary = select_top_features(slope_df, future_pair_to_analyze, num_features)

        # Print out top n features with corresponding slopes
        print('Dictionary with ' + str(num_features) + ' highest values:')
        print("Keys: Values")
        for i in top_features_dictionary:
            print(i[0], " :", i[1], " ")
        # Engineer_lagged_features
        # Get Features to Lag
        features_to_lag_names = []
        for i in top_features_dictionary:
            features_to_lag_names.append(i[0])

        # Set Lags
        lags = [1, 2, 3, 4, 5]

        # Create new df with lagged features
        merged_df, added_features_names = engineer_lagged_features(merged_df, features_to_lag_names, lags)
        merged_df.dropna(inplace=True)

        # Re-create Slope DF with lagged features
        # Slope df for selecting top features
        slope_df = feature_selection_slopes_expansive(merged_df, futures_names, features_names,
                                                      lagged_features=added_features_names,
                                                      lagged_features_incl=True, normalize=True)

        # Top n features for ZL-ZM
        top_features_dictionary = select_top_features(slope_df, future_pair_to_analyze, num_features)

        # Print out top n features with corresponding slopes

        print('Dictionary with ' + str(num_features) + ' highest values (after lagged features introduced):')
        print("Keys: Values")
        for i in top_features_dictionary:
            print(i[0], " :", i[1], " ")
    elif(file == 'AltModels'):
        print("TEST: AltModels")

        futures = [['ZS', 'ZM'], []]
        all_features = list(marketFeaturefunctions.keys())

        time_interval = 'weekly'
        lags = []
        df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features, lags)

        # use raw data in modeling
        all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features


        print("Test: generate_mixture_model")
        generate_mixture_model(df, futures, all_features, lags, time_interval, target=['correlations', 'this'], components=3)

        print("Test: generate_classification_model")
        accuracy, features = generate_classification_model(df, futures, all_features, lags, time_interval,
                                                           target=['correlations', 'this'], model_type='GradientBoost',
                                                           components=3, partition='match_distribution')

        print(accuracy)
        print(features)

        print("Test: regime_and_switch")
        switch_date = '2017-09-04'
        regimes = regime_and_switch(df, futures, all_features, lags, time_interval, switch_date,
                                    target=['correlations', 'this'], model_type='GradientBoost', components=8,
                                    partition='match_distribution')

        print(regimes)
    elif(file == 'HMM_all_futures_features'):
        print("TEST: HMM_all_futures_features")
        no_of_weeks = 20

        print('Test: CorrelationPredictorAll()')
        model = CorrelationPredictorAll()
        model.fit()

        print('Test: predict_future_correlations')
        predictions = model.predict_future_correlations(no_of_weeks)
        original = model.ES_corr_test.iloc[30:, 0:325]
        predictions = predictions.iloc[:, 0:325]

        weeks = np.linspace(0, no_of_weeks, no_of_weeks)

        mean_squared_error_list = []
        mean_absolute_error_list = []
        direction_accuracy_list = []

        for i in range(no_of_weeks):
            true_changes = np.array(original.iloc[i, :])
            predictions1 = np.array(predictions.iloc[i, :])

            binary_true_changes = np.copy(true_changes)
            binary_true_changes[binary_true_changes < 0] = 0
            binary_true_changes[binary_true_changes != 0] = 1
            binary_predicted_changes = np.copy(predictions1)
            binary_predicted_changes[binary_predicted_changes < 0] = 0
            binary_predicted_changes[binary_predicted_changes != 0] = 1

            direction_accuracy_res = np.mean(binary_true_changes == binary_predicted_changes) * 100
            direction_accuracy_list.append(direction_accuracy_res.item())

            mean_squared_error_res = mean_squared_error(true_changes, predictions1)
            mean_squared_error_list.append(mean_squared_error_res)

            mean_absolute_error_res = mean_absolute_error(true_changes, predictions1)
            mean_absolute_error_list.append(mean_absolute_error_res)

        print("Mean Squared Error of this Prediction is:")
        print(statistics.mean(mean_squared_error_list))

        print("Mean Absolute Error of this Prediction is:")
        print(statistics.mean(mean_absolute_error_list))

        print("Direction Accuracy is:")
        print(str(statistics.mean(direction_accuracy_list)) + "%")

        x_axis = np.arange(no_of_weeks)

        plt.figure()
        plt.plot(x_axis, mean_squared_error_list)
        plt.xlabel("Weeks")
        plt.ylabel("Mean_squared_error")
        plt.show()
    elif(file == 'data_split'):
        print("TEST: data_split")

        futures = ['ZS', 'ZM']
        # all_features = list(marketFeaturefunctions.keys())
        time_interval = 'weekly'
        df = merge_df_reduce(futures, time_interval)
        print(df.head())

        df = target_generator(df, 'weekly_corr_diff_ZS-ZM', [5, 10])
        print(df)
        df.tail()

        x = df.loc[:, ['weekly_corr_diff_ZS-ZM']]
        y = df.loc[:, ['weekly_corr_diff_ZS-ZM + 5 weeks', 'weekly_corr_diff_ZS-ZM + 10 weeks']]
        # reserve 10% of data for testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.1)

        print(x_train)
        print(x_test)
        print(y_train)
        print(y_test)
    elif(file == 'Lasso_classification'):
        print('TEST: Lasso_classification')

        print('Test: create_regime_dataframe')
        futures = ['ZM', 'ZS']
        pair_plus_futures = []
        all_features = list(marketFeaturefunctions.keys())
        top_features = ['AROONOSC', 'PPO', 'STOCH_k', 'AROON_UP']  # top futures identified by HMM (use only for HMM)
        no_of_states = 5
        model1_mse = []
        model2_mse = []
        regimes = []
        reg = 0.01  # Regularization Parameter
        model = 'GradientBoost'
        partition = 'uniform_min_max'
        time_interval = 'monthly'
        switch_date = '2016-09-04'
        train_state_dataframes, test_set_dataframes = create_regime_dataframes(no_of_states, futures, pair_plus_futures,
                                                                               all_features, model, time_interval,
                                                                               switch_date, partition)
        print(train_state_dataframes)
        print(test_set_dataframes)

        print('Test: all_data_df')
        regime = 2
        df_model1 = all_data_df(no_of_states, futures, pair_plus_futures, regime, all_features, model, time_interval,
                                switch_date, partition, subset="True")

        print('Test: get_x_y')
        df_model1 = df_model1.drop(['Timestamp'], axis=1)
        x_train_model1, y_train_model1 = get_x_y(df_model1, 3)
        print(x_train_model1)
        print(y_train_model1)

        print('Test: Lasso_Classification')
        mae1, mae2, mse1, mse2, score1, score2 = Lasso_Classification(futures, pair_plus_futures, all_features,
                                                                      no_of_states, regime, model, reg, time_interval,
                                                                      switch_date, partition)
        print(mae1)
        print(mae2)
        print(mse1)
        print(mse2)
        print(score1)
        print(score2)

        print('Test: compare_future_add')
        pair_plus = ['ZL', 'ZC', 'ZW']  # adding some more futures to see how the model improves
        score1, score2, mse1, mse2, mae1, mae2 = compare_future_add(no_of_states, futures, pair_plus, all_features,
                                                                    model, regime, time_interval, switch_date, reg,
                                                                    partition)
        print(score1)
        print(score2)
        print(mse1)
        print(mse2)
        print(mae1)
        print(mae2)
    elif(file=='lasso_hmm'):
        print('TEST: lasso_hmm')

        print('Test: Lasso_HMM')
        futures = ['ZM', 'ZS']
        pair_plus_futures = []
        all_features = list(marketFeaturefunctions.keys())
        top_features = ['AROONOSC', 'PPO', 'STOCH_k', 'AROON_UP']  # top futures identified by HMM (use only for HMM)
        no_of_states = 1
        model1_mse = []
        model2_mse = []
        regimes = []
        reg = 0.01  # Regularization Parameter
        time_interval = 'monthly'
        regime = 0

        mae1,mae2,mse1,mse2,score1,score2 = Lasso_HMM(futures, all_features, no_of_states, regime, reg, time_interval)

        print(mae1)
        print(mae2)
        print(mse2)
        print(mse2)
        print(score1)
        print(score2)

from Data_Master.read_data import *
test_file('read_data')

from Data_Master.feature_engineering import *
test_file('feature_engineering')

from Data_Exploration.polar_correlation_generator import *
test_file('polar_correlation_generator')

from MarketFeatures.feature_selection_exploration import *
test_file('feature_selection_exploration')

from MarketFeatures.select_top_features import *
test_file('select_top_features')

from Modeling.AltModels import *
test_file('AltModels')

from Modeling.HMM_all_futures_features import *
test_file('HMM_all_futures_features')

from Data_Master.data_split import *
test_file('data_split')

from Modeling.Lasso_classification import *
test_file('Lasso_classification')

from Modeling.lasso_hmm import *
test_file('lasso_hmm')