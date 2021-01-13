# Test Example (Select Top Features)
# from Data_Master.read_data_jordan import *
from Data_Master.read_data import *
from MarketFeatures.feature_selection_exploration import *
from MarketFeatures.select_top_features import *
from Data_Master.feature_engineering import *
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt

def testing_features(merged_df, futures_names, features_names,
                     heatmap_1=False, heatmap_2=False, top_features=False,  top_features_with_lag=False):
    # Uncomment if creating scatterplot visual
    #merged_df = merged_df.resample(rule='W').mean()

    if heatmap_1:
        print('============')
        print('Testing: heatmap_1')
        # Create Slope DF
        slope_df = feature_selection_slopes(merged_df, futures_names, features_names, normalize=True)

        # Heatmap Visual
        heatmap, r = feature_selection_slopes_visual_heatmap(slope_df)

    if heatmap_2:
        print('============')
        print('Testing: heatmap_2')
        # Slope df for selecting top features
        slope_df = feature_selection_slopes_expansive(merged_df, futures_names, features_names, lagged_features=[],
                                                      lagged_features_incl=False, normalize=True)

        # Heatmap Visual
        heatmap, r = feature_selection_slopes_visual_heatmap(slope_df)

    if top_features:
        print('============')
        print('Testing: top_features')
        # Slope df for selecting top features
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

    if top_features_with_lag:
        print('============')
        print('Testing: top_features_with_lag')
        # Slope df for selecting top features
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

    print('============')
    return print("Finished Testing")

# Define Data
categories = {'Soybeans': ['ZL','ZM','ZS'],
              'Livestock': ['HE', 'LE', 'GF'],
              'Energy': ['CL', 'HO', 'NG'],
              'Agriculture': ['ZC', 'ZS', 'ZW'],
              'Metals': ['GC', 'PL', 'SI']}

# Features w/o MACD, MACDEXT, and MACDFIX (T/F)
features_names = ['ADX', 'ADXR', 'APO', 'AROON_UP', 'AROON_DOWN', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
                      'MFI', 'MINUS_DI', 'MOM', 'PLUS_DI', 'PLUS_DM',
                      'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH_k', 'STOCH_d', 'STOCH_Fast_k',
                      'STOCH_Fast_d',  'STOCHRSI_k', 'STOCHRSI_d', 'TRIX', 'ULTOSC', 'WILLR', 'OBV', 'CHAIKIN'
                      ]

# Load Data
futures_names = categories['Soybeans']
merged_df = merge_df_reduce(futures_names, 'weekly')

# Format Data
merged_df.dropna(inplace=True)
merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], utc=True).dt.tz_localize(tz=None)
merged_df = merged_df.set_index('Timestamp')

# Testing
testing_features(merged_df, futures_names, features_names,
                 heatmap_1=True, heatmap_2=True, top_features=True, top_features_with_lag=True)
