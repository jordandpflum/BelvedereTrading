from Data_Master.read_data import *
#from Data_Master.read_data_jordan import *
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

# For testing
from Data_Master.feature_engineering import *
from MarketFeatures.select_top_features import select_top_features


def feature_selection_slopes(df, futures_names, features_names, normalize=False):
    """
    Explore the importance of features by measuring the slope of a particular feature with respect to change of
    correlation for a given time interval

    Parameters:
    futures_names (list): List of strings, where each element represents the name of a future that you would like to
                          generate features for
    features_names (list): List of strings, where each element represents the name of a feature that you would like to
                          generate
    Returns:
    slope_df (pandas dataframe): slope-dataframe, with correlation changes for each desired future appended as columns
                                 and the rows representing a single feature.

    """
    # Create Pairs
    pairs = {}
    for future_name1 in futures_names:
        for future_name2 in futures_names:
            if future_name1 + '-' + future_name2 not in pairs.keys() and future_name2 + '-' + future_name1 not in pairs.keys() and future_name1 != future_name2:
                pairs[future_name1 + '-' + future_name2] = [future_name1, future_name2]

    # Create Column Names
    columns = []
    for pair, pair_futures in pairs.items():
        for future in pair_futures:
            columns.append(future + '--weekly_corr_diff_' + pair)

    # Create Slope Dataframe
    slope_df = pd.DataFrame(columns=columns, index=features_names)
    for pair, pair_futures in pairs.items():
        for future in pair_futures:
            for feature_name in features_names:
                # Get x-values
                if normalize:
                    # Normalize Feature Values
                    max_value = df[future + '_' + feature_name].max()
                    min_value = df[future + '_' + feature_name].min()
                    if max_value == min_value:
                        x_values_scaled = pd.DataFrame(np.zeros((len(df.index), 1)))
                        x = x_values_scaled.iloc[:,0]
                    else:
                        x_values_scaled = (df[future + '_' + feature_name] - min_value) / (max_value - min_value)
                        x = x_values_scaled
                else:
                    x = df[future + '_' + feature_name]
                # Get y-values
                y = df['weekly_corr_diff_' + pair]

                # Calculate Slope and add to slope_df
                b, m = polyfit(x, y, 1)
                slope_df.loc[feature_name, future + '--weekly_corr_diff_' + pair] = m
    return slope_df

def feature_selection_slopes_expansive(df, futures_names, features_names, lagged_features,
                                              lagged_features_incl = False, normalize=True):
    """
    Explore the importance of features by measuring the slope of a particular feature with respect to change of
    correlation for a given time interval.

    Differs from feature_selection_slopes by measuring the slope of all features, regardless of which pair of futures
    is being analyzed.

    Parameters:
    df (pandas df): X-DF of slopes
    futures_names (list): List of strings, where each element represents the name of a future that you would like to
                          generate features for
    features_names (list): List of strings, where each element represents the name of a feature that you would like to
                          generate
    lagged_features (list): List of strings, where each element represents the name of a lagged feature that you would
                            like to generate
    lagged_features_incl (boolean): A boolean, representing whether lagged features are included
    normalize (boolean): A boolean, indicating whether or not to normalize the features

    Returns:
    slope_df (pandas dataframe): slope-dataframe, with correlation changes for each desired future appended as columns
                                 and the rows representing a single feature.

    """
    # Create Pairs
    pairs = {}
    for future_name1 in futures_names:
        for future_name2 in futures_names:
            if future_name1 + '-' + future_name2 not in pairs.keys() and future_name2 + '-' + future_name1 not in pairs.keys() and future_name1 != future_name2:
                pairs[future_name1 + '-' + future_name2] = [future_name1, future_name2]

    # Create Column Names
    columns = []
    for pair, pair_futures in pairs.items():
        columns.append('weekly_corr_diff_' + pair)

    # Create Index Values
    index_values = []
    for future in futures_names:
        for feature in features_names:
            index_values.append(future + '_' + feature)

    if lagged_features_incl:
        for feature in lagged_features:
            index_values.append(feature)

    # Create Slope Dataframe
    slope_df = pd.DataFrame(columns=columns, index=index_values)
    for pair, pair_futures in pairs.items():
        for feature in index_values:
            # Get x-values
            if normalize:
                # Normalize Feature Values
                max_value = df[feature].max()
                min_value = df[feature].min()
                if max_value == min_value:
                    x_values_scaled = pd.DataFrame(np.zeros((len(df.index), 1)))
                    x = x_values_scaled.iloc[:,0]
                else:
                    x_values_scaled = (df[feature] - min_value) / (max_value - min_value)
                    x = x_values_scaled
            else:
                x = df[feature]
            # Get y-values
            y = df['weekly_corr_diff_' + pair]

            # Calculate Slope and add to slope_df
            b, m = polyfit(x, y, 1)
            slope_df.loc[feature, 'weekly_corr_diff_' + pair] = m
    return slope_df

def feature_selection_slopes_visual_heatmap(df):
    """
    Graph the results of feature_selection_slopes on a heat map

    Parameters:
    df: dataframe of slopes generated by feature_selection_slopes

    Returns: heatmap


    """
    # Get Labels
    x_labels = df.columns
    y_labels = df.index

    # Ensure no NAs (replace with 0 if there are)
    df.fillna(0, inplace=True)

    # Covert to array
    df_matrix = df.to_numpy()

    # Take absolute value of slopes for visual purposes
    df_matrix = np.absolute(df_matrix)


    # Plot Figure
    '''
    fig = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(24, 24))
    heatplot = ax.imshow(df_matrix, cmap='BuPu')
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.show()
    '''


    fig = plt.figure(figsize=(12, 12))
    r = sns.heatmap(df_matrix, cmap='BuPu')
    r.set_title("Heatmap of Correlation Differences vs Features (Slopes - Absolute Values)")
    plt.show()

    return fig, r





