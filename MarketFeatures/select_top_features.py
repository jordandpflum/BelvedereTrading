from collections import Counter

def select_top_features(slope_df, interested_future_pair, num_features):
    """
    Select top n features of interested future pair from slope_df

    Parameters:
    slope_df (pandas dataframe): slope-dataframe, with correlation changes for each desired future appended as columns
                                 and the rows representing a single feature.
    interested_future (str): String, which represents the future pair you are interested in finding the top features for
    num_features (int): number of top features you want to return

    Returns:
    top_features (dictionary): a dictionary, where each key is a string representing a top feature and each
                               corresponding value is a real number representing the absolute slope value

    """

    # Create Copy of slope_df
    slope_df_dictionary = {}
    for feature in slope_df.index:
        slope_df_dictionary[feature] = slope_df.loc[feature, 'weekly_corr_diff_' + interested_future_pair]

    k = Counter(slope_df_dictionary)

    # Finding 3 highest values
    top_slopes_dictionary = k.most_common(num_features)

    return top_slopes_dictionary
