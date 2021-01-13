from Data_Master.read_data import *
from Data_Master.feature_engineering import *
from MarketFeatures.feature_selection_exploration import *
from MarketFeatures.select_top_features import *
from Modeling.AltModels import *
from Modeling.Lasso_classification import *
from Modeling.lasso_hmm import *

def classification_model_workflow():
    """
    Demonstrates the steps required to run the a correlation range prediction classification model.
    :return: N/A
    """

    # select the future pair of interest, and additional futures to include in modeling
    futures = [['ZS', 'ZM'], []]
    # futures = [['ZS', 'ZM'],['ZL', 'ZC', 'ZW']] # use this instead for additional future pairs

    # select all possible features to use
    all_features = list(marketFeaturefunctions.keys())

    # select time interval from ['weekly', 'biweekly', 'monthly', 'seasonal']
    time_interval = 'weekly'

    # include lagged versions of all selected features if desired
    lags = []
    # lags = [1,3,5] will generate 1-lag, 3-lag, and 5-lag of the features

    # get the feature-engineered dataframe for modeling
    df = merge_df_reduce(futures[0] + futures[1], time_interval, all_features, lags)

    # include raw data in modeling
    all_features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TRADED_VOLUME'] + all_features

    # target is correlation prediction for current interval
    target = ['correlations', 'this']

    # select model type
    model_type = 'GradientBoost'

    # selected number of components
    components = 3

    # select method of label assignment
    partition = 'match_distribution'

    # run the classification model and receive a dictionary accuries and top selected features (if applicable)
    accuracy, features = generate_classification_model(df,futures,all_features,lags,time_interval,target,model_type,components,partition)

    print(accuracy)
    print("Features: " + str(features))

def Lasso_classification_workflow():
    """
    Demonstrates the steps required to run the state-switching integrated lasso and classification model.
    :return: N/A
    """

    # select the future pair of interest
    futures = ['ZS', 'ZM']

    # add additional futures to generate features for, if desired
    pair_plus_futures = []

    # select all possible features to use
    all_features = list(marketFeaturefunctions.keys())

    # select number of states (components)
    no_of_states = 3

    # select a regime of interest to study
    regime = 1

    # select model type
    model = 'GradientBoost'

    # Regularization Parameter
    reg = 0.01

    # select time interval
    time_interval = 'monthly'

    # select switch date for training (before switch date) and testing (after switch date)
    switch_date = '2016-09-04'

    # select method of label assignment
    partition = 'uniform_min_max'

    # run the model and generate results
    mae1, mae2, mse1, mse2, score1, score2 = Lasso_Classification(futures, pair_plus_futures, all_features,
                                                                  no_of_states, regime, model, reg, time_interval,
                                                                  switch_date, partition)

    MSE_decrease = ((mse1 - mse2) / mse1) * 100
    MAE_decrease = ((mae1 - mae2) / mae1) * 100
    print("R2 score with model trained with entire training data: " + str(score1))
    print("R2 score with model trained with regime" + str(regime + 1) + " data: " + str(score2))
    print("MAE with model trained with entire training data: " + str(mae1))
    print("MAE with model trained with regime" + str(regime + 1) + " data: " + str(mae2))
    print("MSE with model trained with entire training data: " + str(mse1))
    print("MSE with model trained with regime" + str(regime + 1) + " data: " + str(mse2))
    print("Percent decrease in MAE: " + str(MAE_decrease) + "%")
    print("Percent decrease in MSE: " + str(MSE_decrease) + "%")

def Lasso_HMM_workflow():
    """
    Demonstrates the steps required to run the state-switching integrated lasso and HMM model.
    :return: N/A
    """

    # select the future pair of interest
    futures = ['ZS', 'ZM']

    # select all possible features to use
    all_features = list(marketFeaturefunctions.keys())

    # select number of states (components)
    no_of_states = 1 # here we are only selecting 1 hidden state for demonstration purposes

    # select a regime of interest to study
    regime = 0 # there is only 1 state to observe in this case

    # Regularization Parameter
    reg = 0.01

    # select time interval
    time_interval = 'monthly'

    # run the model and generate results
    mae1, mae2, mse1, mse2, score1, score2 = Lasso_HMM(futures, all_features, no_of_states, regime, reg, time_interval)

    MSE_decrease = ((mse1 - mse2) / mse1) * 100
    MAE_decrease = ((mae1 - mae2) / mae1) * 100
    print("R2 score with model trained with entire training data: " + str(score1))
    print("R2 score with model trained with regime" + str(regime + 1) + " data: " + str(score2))
    print("MAE with model trained with entire training data: " + str(mae1))
    print("MAE with model trained with regime" + str(regime + 1) + " data: " + str(mae2))
    print("MSE with model trained with entire training data: " + str(mse1))
    print("MSE with model trained with regime" + str(regime + 1) + " data: " + str(mse2))
    print("Percent decrease in MAE: " + str(MAE_decrease) + "%")
    print("Percent decrease in MSE: " + str(MSE_decrease) + "%")

classification_model_workflow()
Lasso_classification_workflow()
Lasso_HMM_workflow()