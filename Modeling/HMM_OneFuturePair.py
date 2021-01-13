# HMM correlation change prediction implimentation by sampling from next predicted state

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CorrelationPredictor():
    def __init__(self, test_size=0.33,
                 n_hidden_states=4, n_latency_weeks= 30):
        """
        :param test_size: Fraction of the dataset to kept aside for the test set
        :param n_hidden_states: Number of hidden states
        :param n_latency_weeks: Number of weeks to be used as history for prediction 
        """
 
        self.n_latency_weeks = n_latency_weeks
 
        self.hmm = GaussianHMM(n_components=n_hidden_states,n_iter=1000)
 
        self._split_train_test_data(test_size,"ZS","ZM") # Enter correlation pairs here

    def _split_train_test_data(self, test_size,future1,future2):
        """
        :params future 1 and future2: Future pairs whose correlations we want to predict. For example if we want to predict correlations between ZS and ZM, plug in future1 as "ZS" and future2 as "ZM"
        """
        corr_with_stamp = pd.read_csv("E:\\Belevedere_Spr20-master\\Data_Master\\belvederedata\\corr_diff_dataframe.csv",sep = '\t') # csv file is in Data Master\Belvedere Data Directory
        _train_data_with_stamp, test_data_with_stamp = train_test_split(
        corr_with_stamp, test_size=test_size, shuffle=False)
        _train_data_with_stamp = _train_data_with_stamp.iloc[:,1:652]
        test_data_with_stamp = test_data_with_stamp.iloc[:,1:652]
        # print(_train_data_with_stamp.head())
        _train_data = _train_data_with_stamp.drop(columns = ['Timestamp'])
        test_data = test_data_with_stamp.drop(columns = ['Timestamp'])
        self._train_data = _train_data
        self.test_data = test_data
        self.ES_corr_train = self._train_data["weekly_corr_diff_" + future1 + "-" + future2]
        #print(len(self.ES_corr_train))
        self.ES_corr_test = self.test_data["weekly_corr_diff_" + future1 + "-" + future2]
        #print(len(self.ES_corr_test))
        
    def fit(self):
        # Estimates HMM parameters
        self.hmm.fit(self.ES_corr_train.reshape(-1,1))
    
    def predict(self,df):
        # Finds the most likely state sequence (Vitterbi Algorithm)
        pred = self.hmm.predict(df.reshape(-1,1))
        return pred

    def predict_future_correlations(self,weeks):
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
            mean_vector = np.zeros((len(next_state_probs),))
            for j in range(len(next_state_probs)):
                mean_vector[j] = hmm.means_[state_vector[j]]
            prediction = np.dot(mean_vector,next_state_probs)
            predicted_next_correlations = predicted_next_correlations.append(pd.Series(prediction),ignore_index = True)
        return predicted_next_correlations


future1 = "ZS"
future2 = "ZM"
no_of_weeks = 20
model = CorrelationPredictor()
model.fit()
predictions = model.predict_future_correlations(no_of_weeks)
print("True Correlation changes are:")
print(model.ES_corr_test.iloc[model.n_latency_weeks:model.n_latency_weeks + no_of_weeks])
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
weeks = np.linspace(0,no_of_weeks,no_of_weeks)
plt.plot(weeks,model.ES_corr_test.iloc[model.n_latency_weeks:model.n_latency_weeks + no_of_weeks],color = "blue",label = "True correlation changes")
plt.plot(weeks,predictions,color = "red",label = "Predicted correlation changes")
plt.legend()
plt.xlabel("Weeks")
plt.ylabel("Correlation changes")
plt.title(future1 + "-" + future2)
plt.show()











