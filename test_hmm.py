from Modeling.HMM_all_futures_features import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics

"""
TEST Modeling/hmm_all_futures_features.py
"""

no_of_weeks = 20
model = CorrelationPredictorAll()
model.fit()

predictions = model.predict_future_correlations(no_of_weeks)
original = model.ES_corr_test.iloc[30:, 0:325]
predictions = predictions.iloc[:, 0:325]

weeks = np.linspace(0, no_of_weeks, no_of_weeks)
'''
os.makedirs('../Desktop/figure/')
for i in range(325):
    #print("True Correlation changes are:")
    #print(original.iloc[:,i])
    #print("Predicted Correlation changes are:")
    #print(predictions.iloc[:,i])
    fig = plt.figure()
    original_corr = original.iloc[:,i]
    plt.plot(weeks, original_corr, color = "blue", label = "True correlation changes")
    plt.plot(weeks, predictions.iloc[:,i], color = "red", label = "Predicted correlation changes")
    names = original_corr.name
    fig.savefig('../Desktop/figure/%s.png' %names)
    plt.close()
print("saved the 325 generated correlation prediction figure to the Desktop/figure directory")
'''
mean_squared_error_list = []
mean_absolute_error_list = []
direction_accuracy_list = []

for i in range(no_of_weeks):
    true_changes = np.array(original.iloc[i,:])
    predictions1 = np.array(predictions.iloc[i,:])

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
plt.plot(x_axis,mean_squared_error_list)
plt.xlabel("Weeks")
plt.ylabel("Mean_squared_error")
plt.show()
