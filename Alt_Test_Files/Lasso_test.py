from Modeling.Lasso_classification import *
from Modeling.lasso_hmm import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Code for Regime vs all data

futures = ['ZM','ZS']
pair_plus_futures = []
all_features = list(marketFeaturefunctions.keys())
top_features = ['AROONOSC', 'PPO', 'STOCH_k','AROON_UP'] # top futures identified by HMM (use only for HMM)
no_of_states = 5
model1_mse = []
model2_mse = []
regimes = []
reg = 0.01 # Regularization Parameter
model = 'RandomForest'
partition = 'uniform_min_max'
time_interval = 'monthly'
switch_date = '2016-09-04'
for regime in range(no_of_states):
    try:
        if model == 'HMM':
            mae1,mae2,mse1,mse2,score1,score2 = Lasso_HMM(futures,all_features,no_of_states,regime,reg,time_interval)
            model1_mse.append(mse1)
            model2_mse.append(mse2) 
            regimes.append(regime)
            MSE_decrease = ((mse1 - mse2)/mse1)*100
            MAE_decrease = ((mae1 - mae2)/mae1)*100
            print("R2 score with model trained with entire training data: " + str(score1))
            print("R2 score with model trained with regime" + str(regime + 1) +  " data: " + str(score2))
            print("MAE with model trained with entire training data: " + str(mae1))
            print("MAE with model trained with regime" + str(regime + 1) +  " data: " + str(mae2))
            print("MSE with model trained with entire training data: " + str(mse1))
            print("MSE with model trained with regime" + str(regime + 1) +  " data: " + str(mse2))
            print("Percent decrease in MAE: " + str(MAE_decrease) + "%")
            print("Percent decrease in MSE: " + str(MSE_decrease) + "%")
        else:
            mae1,mae2,mse1,mse2,score1,score2 = Lasso_Classification(futures,pair_plus_futures,all_features,no_of_states,regime,model,reg,time_interval,switch_date,partition)
            model1_mse.append(mse1)
            model2_mse.append(mse2) 
            regimes.append(regime)
            MSE_decrease = ((mse1 - mse2)/mse1)*100
            MAE_decrease = ((mae1 - mae2)/mae1)*100
            print("R2 score with model trained with entire training data: " + str(score1))
            print("R2 score with model trained with regime" + str(regime + 1) +  " data: " + str(score2))
            print("MAE with model trained with entire training data: " + str(mae1))
            print("MAE with model trained with regime" + str(regime + 1) +  " data: " + str(mae2))
            print("MSE with model trained with entire training data: " + str(mse1))
            print("MSE with model trained with regime" + str(regime + 1) +  " data: " + str(mse2))
            print("Percent decrease in MAE: " + str(MAE_decrease) + "%")
            print("Percent decrease in MSE: " + str(MSE_decrease) + "%")
    except ValueError:
        print("No data for Regime" + str(regime + 1))

# Plotting MSE for different regimes
fontP = FontProperties()
fontP.set_size('small')

barWidth = 0.25

bars3 = model1_mse
bars4 = model2_mse

r3 = np.arange(len(bars3))
r4 = [x + barWidth for x in r3]

plt.bar(r3, bars3, color='#7f6d5f', width=barWidth, edgecolor='white', label='MSE for model trained on entire training data')
plt.bar(r4, bars4, color='#557f2d', width=barWidth, edgecolor='white', label='MSE for model trained on Regime Data')

plt.xlabel('Regime', fontweight='bold')
Regimes = ['Regime' + str(i + 1) for i in regimes]
plt.xticks([r + barWidth for r in range(len(bars3))], Regimes)

plt.legend(prop=fontP,bbox_to_anchor=(0.5,1.03))
plt.show()


# Code to test the effect of adding more futures 
model = 'RandomForest'
pair_plus_futures = [['ZL'],['ZL','ZC'],['ZL','ZC','ZW']] #Progressively adding futures to see how the model improves
regime = 2
try:
    score1,score2,mse1,mse2,mae1,mae2 = compare_future_add(no_of_states,futures,[],all_features,model,regime,time_interval,switch_date,reg,partition)
    mse_list = [mse1]
    for pair_plus in (pair_plus_futures):
        score1,score2,mse1,mse2,mae1,mae2 = compare_future_add(no_of_states,futures,pair_plus,all_features,model,regime,time_interval,switch_date,reg,partition)
        mse_list.append(mse2)
        MSE_decrease = ((mse1 - mse2)/mse1)*100
        MAE_decrease = ((mae1 - mae2)/mae1)*100
        print("R2 score with model trained and tested with just ZM and ZS: " + str(score1))
        print("R2 score with model trained and tested with ZM,ZS plus correlated futures: " + str(score2))
        print("MSE with model trained and tested with ZM and ZS: " + str(mse1))
        print("MSE with model trained and tested with ZM,ZS plus correlated futures: " + str(mse2))
        print("Percent decrease in MSE: " + str(MSE_decrease) + "%")
        print("MAE with model trained and tested with ZM and ZS: " + str(mae1))
        print("MAE with model trained and tested with ZM,ZS plus correlated futures: " + str(mae2))
        print("Percent decrease in MAE: " + str(MAE_decrease) + "%")
    
    #Plotting Mean Squared Error as futures are added
    objects = ('ZS,ZM', 'ZS,ZM,ZL', 'ZS,ZM,ZL,ZC','ZS,ZM,ZL,ZC,ZW')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, mse_list, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('MSE')
    plt.title('MSE as correlated futures are added in regime ' + str(regime + 1))
    plt.show()

except ValueError:
    print("No data")













