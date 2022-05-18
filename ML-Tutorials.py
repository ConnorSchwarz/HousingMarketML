#%%
from statistics import mean
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy import rand
from sklearn.model_selection import train_test_split 




#%%
#File path 
melbourne_file_path = 'melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data = melbourne_data.dropna(axis = 0) #Drop NA's in data 


melbourne_data.head()

# Stat analysis 
Max_price = max(melbourne_data["Price"])
#Max_price

Newest_House_age =  min(2022 - melbourne_data["YearBuilt"])

#print(Newest_House_age)




# %%
#Predicted Variable 
Response_var = melbourne_data.Price 

#Features / Predictor Variables 
melbourne_features = ["Rooms", "Distance", "YearBuilt", "Landsize","BuildingArea", "Bathroom", "Lattitude"]

Mel_Predictors = melbourne_data[melbourne_features]


# %%

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#Prediction Cell 

#Training and Testing 
Train_predict, vals_predict, train_response, val_response = train_test_split(Mel_Predictors, Response_var, random_state= 0)
#Mel_Predictors.describe()


# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit the Model 
melbourne_model.fit(Train_predict, train_response)

#Model Validation 
from sklearn.metrics import mean_absolute_error

predicted_prices = melbourne_model.predict(vals_predict)
print(mean_absolute_error(val_response, predicted_prices) )# $281,429


# %%
def get_MAE(max_leaf_nodes, Train_predict, vals_predict, train_response, val_response):
    model = DecisionTreeRegressor(max_leaf_nodes= max_leaf_nodes, random_state= 0)
    model.fit(Train_predict, train_response)
    preds_vals = model.predict(vals_predict)
    mae = mean_absolute_error(val_response, preds_vals)
    return(mae)
# %%
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae_loop = get_MAE(max_leaf_nodes, Train_predict, vals_predict, train_response, val_response)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, mae_loop))
# %%

#Final Decision Tree Model with improved MSE from Best Max Leaf Node Count 
final_model = DecisionTreeRegressor(max_leaf_nodes= 50, random_state= 1)
final_model.fit(Mel_Predictors, Response_var)

# %%
#Random Forests 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state= 1) 
forest_model.fit(Train_predict, train_response)
forest_predicts = forest_model.predict(vals_predict)
print(mean_absolute_error(val_response, forest_predicts)) #201162.69
# %%
