# importing required libraries
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
 
# read the train and test dataset
train_data = pd.read_csv('./NH3_MARVELV18.states.csv')
test_data = pd.read_csv('./NH3_MARVELV18.states.csv')
 
# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)
 
# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['Energy'],axis=1)
train_y = train_data['Energy']
 
# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Energy'],axis=1)
test_y = test_data['Energy']

model = xgb.XGBRegressor(n_estimators=2000, max_depth=50, learning_rate=0.05, 
                         num_parallel_tree=40, n_jobs=-2, silent=False, 
                         objective='reg:squarederror',
                         verbosity=3, booster='dart', 
                         min_child_weight=2)
 
# fit the model with the training data
model.fit(train_x,train_y)
 
 
# predict the target on the train dataset
predict_train = model.predict(train_x)
print('\nTarget on train data',predict_train) 
 
fig = plt.figure() 
plt.scatter(train_y, train_y - predict_train, c='r')
plt.show()

# # Accuray Score on train dataset
# accuracy_train = accuracy_score(train_y,predict_train)
# print('\naccuracy_score on train dataset : ', accuracy_train)
 
# # predict the target on the test dataset
# predict_test = model.predict(test_x)
# print('\nTarget on test data',predict_test) 
 
# # Accuracy Score on test dataset
# accuracy_test = accuracy_score(test_y,predict_test)
# print('\naccuracy_score on test dataset : ', accuracy_test)