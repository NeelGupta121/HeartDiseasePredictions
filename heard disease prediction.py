#!/usr/bin/env python
# coding: utf-8
# importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
# # Data collection and Pre-processing
# loading the dataset to pandas DataFrame
heart_data=pd.read_csv('heart.csv')





# printing first 5 rows of the data
heart_data.head()





#printing last 5 rows of the data set
heart_data.tail()





# number of rows and columns in the dataset
heart_data.shape




# taking some info about the data
heart_data.info()





# checking for any missing value
heart_data.isnull().sum() # no missing value in the data set




# statistical measures about the data
heart_data.describe()




#checking for the distribution of the target variable
heart_data['target'].value_counts()

### the data is evenly distributed


# 0 -> healthy heart
# 
# 1 -> unhealthy heart




# spliting the features and target
x=heart_data.drop(columns='target')
y=heart_data['target']
scaler=StandardScaler()
scaler.fit_transform(x)





print(x)





print(y)





# spliting the dataset into training and testing dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)





#printing rows and columns of x,x_train,x_test
print(x.shape,x_train.shape,x_test.shape)


# # Model  training

# Logistics Regression model




model=LogisticRegression()





#training the model with training dataset
model.fit(x_train,y_train)


# # Model Evaluation

# accuracy score as evaluation metric




# accuracy on training data
x_train_predictions=model.predict(x_train)
x_train_acc=accuracy_score(x_train_predictions,y_train)





print(x_train_acc)


# accuracy of our model for training data is around 85%




# accuracy on test data
x_test_predictions=model.predict(x_test)
x_test_acc=accuracy_score(x_test_predictions,y_test)




print(x_test_acc)


# accuracy of our model for test dataset is around 82%



pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))



