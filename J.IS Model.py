import sys
import os

# The pandas library allows you to import the dataset and allows you to examine, manipulate, and clean the dataset
import pandas as pd
# The tensorflow library is neccessary for machine learning but for usually for complex data analysis and deep learning
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
import tensorflow as tf
# The numpy library gives you access to different math and algerba functions and expressions to be used during machine learning
import numpy as np
# The matplotlib is used to creaete plots and graphs displaying the information from the dataset
import matplotlib.pyplot as plt
# The sklearn library (known as scikit-learn) is the library that provides the most simple ways to perform machine learning and access related functions and resources

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
#These are different features that can be used to measure the effectiveness of your model
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted
import seaborn as sns

#This library is used to save and upload trained models
import joblib
cwd = os.getcwd() 
print(cwd)

Shots = pd.read_csv('combined_shots.csv')

print ("This is the Merged DataSet")
print(Shots.head())
print(Shots.tail())

#Creates a Heatmap for all variables
corr = Shots.corr()
plt.figure(figsize=(12,9))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.savefig('heatmap.png')
plt.show()

#Creates chunks to allow you iterate through the dataset and train it in chunks do to its extremely large size
chunk_size = 10000
data_set = pd.read_csv('combined_shots.csv', chunksize =chunk_size)



#Create the model, You should do a logistic regression when the result is either yes or no and there are linear relationships in the data
model = LogisticRegression(max_iter = 100000)


#Allows you to loop through the different parts of dat_set and trains a model
# This dataset trains the intial model of dataset with all variables and is the core one I played around with to determine each variables impact
#Commeneted out since it is the testing model

#for i, chunk in enumerate(data_set):
    #print(f'Processing chunk{i+1}')

    
    # Also Isolating the target variable to show what your trying to get the model to train based on
    #column = ['made']
    #x = chunk.drop(columns = column)
    #y = chunk['made']

    # Then split the data into different parts for the training and testing
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=56)

    #Train the Model
    #model.fit(x_train, y_train)

    # Creates predictions for th e model at each chunk
    #prediction = model.predict(x_test)
    #print(f'Prediction on chunk {i+1}: {prediction}')
   
    #Accuracy of the model for each chunk
    #accuracy = model.score(x_test, y_test)
    #print(f'Accuracy on chunk {i+1}: {accuracy}')

# The Next Steps 

# 1. Create a Model removing certain uneccesary variables and determine which variables are the most important and comparing it to the accuracy of the original model
columns_to_drop = ['caz', 'cay', 'cax', 'cvx', 'cvy', 'cvz', 'hght', 'cx', 'cy', 'cv', 'ca', 'dx', 'dy', 'dz', 'd', 'rt' ]
Shots.drop(columns= columns_to_drop, inplace =True)
Shots.to_csv('combined_shots_reduced.csv', index =False)



chunk_size = 10000
Shots2 = pd.read_csv('combined_shots_reduced.csv', chunksize = chunk_size)

# Creates a Heatmap that lets you look at only the variables that are left in data set
corr = Shots.corr()
plt.figure(figsize=(12,9))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.savefig('heatmap.png')
plt.show()


model2 = LogisticRegression(max_iter= 100000)

#Allows you to loop through the different parts of dat_set and allows
# Creates a finished model based on only the Key Variables, Power, Height, and Time
for i, chunk in enumerate(Shots2):
    print(f'Processing chunk{i+1}')

    
    # Also Isolating the target variable to show what your trying to get the model to train based on
    column = ['made']
    x = chunk.drop(columns = column)
    y = chunk['made']

    # Then split the data into different parts for the training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=56)

    #Train the Model
    model2.fit(x_train, y_train)

    #Predictions for the model for each chunk
    prediction = model2.predict(x_test)
    print(f'Prediction on chunk {i+1}: {prediction}')
    
    # Returns the Accuracy of the model for each chunk
    accuracy = model2.score(x_test, y_test)
    print(f'Accuracy on chunk {i+1}: {accuracy}')


#Saves the Dataset as its own file to be used in interface upload
file_path = 'trained_model2.pk1'
joblib.dump(model2, file_path)
print("Trained model saved to:", file_path)
