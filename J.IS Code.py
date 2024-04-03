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

# The code below is used to read the datasets and save them to be able to accessed/referenced in the code later on.
cwd = os.getcwd() 
print(cwd)
file_path = 'C:/Users/chess/OneDrive/Desktop/J.I.S Project/path_detail.csv'
file_path_made = 'C:/Users/chess/OneDrive/Desktop/J.I.S Project/path_detail_made.csv'
file_path_miss = 'C:/Users/chess/OneDrive/Desktop/J.I.S Project/path_detail_miss.csv'
data = pd.read_csv(file_path)


data_made = pd.read_csv(file_path_made)
data_miss = pd.read_csv(file_path_miss)


# The following code adds a column which says wether a shot was a make or a miss. 
# These columns are not previously added but there are two different datasets one for made shots and one for missed shots so this allows them to be merged and differentiated from one another.

data_made['made'] = 1
data_miss['made'] = 0


# These functions below allows you to view and look at the different parts of the dataset

# print(data.head())
# print(data.info())
# print(data.describe())
# print(data_miss.info())
# print(data_made.info())


#These functions are used to clean the datasets by removing and row with missing values and any duplicate values found in the dataset

print("Number of Missing Values")
print(data_miss.isnull().sum())
data_miss.dropna(inplace=True)
data_miss.drop_duplicates(inplace = True)

print("Number of Missing Values")
print(data_made.isnull().sum())
data_made.dropna(inplace=True)
data_made.drop_duplicates(inplace = True)

#The Following code is used to combine the cleaned datasets and save them to prevent having to clean and run all the above code everytime.

data_combined = pd.concat([data_made, data_miss], ignore_index= True)
data_shuffled = data_combined.sample(frac=1, random_state = 56)
data_shuffled.to_csv('combined_shots.csv', index =False)



Shots = pd.read_csv('combined_shots.csv')

Shots_small = pd.read_csv('combined_shots.csv', nrows=100)
Shots_made = pd.read_csv(file_path_made, nrows=100 )
print ("tHIS IS TEH mERGED Dataset")
print(Shots.head())
print(Shots.tail())

# Scatterplot that shows the differences between 
plt.figure(figsize=(8,6))
plt.scatter(Shots_made['t'], Shots_made['cv'])
plt.colorbar(label = 'Shot Outcome')
plt.xlabel('Shot Release Time')
plt.ylabel('Shot Speed')
plt.title('Basketball Shots: Shot Duration vs Speed')
plt.grid(True)
plt.show()

# Scatterplot showing the correlation Height and Release Time

# plt.figure(figsize=(8,6))
# plt.scatter(Shots_made['hght'], Shots_made['rt'])
# plt.colorbar(label = 'Shot Outcome')
# plt.xlabel('Shot Release Time')
# plt.ylabel('Shot Speed')
# plt.title('Basketball Shots: Distance vs Speed')
# plt.grid(True)
# plt.show()

# Scatterplot Showing the correlation between shot speed and acceleration

plt.figure(figsize=(8,6))
plt.scatter(Shots_made['ca'], Shots_made['cv'])
plt.colorbar(label = 'Shot Outcome')
plt.xlabel('Shot Acceleration')
plt.ylabel('Shot Speed')
plt.title('Basketball Shots: Acceleration vs Speed')
plt.grid(True)
plt.show()

# Create a Combined Variable for Velocity and Do a scatterplot for it
Shots_made['Velocity'] = Shots_made['cvx'] * Shots_made['cvy'] * Shots_made['cvz']

# plt.figure(figsize=(8,6))
# plt.scatter(Shots_made['Velocity'], Shots_made['rt'])
# plt.colorbar(label = 'Shot Outcome')
# plt.xlabel('Shot Velocity')
# plt.ylabel('Release Time')
# plt.title('Basketball Shots: Velocity vs Release Time')
# plt.grid(True)
# plt.show()
# Machine Learning Model Creation

# The dataset is Extremly large so this code is implemented to allow you to process the data when training the model one chunk(part at a time)

chunk_size = 10000
data_set = pd.read_csv('combined_shots.csv', chunksize =chunk_size)



#Create the model, You should do a logistic regression when the result is either yes or no and there are linear relationships in the data
model = LogisticRegression(max_iter = 100000)


#Allows you to loop through the different parts of dat_set and allows
for i, chunk in enumerate(data_set):
    print(f'Processing chunk{i+1}')

    # Split the dataset into different parts with some to be used to train and some being used to test
    # Also Isolating the target variable to show what your trying to get the model to train based on
    columns_to_drop = ['made', 'fnm', 'lnm', 'pid']
    x = chunk.drop(columns = columns_to_drop)
    y = chunk['made']

    # Then split the data into different parts for the training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=56)

    #Train the Model
    model.fit(x_train, y_train)

    #
    #prediction = model.predict(x_test)
    #print(f'Prediction on chunk {i+1}: {prediction}')

    accuracy = model.score(x_test, y_test)
    print(f'Accuracy on chunk {i+1}: {accuracy}')

# The Next Steps 
# 1. Create a Model removing certain uneccesary variables and determine which variables are the most important and comparing it to the accuracy of the original model

# for i, chunk in enumerate(data_set):
    #print(f'Processing chunk{i+1}')

    # Split the dataset into different parts with some to be used to train and some being used to test
    # Also Isolating the target variable to show what your trying to get the model to train based on
    #columns_to_drop = ['made', 'fnm', 'lnm', 'pid']
    #x = chunk.drop(columns = columns_to_drop)
    # y = chunk['made']

    # Then split the data into different parts for the training and testing
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=56)

    #Train the Model
    # model.fit(x_train, y_train)

    #
    #prediction = model.predict(x_test)
    #print(f'Prediction on chunk {i+1}: {prediction}')

    #accuracy = model.score(x_test, y_test)
    # print(f'Accuracy on chunk {i+1}: {accuracy}')
#2. Check for any problems with the secondary model
#3. Create an Interface to be ables to use the trained model 
#4. Create a graphic to allow the user to input data to see if they can perform a good shot