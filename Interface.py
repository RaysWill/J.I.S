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

import tkinter as tk
from PIL import Image, ImageTk
import joblib

# The code below is used to read the datasets and save them to be able to accessed/referenced in the code later on.


file_path = r"C:\Users\chess\OneDrive\Desktop\J.I.S Project\trained_model2.pk1"  # Specify the correct file name and path
loaded_model = None
if os.path.exists(file_path):
    try:
        # Load the trained model from the file
        loaded_model = joblib.load(file_path)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error occurred while loading the model:", e)
else:
    print("File does not exist.")
    # Use trained model to make prediction


# Create a Tkinter window
window = tk.Tk()
window.title("Shot Prediction")

# Function to make prediction based on user input
def predict_shot():
    # Retrieve user input from entry fields
    height = float(height_entry.get())
    power = float(power_entry.get())
    t = float(Release_Time_entry.get())

    # Add preprocessing steps here if needed
   
    probability = loaded_model.predict_proba([[height, power, t]])[0][1]  # Probability of shot being made
    
    # Display predicted probability
    result_label.config(text=f"Probability of shot being made: {probability:.2f}")

# Input fields for speed and acceleration
Distance_label = tk.Label(window, text = "Distance From Hoop is - 25FT")
Distance_label.grid(row = 0, columnspan=2) 

height_label = tk.Label(window, text="Enter Height:")
height_label.grid(row=1, column=0)
height_entry = tk.Entry(window)
height_entry.grid(row=1, column=1)

power_label = tk.Label(window, text="Enter Power:")
power_label.grid(row=2, column=0)
power_entry = tk.Entry(window)
power_entry.grid(row=2, column=1)

Release_Time_label = tk.Label(window, text = "Enter Shot Time")
Release_Time_label.grid(row=3, column=0)
Release_Time_entry = tk.Entry(window)
Release_Time_entry.grid(row =3, column=1 )


# Button to trigger prediction
predict_button = tk.Button(window, text="Predict", command=predict_shot)
predict_button.grid(row=4, columnspan=2)

# Label to display result
result_label = tk.Label(window, text="")
result_label.grid(row=5, columnspan=2)

court = Image.open("Basketball_court.jpg")

court_display = ImageTk.PhotoImage(court)

canvas = tk.Canvas(window, width =court.width, height = court.height)
canvas.grid(row= 6, columnspan =2)

canvas.create_image(0,0, anchor=tk.NW, image=court_display)

# Run the Tkinter event loop
window.mainloop()
