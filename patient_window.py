import tkinter as tk
from tkinter import filedialog
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, ReLU

# Import other necessary libraries and functions here

def create_windows(data, window_size, overlap):
    sampling_frequency = 500  # Sampling frequency is 500 Hz
    window_length = window_size * sampling_frequency
    overlap_length = overlap * sampling_frequency
    windows = []
    for i in range(0, len(data) - window_length, window_length - overlap_length):
        windows.append(data[i:i+window_length])
    return windows

def preprocess_windows_with_warping(windows):
    preprocessed_windows = []
    scaler = StandardScaler()
    for window in windows:
        # Apply time warping transformation here
        # For example, you can stretch or compress different segments of the signal
        
        # Standardize each window without detrending
        window_std = scaler.fit_transform(window)  # Standardization
        preprocessed_windows.append(window_std)
    return preprocessed_windows

from tensorflow.keras.models import load_model

model_path = "model.h5"  # Path to your model file
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

def open_patient_window():
    # Create a new window for the patient
    patient_window = tk.Toplevel()
    patient_window.title("Patient Upload Window")

    # Set the size of the window
    patient_window.geometry("400x300")

    def upload_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            file_label.config(text="File uploaded:\n" + file_path)
            predict_button.config(state="normal")  # Enable predict button after file upload
        else:
            file_label.config(text="No file selected.")
            predict_button.config(state="disabled")  # Disable predict button if no file is selected

    def predict():
        print("Processing...")  # Placeholder for actual processing code
        
        # Read the file
        file_path = file_label.cget("text").split("\n")[-1]  # Extract file path from label
        if file_path:
            data = pd.read_csv(file_path, nrows=31000, usecols=range(20))  # Read the CSV file
            print("File loaded successfully.")
            
            # Divide data into windows
            windows = create_windows(data, window_size=2, overlap=1)
            print("Windows created successfully.")
            
            # Preprocess windows
            preprocessed_windows = preprocess_windows_with_warping(windows)
            preprocessed_windows_array = np.array(preprocessed_windows)
    
            # Print the shape of the preprocessed data
            print("Preprocessed data shape:", preprocessed_windows_array.shape)
            
            print("Windows preprocessed successfully.")
            
            # Use the model to predict
            predictions = model.predict(preprocessed_windows_array)  # Assuming model is already loaded
            
            # Process predictions further if needed
            print("Predictions:", predictions)

            # Convert predictions to binary format
            y_pred = np.argmax(predictions, axis=1)

            # Define class labels
            class_labels = ['Relaxed', 'Stressed']

            # Determine the most probable class for each prediction
            predicted_classes = [class_labels[pred] for pred in y_pred]

            # Display the result to the user
            result_label.config(text="Predicted state: " + predicted_classes[0])
        else:
            print("No file selected.")

    # Add a button to upload a file
    upload_button = tk.Button(patient_window, text="Upload File", command=upload_file, width=20, height=2)
    upload_button.pack(pady=20)

    # Add a button to predict
    predict_button = tk.Button(patient_window, text="Predict", command=predict, width=20, height=2, state="disabled")
    predict_button.pack(pady=10)

    # Add a label to display the uploaded file path
    file_label = tk.Label(patient_window, text="", font=("Arial", 12))
    file_label.pack(pady=10)

    # Add a label to display the prediction result
    result_label = tk.Label(patient_window, text="", font=("Arial", 12))
    result_label.pack(pady=10)
