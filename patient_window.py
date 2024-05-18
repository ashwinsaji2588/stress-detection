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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    patient_window.geometry("1920x1080")
    data = pd.DataFrame()
    preprocessed_windows_array = None

    def upload_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            file_label.config(text="File uploaded:\n" + file_path)
            read_data()
            predict_button.config(state="normal")  # Enable predict button after file upload
        else:
            file_label.config(text="No file selected.")
            predict_button.config(state="disabled")  # Disable predict button if no file is selected

    def read_data():
        file_path = file_label.cget("text").split("\n")[-1]  # Extract file path from label
        # Read the file
        global data 
        data = pd.read_csv(file_path, nrows=31000, usecols=range(19))
        print("File loaded successfully.")
        print("Data shape:", data.shape)
            
        # Divide data into windows
        windows = create_windows(data, window_size=2, overlap=1)
        print("Windows created successfully.")
            
        # Preprocess windows
        preprocessed_windows = preprocess_windows_with_warping(windows)
        global preprocessed_windows_array 
        preprocessed_windows_array = np.array(preprocessed_windows)
    
        # Print the shape of the preprocessed data
        print("Preprocessed data shape:", preprocessed_windows_array.shape)
        print("Windows preprocessed successfully.")

    def predict():
        print("Processing...")  # Placeholder for actual processing code
        global preprocessed_windows_array
        # Read the file
        file_path = file_label.cget("text").split("\n")[-1]  # Extract file path from label
        if file_path:
            global predictions
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
            plot_button.config(state="normal")  # Enable plot button after file upload
        else:
            print("No file selected.")


    def plot_data():
        file_path = file_label.cget("text").split("\n")[-1]
        global data
        global preprocessed_windows_array
        if file_path:

            # Create a new figure and axis if not already created
            if 'fig' not in globals():
                fig, (ax_main, ax_nav) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [6, 1]})
                fig.canvas.draw()

            # Clear the existing plot
            ax_main.clear()
            ax_nav.clear()

            plt.subplots_adjust(hspace=1)

            # Generate time values for the x-axis
            time = np.linspace(0, 60, 31000)

            # Plot the navigator subplot
            sns.lineplot(x=time, y=data.iloc[:, 0], ax=ax_nav)
            ax_nav.get_yaxis().set_visible(False)

            # Plot the initial detailed view (first 1 second)
            for i in range(19):  # Assuming you want to plot 20 line plots
                sns.lineplot(x = range(1000), y=data.iloc[0:1000, i], ax=ax_main)
            ax_main.set_xlim(0, 1000)
            ax_main.set_xlabel("Segments (500 Hz)")
            ax_main.set_ylabel("Signal")
            ax_main.set_title("Heat Map (0-2 seconds)")

            # Initialize the shaded area in the navigator
            shaded_area = ax_nav.axvspan(0, 2, color='grey', alpha=0.5)

            def on_click(event):
                # Check if the mouse is clicked within the navigation plot area
                if event.inaxes == ax_nav:
                    # Calculate the start and end of the one-second window based on the click position
                    x = int(event.xdata)
                    start_x = max(0, x)
                    start_x = min(58, start_x)
                    end_x = min(60, start_x + 2)

                    # Update the shaded area in the navigator
                    shaded_area.set_xy([[start_x, 0], [start_x, 2], [end_x, 2], [end_x, 0], [start_x, 0]])

                    # Clear the detailed view plot before redrawing
                    ax_main.clear()
                    # Update the detailed view plot
                    for i in range(19):  # Assuming you want to plot 20 line plots
                        sns.lineplot(x=range(start_x*500, end_x*500), y=data.iloc[start_x*500: end_x*500, i], ax=ax_main)
                    ax_main.set_xlim(start_x*500, end_x*500)
                    ax_main.set_xlabel("Segments (500 Hz)")
                    ax_main.set_ylabel("Signal")
                    ax_main.set_title(f"Heat Map ({start_x}-{end_x} seconds)")

                    # Redraw the canvas to update the plot
                    fig.canvas.draw_idle()

            # Connect the button_press_event to the on_click function
            fig.canvas.mpl_connect("button_press_event", on_click)

            # Embed the Matplotlib figure into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=patient_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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

    # Add a button to plot the data
    plot_button = tk.Button(patient_window, text="Plot", command=plot_data, width=20, height=2, state="disabled")
    plot_button.pack(pady=10)