    upload_button = tk.Button(canvas_frame, text="Upload File", command=upload_file, width=20, height=2)
    upload_button.pack(pady=20, anchor=tk.CENTER)

    # Add a button to predict
    predict_button = tk.Button(canvas_frame, text="Predict", command=predict, width=20, height=2, state="disabled")
    predict_button.pack(pady=10, anchor=tk.CENTER)