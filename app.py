import joblib
import pandas as pd
import streamlit as st
from numpy import outer
import cv2
import numpy as np
import boto3
from io import BytesIO
import os
#import s3fs
#fs = s3fs.S3FileSystem() # Updated method name
#import s3fs

#fs = s3fs.S3FileSystem() # Updated method name
#filename = "s3://pedestrianfriendlyproject/model1.joblib>"
#with fs.open(filename, encoding='utf8') as fh:
#    model = joblib.load(fh)
# Specify your S3 bucket and file paths
s3 = boto3.resource('s3')
bucket_name = 'pedestrianfriendlyproject'
location_folder_key = 'Location/'

# Specify the local folder path to download the images
local_folder_path = 's3://pedestrianfriendlyproject/MLModels/'  # Update this with your local folder path

# Ensure the local folder exists
os.makedirs(local_folder_path, exist_ok=True)

model = joblib.load(local_folder_path+'model1.joblib')
img_size = (224, 224)
 
def web_app():
 
    st.title("Image-based Model Deployment")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type="png")
    print("Uploaded file is",uploaded_file)
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make a prediction when the user clicks the button
        if st.button("Make Prediction"):
            # Load the image
            image = cv2.imread("https://s3.console.aws.amazon.com/s3/object/pedestrianfriendlyproject/Location/1-10.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # If using OpenCV, convert BGR to RGB
            actual_rating = 5.0
            # Resize the image to match the input size of your model
            # (assuming img_size is the target size used during training)
            image = cv2.resize(image, img_size)
 
            # Preprocess the image if necessary (e.g., rescaling)
            image = image / 255.0  # Assuming rescale=1./255 in your ImageDataGenerator during training
 
            # Expand dimensions to match the model's expected input shape
            image = np.expand_dims(image, axis=0)
 
            # Make a prediction using the loaded model
            prediction = model.predict(image)[0][0]
 
            st.text_area(label='Predicted Value is:- ',value=prediction , height= 100)
run = web_app()
