import joblib
import pandas as pd
import streamlit as st
from numpy import outer
import cv2
import numpy as np
#import s3fs
#fs = s3fs.S3FileSystem() # Updated method name
 
model = joblib.load('s3://pedestrianfriendlyproject/model1.joblib')
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
            image = cv2.imread("s3://pedestrianfriendlyproject/Location/1-11.png")
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
