import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model here
model = tf.keras.models.load_model('potatoes.keras')

# Tensorflow Model Prediction
def model_prediction(image):
    # Resize image to match model input size
    image = image.resize((512, 512))  # Update to the size your model expects
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Convert single image to batch
    image = image / 255.0  # Normalize the image
    predictions = model.predict(image)
    return predictions

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://www.example.com/farm_background.jpg');
        background-size: cover;
        background-position: center;
        background-color: black;
    }
    .sidebar .sidebar-content {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.title("Leaf Disease Recognition System")
    st.image("Home2.jpg", use_column_width=True)
    st.markdown("""
    Welcome to the Leaf Disease Recognition System. Use the sidebar to navigate to the Disease Recognition page.
    """)

elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg", "gif"])
    
    if test_image:
        image = Image.open(test_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if st.button("Predict"):
            st.write("Analyzing the image...")
            # Ensure the image is resized correctly
            image = image.resize((512, 512))  # Resize to the size your model expects
            predictions = model_prediction(image)
            result_index = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # Class names for prediction
            class_names = ['Tomato___Bacterial_spot', 
                           'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                           'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                           'Tomato___healthy']

            # Threshold for confidence
            confidence_threshold = 0.5
            if confidence > confidence_threshold:
                st.success(f"Model is predicting it's a {class_names[result_index]} with confidence {confidence:.2f}")
            else:
                st.error("The image is not a plant or is not recognized by the model. Please try again with a different image.")
