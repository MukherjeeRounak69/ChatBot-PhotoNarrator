import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (load_gemini_pro_model, gemini_pro_vision_response)
import tensorflow as tf
import numpy as np

# Set up working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

# Set Streamlit page configuration
st.set_page_config(
    page_title="Gemini AI",
    layout="centered"
)

# Custom CSS for styling
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f3f1fc;
#         color: #333333;
#     }
#     .sidebar .sidebar-content {
#         background-color: #e4e1f8;
#         color: #333333;
#     }
#     .stTextInput > div > div > input {
#         background-color: #ffffff;
#         color: #000000;
#     }
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#     }
#     .stFileUploader {
#         color: #333333;
#     }
#     .css-1v3fvcr {
#         background-color: #f3f1fc;
#         color: #333333;
#     }
#     .css-1544g2n {
#         background-color: #e4e1f8;
#         color: #333333;
#     }
#     .css-1q5akcf {
#         color: #000000;
#     }
#     .css-1d391kg p {
#         color: #000000;
#     }
#     .css-145kmo2, .css-1d391kg, .css-1q5akcf, .css-1v3fvcr {
#         color: #333333;
#     }
#     .css-1q1cmh1, .css-1lsmgbg, .css-1ko4ion {
#         color: #333333;
#     }
#     .css-1d391kg {
#         background-color: #ffffff;
#         color: #333333;
#     }
#     .stButton button:hover {
#         background-color: #45a049;
#         color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu("RounakAI",
                           ["ChatBot", "Image Captioning", "Plant Disease Prediction"],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'leaf-fill'],
                           default_index=0)

def translate_role_for_streamlit(user_role):
    if user_role == 'model':
        return "assistant"
    else:
        return user_role

# ChatBot functionality
if selected == "ChatBot":
    model = load_gemini_pro_model()
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
    st.title("🦾 ChatBot")
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Gemini Pro...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image Captioning functionality
elif selected == "Image Captioning":
    st.title("📷 Photo Narrator")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("Generate Caption"):
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            resized_img = image.resize((800, 500))
            st.image(resized_img)

        default_prompt = "Write a short caption for this image"
        caption = gemini_pro_vision_response(default_prompt, image)
        with col2:
            st.info(caption)

# Plant Disease Prediction functionality
elif selected == "Plant Disease Prediction":
    st.title("🌿 Plant Disease Classifier")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    UPLOAD_FOLDER = 'uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    model_path = os.path.join(working_directory, 'model/Plant_Disease_Classification (2).h5')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
    else:
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")

    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
                   'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                   'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
                   'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                   'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                   'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    def load_and_prep_image(image, target_size=(210, 210)):
        img = image.resize(target_size)
        img_arr = np.array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = img_arr / 255.0
        return img_arr

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Predict Disease"):
            if 'model' in locals():
                try:
                    preprocessed_image = load_and_prep_image(image)
                    prediction = model.predict(preprocessed_image)
                    prediction_class = np.argmax(prediction[0])
                    confidence = float(np.max(prediction))
                    prediction_result = {
                        'prediction': class_names[prediction_class],
                        'confidence': confidence
                    }

                    st.success(f"Prediction: {prediction_result['prediction']}")
                    st.info(f"Confidence: {prediction_result['confidence']:.2f}")
                except Exception as e:
                    st.error(f"Error processing image or making prediction: {e}")
            else:
                st.error("Model not loaded.")
