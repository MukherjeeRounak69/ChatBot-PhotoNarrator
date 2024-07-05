import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (load_gemini_pro_model,gemini_pro_vision_response)
working_directory=os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Gemini AI",
    layout="centered"
)

with st.sidebar:
    selected=option_menu("RounakAI",
                         ["ChatBot",
                          "Image Captioning"],
                         menu_icon='robot',icons=['chat-dots-fill','image-fill'],
                         default_index=0)

def translate_role_for_streamlit(user_role):
    if user_role == 'model':
        return "assistant"
    else:
        return user_role

if selected=="ChatBot":
    model=load_gemini_pro_model()
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])
    st.title("🦾 ChatBot")
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt=st.chat_input("Ask Gemini Pro...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        gemini_response=st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

if selected == "Image Captioning":
    st.title("📷 Photo Narrator")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if st.button("Generate Caption"):
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            resized_img = image.resize((800, 500))
            st.image(resized_img)

        default_prompt = "Write a short caption for this image"
        caption = gemini_pro_vision_response(default_prompt, image)
        with col2:
            st.info(caption)

