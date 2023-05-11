import streamlit as st
import os
from processor import preprocess, predict
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import requests


st.set_page_config(
   page_title="Alzheimer's Disease Detection",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.image("data/cn_example.png")

with st.sidebar:

    ATLAS = {
        "MNI305": "atlas/tpl-MNI305_T1w.nii.gz",
        "Empty_slot": "atlas/tpl-MNI305_T1w.nii.gz"}

    MODELS = {
        "Inception": "models/inception_model1.h5",
        "Empty_slot": "models/inception_model1.h5"}

    st.header("Upload subject MRI image:")
    image = st.file_uploader("Choose an image", type=".nii", key="image")

    st.header("Select atlas and model:")
    atlas_key = st.selectbox("Choose a referencece image", [i for i in ATLAS.keys()])
    atlas_value = ATLAS.get(atlas_key)
    model_key = st.selectbox("Choose a deep-learning model", [i for i in MODELS.keys()])
    model_value = MODELS.get(model_key)

    #st.cache(allow_output_mutation=True)
    def detect_AD():
        preprocess(os.path.join("./input",image.name), atlas_value)
        predict(f"./output/{image.name}_2d.npy", chosen_model)

    with st.spinner('Loading model'):
        if st.button("Load model"):
            if image is not None and atlas_value is not None and model_value is not None:
                with open(os.path.join("./input",image.name),"wb") as f:
                    f.write(image.getbuffer())
                chosen_model = load_model(model_value)
                #st.success("Model loaded")
                trigger = st.button('Begin analysis', on_click=detect_AD)
            else:
                st.error("Error: One or more input is/are missing.", icon="🚨")


# Example local Docker container URL
# url = 'http://api:8000'
# Example localhost development URL
# url = 'http://localhost:8000'
load_dotenv()
url = os.getenv('API_URL')


# App title and description
st.header('Simple Image Uploader 📸')
st.markdown('''
            > This is a Le Wagon boilerplate for any data science projects that involve exchanging images between a Python API and a simple web frontend.
            > **What's here:**
            > * [Streamlit](https://docs.streamlit.io/) on the frontend
            > * [FastAPI](https://fastapi.tiangolo.com/) on the backend
            > * [PIL/pillow](https://pillow.readthedocs.io/en/stable/) and [opencv-python](https://github.com/opencv/opencv-python) for working with images
            > * Backend and frontend can be deployed with Docker
            ''')

st.markdown("---")

### Create a native Streamlit file upload input
st.markdown("### Let's do a simple face recognition 👇")
img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_file_buffer.getvalue()

      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes})

      if res.status_code == 200:
        ### Display the image returned by the API
        st.image(res.content, caption="Image returned from API ☝️")
      else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)
