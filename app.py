
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Fix TensorFlow Lambda deserialization compatibility
import keras.src.utils.python_utils as python_utils
python_utils.tf = tf


# ------------------------------------------
# PROJECT CONFIGURATION
# ------------------------------------------

REPO_ID = "kanimozhi23/milk-tea-freshness-model"

CLASS_NAMES = ["FRESH", "MODERATE", "LOW"]

IMAGE_SIZE = (224, 224)


# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Milk Tea Freshness Detection",
    page_icon="🧋",
    layout="wide"
)


# ------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.keras",
        repo_type="model"
    )

    loaded_model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    return loaded_model


# ------------------------------------------
# FRESHNESS PREDICTION
# ------------------------------------------

def predict_freshness(image, model):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    return predicted_class, confidence, predictions


# ------------------------------------------
# APPLICATION TITLE
# ------------------------------------------

st.title("🧋 Milk Tea Freshness Detection")

st.subheader(
    "EfficientNetV2-S + CBAM Deep Learning Model"
)

st.write(
    "Estimate the freshness level of brewed milk tea "
    "using an RGB image."
)

st.divider()


# ------------------------------------------
# FRESHNESS CATEGORY INFORMATION
# ------------------------------------------

st.subheader("Freshness Categories")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("FRESH")
    st.write("0–3 minutes")

with col2:
    st.warning("MODERATE")
    st.write("4–10 minutes")

with col3:
    st.error("LOW")
    st.write("11–20 minutes")

st.divider()


# ------------------------------------------
# IMAGE INPUT
# ------------------------------------------

st.subheader("Upload or Capture Milk Tea Image")

input_method = st.radio(
    "Select image input method",
    ["Upload Image", "Capture Image"],
    horizontal=True
)

if input_method == "Upload Image":

    uploaded_image = st.file_uploader(
        "Choose a milk tea image",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded_image = st.camera_input(
        "Capture a milk tea image"
    )


# ------------------------------------------
# DISPLAY AND PREDICT
# ------------------------------------------

if uploaded_image is not None:

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    st.image(
        image,
        caption="Input Milk Tea Image",
        width=400
    )

    if st.button(
        "Predict Freshness",
        type="primary"
    ):

        try:

            with st.spinner(
                "Loading model and estimating freshness..."
            ):

                model = load_model()

                predicted_class, confidence, predictions = (
                    predict_freshness(image, model)
                )

            st.divider()

            st.subheader("Prediction Result")

            st.metric(
                label="Predicted Freshness",
                value=predicted_class,
                delta=f"{confidence:.2f}% confidence"
            )

            st.subheader("Class-wise Probability")

            for i, class_name in enumerate(CLASS_NAMES):

                probability = float(
                    predictions[i]
                )

                st.write(
                    f"**{class_name}: "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

            st.success(
                "Freshness prediction completed!"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ------------------------------------------
# ABOUT THE PROJECT
# ------------------------------------------

st.divider()

st.subheader("About the Project")

st.write(
    """
    This application estimates the freshness of brewed
    milk tea using a deep learning model.

    **Architecture:** EfficientNetV2-S + CBAM

    **Input:** RGB milk tea image

    **Image Size:** 224 × 224 pixels

    **Output:** FRESH, MODERATE, or LOW
    """
)

st.info(
    "This application estimates visual freshness patterns "
    "learned from the training dataset. It is not a laboratory "
    "test or a food-safety assessment."
)

st.caption(
    "Developed for the brewed milk tea freshness "
    "estimation research project."
)
