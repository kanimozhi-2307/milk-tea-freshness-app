
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import torch

from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import pipeline

# TensorFlow Lambda deserialization compatibility
import keras.src.utils.python_utils as python_utils
python_utils.tf = tf


# ------------------------------------------
# PROJECT CONFIGURATION
# ------------------------------------------

REPO_ID = "kanimozhi23/milk-tea-freshness-model"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

CLASS_NAMES = ["FRESH", "MODERATE", "LOW"]

IMAGE_SIZE = (224, 224)

# Initial threshold. Must be tested and adjusted.
TEA_THRESHOLD = 0.65

TEA_LABEL = (
    "a photo of brewed milk tea in a cup or glass"
)

NOT_TEA_LABEL = (
    "a photo without brewed milk tea, such as "
    "an empty glass, water, coffee, another drink, "
    "food, a person, an animal, or an unrelated object"
)


# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Milk Tea Freshness Detection",
    page_icon="🧋",
    layout="wide"
)


# ------------------------------------------
# LOAD CLIP TEA DETECTOR
# ------------------------------------------

@st.cache_resource
def load_tea_detector():

    detector = pipeline(
        task="zero-shot-image-classification",
        model=CLIP_MODEL_ID,
        device=-1
    )

    return detector


# ------------------------------------------
# TEA / NOT-TEA DETECTION
# ------------------------------------------

def detect_tea(image, detector):

    image = image.convert("RGB")

    results = detector(
        image,
        candidate_labels=[
            TEA_LABEL,
            NOT_TEA_LABEL
        ],
        hypothesis_template="{}"
    )

    score_map = {
        item["label"]: float(item["score"])
        for item in results
    }

    tea_score = score_map.get(TEA_LABEL, 0.0)

    not_tea_score = score_map.get(NOT_TEA_LABEL, 0.0)

    is_tea = tea_score >= TEA_THRESHOLD

    return is_tea, tea_score, not_tea_score


# ------------------------------------------
# LOAD EXISTING FRESHNESS MODEL
# ------------------------------------------

@st.cache_resource
def load_freshness_model():

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
    "CLIP AI + EfficientNetV2-S + CBAM"
)

st.write(
    "Upload a brewed milk tea image to estimate its "
    "visual freshness level using deep learning."
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
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded_image = st.camera_input(
        "Capture a milk tea image"
    )


# ------------------------------------------
# DISPLAY IMAGE AND PREDICT
# ------------------------------------------

if uploaded_image is not None:

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    if st.button(
        "Predict Freshness",
        type="primary"
    ):

        try:

            # STEP 1: CHECK WHETHER IMAGE CONTAINS TEA

            with st.spinner(
                "AI is checking the uploaded image..."
            ):

                tea_detector = load_tea_detector()

                is_tea, tea_score, not_tea_score = (
                    detect_tea(image, tea_detector)
                )

            st.divider()

            st.subheader("AI Image Verification")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Milk Tea Match Score",
                    f"{tea_score * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Non-Tea Match Score",
                    f"{not_tea_score * 100:.2f}%"
                )

            # STEP 2: REJECT NON-TEA IMAGES

            if not is_tea:

                st.error("🚫 TEA NOT FOUND")

                st.warning(
                    "The uploaded image could not be verified "
                    "as brewed milk tea. Please upload a clear "
                    "milk tea image."
                )

                st.stop()

            # STEP 3: PREDICT FRESHNESS

            st.success("✅ MILK TEA DETECTED")

            with st.spinner(
                "Estimating milk tea freshness..."
            ):

                model = load_freshness_model()

                predicted_class, confidence, predictions = (
                    predict_freshness(image, model)
                )

            st.divider()

            st.subheader("Freshness Prediction Result")

            if predicted_class == "FRESH":

                st.success("FRESH")

            elif predicted_class == "MODERATE":

                st.warning("MODERATE")

            else:

                st.error("LOW")

            st.metric(
                label="Predicted Freshness",
                value=predicted_class,
                delta=f"{confidence:.2f}% model score"
            )

            # CLASS-WISE PROBABILITY

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
    This application uses a two-stage AI approach.

    **Stage 1 – Image Verification:**
    CLIP checks whether the uploaded image resembles brewed milk tea.

    **Stage 2 – Freshness Estimation:**
    EfficientNetV2-S with CBAM predicts the freshness category.

    **Input:** RGB image

    **Image Size for Freshness Model:** 224 × 224 pixels

    **Output:** FRESH, MODERATE, LOW, or TEA NOT FOUND
    """
)

st.info(
    "The application estimates visual freshness patterns "
    "learned from the dataset. It is not a laboratory test "
    "or a food-safety assessment."
)

st.caption(
    "Developed for the brewed milk tea freshness "
    "estimation research project."
)
