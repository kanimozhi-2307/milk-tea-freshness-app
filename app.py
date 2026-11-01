
import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download
from google import genai


# TensorFlow Lambda deserialization compatibility
import keras.src.utils.python_utils as python_utils
python_utils.tf = tf


# ------------------------------------------
# PROJECT CONFIGURATION
# ------------------------------------------

REPO_ID = "kanimozhi23/milk-tea-freshness-model"

MODEL_FILENAME = "model.keras"

GEMINI_MODEL = "gemini-3.5-flash-lite"

CLASS_NAMES = ["FRESH", "MODERATE", "LOW"]

IMAGE_SIZE = (224, 224)


# ------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Milk Tea Freshness Detection",
    page_icon="🧋",
    layout="wide"
)


# ------------------------------------------
# LOAD GEMINI AI CLIENT
# ------------------------------------------

@st.cache_resource
def load_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. "
            "Please add GEMINI_API_KEY in Streamlit Secrets."
        )

    client = genai.Client(api_key=api_key)

    return client


# ------------------------------------------
# GEMINI AI: TEA / NON-TEA DETECTION
# ------------------------------------------

def detect_tea(image, client):

    prompt = """
    Examine the uploaded image carefully.

    Determine whether the image clearly contains brewed milk tea
    as a prepared beverage in a cup, glass, or container.

    Classification rules:

    1. MILK_TEA:
       Select this only when brewed milk tea is visibly present.

    2. NOT_MILK_TEA:
       Select this for water, coffee, juice, plain milk,
       other beverages, food, people, animals, objects,
       empty cups, tea leaves, tea powder, or unrelated images.

    3. UNCERTAIN:
       Select this if the image is unclear, too dark, blurry,
       or does not provide enough visual evidence.

    Do not assume that every brown or beige liquid is milk tea.
    Do not identify freshness or food safety.

    Return only the requested structured result.
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            image
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "classification": {
                        "type": "STRING",
                        "enum": [
                            "MILK_TEA",
                            "NOT_MILK_TEA",
                            "UNCERTAIN"
                        ]
                    },
                    "reason": {
                        "type": "STRING"
                    }
                },
                "required": [
                    "classification",
                    "reason"
                ]
            }
        }
    )

    if not response.text:
        raise ValueError(
            "Gemini did not return a classification."
        )

    result = json.loads(response.text)

    classification = result.get(
        "classification",
        "UNCERTAIN"
    )

    reason = result.get(
        "reason",
        "The image could not be verified."
    )

    return classification, reason


# ------------------------------------------
# LOAD EXISTING FRESHNESS MODEL
# ------------------------------------------

@st.cache_resource
def load_freshness_model():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
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

    predicted_index = int(
        np.argmax(predictions)
    )

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
    "Gemini AI + EfficientNetV2-S + CBAM"
)

st.write(
    """
    Upload or capture a brewed milk tea image.
    The application first verifies whether the image contains
    milk tea and then estimates its visual freshness level.
    """
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

    try:
        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            caption="Selected Image",
            width=400
        )

    except Exception:
        st.error("Unable to read this image. Please try another image.")
        st.stop()

    if st.button(
        "Predict Freshness",
        type="primary"
    ):

        try:

            # STEP 1: GEMINI TEA DETECTION

            with st.spinner(
                "Gemini AI is checking whether this is milk tea..."
            ):

                gemini_client = load_gemini_client()

                classification, reason = detect_tea(
                    image,
                    gemini_client
                )

            st.divider()

            st.subheader("AI Image Verification")

            if classification == "NOT_MILK_TEA":

                st.error("🚫 TEA NOT FOUND")

                st.write(reason)

                st.warning(
                    "Please upload or capture a clear image "
                    "of brewed milk tea."
                )

                st.stop()

            elif classification == "UNCERTAIN":

                st.warning("⚠️ IMAGE COULD NOT BE VERIFIED")

                st.write(reason)

                st.info(
                    "Please try again with a clear, well-lit "
                    "image of brewed milk tea."
                )

                st.stop()

            # STEP 2: MILK TEA CONFIRMED

            st.success("✅ MILK TEA DETECTED")

            st.write(reason)

            # STEP 3: FRESHNESS PREDICTION

            with st.spinner(
                "EfficientNetV2-S + CBAM is estimating freshness..."
            ):

                model = load_freshness_model()

                predicted_class, confidence, predictions = (
                    predict_freshness(image, model)
                )

            st.divider()

            st.subheader("Freshness Prediction Result")

            if predicted_class == "FRESH":

                st.success("🟢 FRESH")

            elif predicted_class == "MODERATE":

                st.warning("🟡 MODERATE")

            else:

                st.error("🔴 LOW")

            st.metric(
                label="Predicted Freshness",
                value=predicted_class,
                delta=f"{confidence:.2f}% model score"
            )

            # CLASS-WISE PROBABILITY

            st.subheader("Class-wise Prediction Scores")

            for i, class_name in enumerate(CLASS_NAMES):

                probability = float(
                    predictions[i]
                )

                st.write(
                    f"**{class_name}: "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(
                    max(
                        0.0,
                        min(1.0, probability)
                    )
                )

            st.success(
                "Freshness prediction completed successfully!"
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

    Gemini AI checks whether the uploaded image contains
    brewed milk tea.

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
