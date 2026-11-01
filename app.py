
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import re

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download
from google import genai
from google.genai import types


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REPO_ID = "kanimozhi23/milk-tea-freshness-model"
MODEL_FILENAME = "model.keras"

GEMINI_MODEL = "gemini-3.5-flash-lite"

CLASS_NAMES = ["FRESH", "MODERATE", "LOW"]
IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Milk Tea Freshness Detection",
    page_icon="🧋",
    layout="centered"
)

st.title("🧋 Milk Tea Freshness Detection")

st.write(
    "Upload or capture a brewed milk tea image "
    "to estimate its freshness level using AI."
)

st.info(
    "First, Gemini AI checks whether the image contains "
    "brewed milk tea. The freshness model runs only after "
    "milk tea is detected."
)


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

@st.cache_resource
def load_gemini_client():

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Please add it in Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


# --------------------------------------------------
# MILK TEA DETECTION
# --------------------------------------------------

def detect_tea(image, client):

    image_bytes = io.BytesIO()

    image.convert("RGB").save(
        image_bytes,
        format="JPEG"
    )

    prompt = """
    Examine the uploaded image carefully.

    Determine whether the main visible subject is
    brewed milk tea in a cup or glass.

    Do not classify tea leaves, plain water, coffee,
    juice, other drinks, or unrelated objects as milk tea.

    Return exactly ONE of these labels:

    MILK_TEA
    NOT_MILK_TEA
    UNCERTAIN

    Use MILK_TEA only when brewed milk tea is clearly visible.
    Use NOT_MILK_TEA when the image clearly does not contain
    brewed milk tea.
    Use UNCERTAIN when the image is unclear, obstructed,
    or insufficient to make a reliable decision.

    Return only the label.
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes.getvalue(),
                mime_type="image/jpeg"
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=30
        )
    )

    if not response.text:
        return "UNCERTAIN"

    result = response.text.strip().upper()

    match = re.search(
        r"\b(NOT_MILK_TEA|MILK_TEA|UNCERTAIN)\b",
        result
    )

    if match:
        return match.group(1)

    return "UNCERTAIN"


# --------------------------------------------------
# LOAD TRAINED FRESHNESS MODEL
# --------------------------------------------------

@st.cache_resource
def load_freshness_model():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    return model


# --------------------------------------------------
# FRESHNESS PREDICTION
# --------------------------------------------------

def predict_freshness(image, model):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

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

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "The model output does not contain exactly "
            "three classes. Please check your trained model."
        )

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return predicted_class, confidence, predictions


# --------------------------------------------------
# IMAGE INPUT
# --------------------------------------------------

st.subheader("Upload or Capture Image")

input_method = st.radio(
    "Choose image input method:",
    ["Upload Image", "Capture Image"],
    horizontal=True
)

uploaded_image = None

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a milk tea image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)

else:

    camera_image = st.camera_input(
        "Capture a milk tea image"
    )

    if camera_image is not None:
        uploaded_image = Image.open(camera_image)


# --------------------------------------------------
# DISPLAY IMAGE
# --------------------------------------------------

if uploaded_image is not None:

    uploaded_image = uploaded_image.convert("RGB")

    st.image(
        uploaded_image,
        caption="Selected Image",
        use_container_width=True
    )

    if st.button(
        "Predict Freshness",
        type="primary",
        use_container_width=True
    ):

        # ------------------------------------------
        # STEP 1: CHECK WHETHER IMAGE CONTAINS TEA
        # ------------------------------------------

        st.subheader("Step 1: Milk Tea Verification")

        try:

            with st.spinner(
                "Gemini AI is checking the image..."
            ):

                client = load_gemini_client()

                detection_result = detect_tea(
                    uploaded_image,
                    client
                )

            if detection_result == "NOT_MILK_TEA":

                st.error("🚫 TEA NOT FOUND")

                st.warning(
                    "The uploaded image does not appear "
                    "to contain brewed milk tea. "
                    "Please upload a milk tea image."
                )

                st.stop()

            elif detection_result == "UNCERTAIN":

                st.warning(
                    "⚠️ MILK TEA NOT CLEARLY IDENTIFIED"
                )

                st.info(
                    "Please upload a clearer image with "
                    "the brewed milk tea clearly visible."
                )

                st.stop()

            elif detection_result == "MILK_TEA":

                st.success("✅ MILK TEA DETECTED")

        except Exception as e:

            st.error(
                "Milk tea verification failed. "
                "Please check your Gemini API key and logs."
            )

            st.exception(e)

            st.stop()


        # ------------------------------------------
        # STEP 2: FRESHNESS PREDICTION
        # ------------------------------------------

        st.subheader("Step 2: Freshness Prediction")

        try:

            with st.spinner(
                "Loading the trained model and predicting freshness..."
            ):

                model = load_freshness_model()

                predicted_class, confidence, predictions = (
                    predict_freshness(
                        uploaded_image,
                        model
                    )
                )

            st.success("Freshness Prediction Completed!")

            # --------------------------------------
            # DISPLAY RESULT
            # --------------------------------------

            st.markdown("### Predicted Freshness Level")

            if predicted_class == "FRESH":

                st.success("🟢 FRESH")

            elif predicted_class == "MODERATE":

                st.warning("🟡 MODERATE")

            else:

                st.error("🔴 LOW")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            # --------------------------------------
            # CLASS-WISE SCORES
            # --------------------------------------

            st.markdown("### Class-wise Prediction Scores")

            for class_name, score in zip(
                CLASS_NAMES,
                predictions
            ):

                percentage = float(score * 100)

                st.write(
                    f"**{class_name}: {percentage:.2f}%**"
                )

                st.progress(
                    min(max(percentage / 100, 0.0), 1.0)
                )

        except Exception as e:

            st.error(
                "Freshness prediction failed. "
                "Please check your model file and deployment logs."
            )

            st.exception(e)


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.divider()

st.subheader("About the Project")

st.write(
    """
    This project estimates the freshness level of brewed
    milk tea using RGB images and deep learning.

    **Freshness classification:**

    - FRESH: 0–3 minutes
    - MODERATE: 4–10 minutes
    - LOW: 11–20 minutes

    The system uses EfficientNetV2-S with CBAM for
    image-based freshness classification.
    """
)

st.caption(
    "This is an experimental image-based freshness "
    "estimation system. It is not a food-safety test "
    "or a substitute for laboratory analysis."
)
