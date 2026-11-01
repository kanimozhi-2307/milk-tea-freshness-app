
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


# ------------------------------------------
# TENSORFLOW COMPATIBILITY
# ------------------------------------------

# Compatibility support for Lambda layers in saved models
try:
    import keras.src.utils.python_utils as python_utils
    python_utils.tf = tf
except (ImportError, AttributeError):
    pass


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
# LOAD GEMINI CLIENT
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

    image_buffer = io.BytesIO()

    image.convert("RGB").save(
        image_buffer,
        format="JPEG",
        quality=90
    )

    image_bytes = image_buffer.getvalue()

    prompt = """
    You are an image verification system for a brewed milk tea
    freshness estimation application.

    Examine the image and classify it into exactly one category.

    MILK_TEA:
    Select this only if a prepared brewed milk tea beverage
    is visibly present in a cup, glass, or container.

    NOT_MILK_TEA:
    Select this for coffee, water, juice, plain milk,
    other beverages, food, people, animals, objects,
    empty cups, tea leaves, tea powder, or unrelated images.

    UNCERTAIN:
    Select this if the image is blurry, too dark, unclear,
    or does not provide enough visual evidence.

    Important rules:
    - Do not assume every brown or beige liquid is milk tea.
    - Do not classify tea leaves or tea powder as brewed milk tea.
    - Do not predict freshness.
    - Do not assess food safety.

    Return only one exact label:
    MILK_TEA
    NOT_MILK_TEA
    UNCERTAIN
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            genai.types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        config=genai.types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=30
        )
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    result = response.text.strip().upper()

    match = re.search(
        r"\b(NOT_MILK_TEA|MILK_TEA|UNCERTAIN)\b",
        result
    )

    if match is None:
        raise ValueError(
            "Gemini returned an unexpected result. "
            "Please try again."
        )

    return match.group(1)


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

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    return model


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

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "Model output does not contain exactly three classes. "
            "Please verify the model output and class order."
        )

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
    brewed milk tea and then estimates its visual freshness level.
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

    except Exception:

        st.error(
            "Unable to read this image. "
            "Please upload a valid JPG or PNG image."
        )

        st.stop()

    st.image(
        image,
        caption="Selected Image",
        width=400
    )

    if st.button(
        "Predict Freshness",
        type="primary"
    ):

        # STEP 1: GEMINI IMAGE VERIFICATION

        try:

            with st.spinner(
                "Gemini AI is checking the uploaded image..."
            ):

                gemini_client = load_gemini_client()

                classification = detect_tea(
                    image,
                    gemini_client
                )

        except Exception as error:

            st.error(
                "Image verification failed. "
                "Please check your Gemini API key, model access "
                "and internet connection."
            )

            st.caption(str(error))

            st.stop()

        st.divider()

        st.subheader("AI Image Verification")

        # STEP 2: HANDLE NON-TEA IMAGE

        if classification == "NOT_MILK_TEA":

            st.error("🚫 TEA NOT FOUND")

            st.warning(
                "The uploaded image does not appear to contain "
                "brewed milk tea. Please upload a clear milk tea image."
            )

            st.stop()

        # STEP 3: HANDLE UNCLEAR IMAGE

        elif classification == "UNCERTAIN":

            st.warning(
                "⚠️ IMAGE COULD NOT BE VERIFIED"
            )

            st.info(
                "Please upload a clear, well-lit image "
                "of brewed milk tea."
            )

            st.stop()

        # STEP 4: MILK TEA DETECTED

        st.success("✅ MILK TEA DETECTED")

        # STEP 5: LOAD FRESHNESS MODEL AND PREDICT

        try:

            with st.spinner(
                "Loading freshness model and predicting..."
            ):

                model = load_freshness_model()

                predicted_class, confidence, predictions = (
                    predict_freshness(image, model)
                )

        except Exception as error:

            st.error(
                "Freshness prediction failed. "
                "Please check your model file and deployment logs."
            )

            st.caption(str(error))

            st.stop()

        # STEP 6: DISPLAY FRESHNESS RESULT

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

        # STEP 7: CLASS-WISE SCORES

        st.subheader("Class-wise Prediction Scores")

        for index, class_name in enumerate(CLASS_NAMES):

            probability = float(
                predictions[index]
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

    **Image Size:** 224 × 224 pixels

    **Output:** FRESH, MODERATE, LOW, or TEA NOT FOUND
    """
)

st.info(
    "This application estimates visual freshness patterns "
    "learned from the dataset. It is not a laboratory test "
    "or a food-safety assessment."
)

st.caption(
    "Developed for the brewed milk tea freshness estimation project."
)
