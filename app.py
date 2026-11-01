
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import io
import streamlit as st
import numpy as np
import tensorflow as tf
import keras

from PIL import Image
from google import genai
from google.genai import types


# ------------------------------------------
# TENSORFLOW / KERAS COMPATIBILITY
# ------------------------------------------

# Compatibility support for older TensorFlow Lambda models
try:
    import keras.src.utils.python_utils as python_utils
    python_utils.tf = tf
except (ImportError, AttributeError):
    pass


# ------------------------------------------
# PROJECT CONFIGURATION
# ------------------------------------------

REPO_ID = "kanimozhi23/milk-tea-freshness-model"

GEMINI_MODEL = "gemini-2.5-flash-lite"

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
            "GEMINI_API_KEY is missing. "
            "Add your API key in Streamlit Settings → Secrets."
        )

    client = genai.Client(api_key=api_key)

    return client


# ------------------------------------------
# GEMINI AI TEA DETECTION
# ------------------------------------------

def detect_tea(image, client):

    # Convert image into JPEG bytes
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

    Examine the provided image and classify it into exactly
    ONE of these three categories:

    MILK_TEA:
    Use this only when a prepared brewed milk tea beverage
    is visibly present in the image.

    NOT_MILK_TEA:
    Use this for coffee, water, juice, plain milk, other drinks,
    food, people, animals, objects, empty cups, tea leaves,
    tea powder, or unrelated images.

    UNCERTAIN:
    Use this when the image is blurry, too dark, unclear,
    or there is insufficient visual evidence.

    Important:
    - Do not assume every brown or beige liquid is milk tea.
    - Do not classify tea leaves or tea powder as brewed milk tea.
    - Do not estimate freshness.
    - Do not assess food safety.

    Return ONLY one exact label:
    MILK_TEA
    NOT_MILK_TEA
    or
    UNCERTAIN
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=20
        )
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response. Please try again."
        )

    result = response.text.strip().upper()

    # Extract the expected classification
    if "NOT_MILK_TEA" in result:
        classification = "NOT_MILK_TEA"

    elif "UNCERTAIN" in result:
        classification = "UNCERTAIN"

    elif "MILK_TEA" in result:
        classification = "MILK_TEA"

    else:
        raise ValueError(
            "Gemini returned an unexpected classification. "
            "Please try the image again."
        )

    return classification


# ------------------------------------------
# LOAD EXISTING FRESHNESS MODEL
# ------------------------------------------

@st.cache_resource
def load_freshness_model():

    model = keras.saving.load_model(
        "hf://" + REPO_ID,
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
            "The model output does not match the three "
            "freshness classes. Check the model architecture."
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
# FRESHNESS CATEGORIES
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
# IMAGE DISPLAY AND PREDICTION
# ------------------------------------------

if uploaded_image is not None:

    try:
        image = Image.open(
            uploaded_image
        ).convert("RGB")

    except Exception:
        st.error(
            "Unable to read the image. "
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

        # STEP 1: CHECK IMAGE USING GEMINI

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
                "Please try again."
            )

            st.caption(str(error))

            st.stop()

        st.divider()

        st.subheader("AI Image Verification")

        if classification == "NOT_MILK_TEA":

            st.error("🚫 TEA NOT FOUND")

            st.warning(
                "The uploaded image does not appear to contain "
                "brewed milk tea. Please upload a clear milk tea image."
            )

        elif classification == "UNCERTAIN":

            st.warning("⚠️ IMAGE COULD NOT BE VERIFIED")

            st.info(
                "Please upload a clearer, well-lit image "
                "of brewed milk tea."
            )

        else:

            # STEP 2: MILK TEA DETECTED

            st.success("✅ MILK TEA DETECTED")

            # STEP 3: FRESHNESS PREDICTION

            try:

                with st.spinner(
                    "Loading the freshness model and predicting..."
                ):

                    model = load_freshness_model()

                    predicted_class, confidence, predictions = (
                        predict_freshness(image, model)
                    )

            except Exception as error:

                st.error(
                    "Freshness prediction failed. "
                    "Please check the model repository and deployment logs."
                )

                st.caption(str(error))

                st.stop()

            # STEP 4: DISPLAY RESULT

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

            # STEP 5: CLASS-WISE SCORES

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
    "Developed for the brewed milk tea freshness "
    "estimation research project."
)
