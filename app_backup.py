
import os
import json
import random
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageChops
from io import BytesIO


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Handwritten Text Generator",
    page_icon="✍️",
    layout="wide"
)

DATASET_DIR = "dataset/english-handwritten-characters-dataset"
MODEL_DIR = "models"


# ---------------- LOAD TRAINED MODELS ----------------

@st.cache_resource
def load_resources():

    encoder = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "handwriting_encoder.keras"),
        compile=False
    )

    decoder = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "handwriting_model.keras"),
        compile=False
    )

    with open(os.path.join(MODEL_DIR, "label_mapping.json"), "r") as f:
        char_to_id = json.load(f)

    df = pd.read_csv(
        os.path.join(DATASET_DIR, "english.csv")
    )

    df["label"] = df["label"].astype(str)

    return encoder, decoder, char_to_id, df


# ---------------- CROP CHARACTER ----------------

def crop_character(image):

    background = Image.new("L", image.size, 255)

    difference = ImageChops.difference(image, background)

    bbox = difference.point(
        lambda pixel: 255 if pixel > 20 else 0
    ).getbbox()

    if bbox:
        image = image.crop(bbox)

    return image


# ---------------- GENERATE CHARACTER ----------------

def generate_character(character, encoder, decoder, char_to_id, df):

    matching_rows = df[df["label"] == character]

    if matching_rows.empty:
        return None

    row = matching_rows.sample(n=1).iloc[0]

    image_path = os.path.join(DATASET_DIR, row["image"])

    image = Image.open(image_path).convert("L")
    image = image.resize(
        (64, 64),
        Image.Resampling.LANCZOS
    )

    original = np.asarray(image, dtype=np.float32) / 255.0

    processed = 1.0 - original

    image_input = processed[np.newaxis, :, :, np.newaxis]

    character_id = char_to_id[character]

    character_input = np.array(
        [[character_id]],
        dtype=np.int32
    )

    mean, log_var = encoder.predict(
        [image_input, character_input],
        verbose=0
    )

    generated = decoder.predict(
        [character_input, mean],
        verbose=0
    )[0, :, :, 0]

    generated = np.clip(1.0 - generated, 0, 1)

    generated = (generated * 255).astype(np.uint8)

    generated_image = Image.fromarray(generated)

    generated_image = crop_character(generated_image)

    return generated_image


# ---------------- GENERATE A LINE ----------------

def generate_line(
    text,
    encoder,
    decoder,
    char_to_id,
    df,
    spacing
):

    character_images = []

    for character in text:

        if character == " ":
            character_images.append(None)
            continue

        if character not in char_to_id:
            continue

        image = generate_character(
            character,
            encoder,
            decoder,
            char_to_id,
            df
        )

        if image is None:
            continue

        # Natural character size variation
        scale = random.uniform(0.90, 1.08)

        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        character_images.append(image)

    if not character_images:
        return Image.new("L", (100, 80), 255)

    word_spacing = 22

    total_width = sum(
        word_spacing if image is None
        else image.width + spacing
        for image in character_images
    )

    canvas = Image.new(
        "L",
        (max(total_width, 100), 100),
        255
    )

    x_position = 0

    for image in character_images:

        if image is None:
            x_position += word_spacing
            continue

        # Slight natural baseline variation
        y_position = random.randint(20, 27)

        canvas.paste(
            image,
            (x_position, y_position)
        )

        x_position += image.width + spacing

    return canvas


# ---------------- GENERATE COMPLETE TEXT ----------------

def generate_handwriting(
    text,
    encoder,
    decoder,
    char_to_id,
    df,
    spacing
):

    lines = text.splitlines()

    if not lines:
        lines = [text]

    generated_lines = []

    for line in lines:

        generated_image = generate_line(
            line,
            encoder,
            decoder,
            char_to_id,
            df,
            spacing
        )

        generated_lines.append(generated_image)

    final_width = max(
        image.width for image in generated_lines
    )

    line_height = 110

    final_image = Image.new(
        "L",
        (
            final_width,
            len(generated_lines) * line_height
        ),
        255
    )

    for index, image in enumerate(generated_lines):

        final_image.paste(
            image,
            (0, index * line_height)
        )

    return final_image


# ---------------- APPLICATION INTERFACE ----------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #777777;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .stButton button {
        border-radius: 8px;
        height: 48px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">✍️ Handwritten Text Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Transform your typed words into handwritten-style text using Deep Learning.</div>',
    unsafe_allow_html=True
)

st.divider()

left, right = st.columns([1.2, 0.8], gap="large")


with left:

    st.subheader("Enter Your Text")

    user_text = st.text_area(
        "Your text",
        value="Hello World!\nWelcome to Handwritten Text Generation.",
        height=180,
        max_chars=250,
        placeholder="Type your text here..."
    )

    spacing = st.slider(
        "Character spacing",
        min_value=0,
        max_value=8,
        value=2
    )

    generate_button = st.button(
        "✨ Generate Handwriting",
        type="primary",
        use_container_width=True
    )


with right:

    st.subheader("About the Project")

    st.write(
        "This application uses a Conditional Variational "
        "Autoencoder (CVAE) trained on handwritten character images."
    )

    st.markdown(
        """
        **Features**

        - Handwritten character generation
        - Text-to-image conversion
        - Natural character spacing
        - Multiple-line text support
        - PNG image download
        """
    )


# ---------------- GENERATION OUTPUT ----------------

if generate_button:

    if not user_text.strip():

        st.warning("Please enter some text first.")

    else:

        try:

            with st.spinner(
                "Generating your handwritten text..."
            ):

                encoder, decoder, char_to_id, df = load_resources()

                generated_image = generate_handwriting(
                    user_text,
                    encoder,
                    decoder,
                    char_to_id,
                    df,
                    spacing
                )

            st.success("Handwriting generated successfully!")

            st.divider()

            st.subheader("Your Generated Handwriting")

            st.image(
                generated_image,
                use_container_width=True
            )

            image_bytes = BytesIO()

            generated_image.save(
                image_bytes,
                format="PNG"
            )

            st.download_button(
                label="⬇️ Download Handwriting Image",
                data=image_bytes.getvalue(),
                file_name="generated_handwriting.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )


st.divider()

st.markdown(
    "<center>CodSoft Machine Learning Internship | Handwritten Text Generation</center>",
    unsafe_allow_html=True
)
