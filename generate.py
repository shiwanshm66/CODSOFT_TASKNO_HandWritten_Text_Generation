
import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

DATASET_DIR = "dataset/english-handwritten-characters-dataset"
MODEL_DIR = "models"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading trained models...")

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

df = pd.read_csv(os.path.join(DATASET_DIR, "english.csv"))
df["label"] = df["label"].astype(str)

def generate_character(character):

    matching_rows = df[df["label"] == character]

    if matching_rows.empty:
        print("Character not found:", character)
        return Image.new("L", (64, 64), 255)

    row = matching_rows.sample(n=1).iloc[0]

    image_path = os.path.join(DATASET_DIR, row["image"])

    image = Image.open(image_path).convert("L")
    image = image.resize((64, 64), Image.Resampling.LANCZOS)

    original = np.asarray(image, dtype=np.float32) / 255.0
    processed = 1.0 - original

    image_input = processed[np.newaxis, :, :, np.newaxis]

    character_id = char_to_id[character]
    character_input = np.array([[character_id]], dtype=np.int32)

    mean, log_var = encoder.predict(
        [image_input, character_input],
        verbose=0
    )

    latent = mean

    generated = decoder.predict(
        [character_input, latent],
        verbose=0
    )[0, :, :, 0]

    generated = np.clip(1.0 - generated, 0, 1)
    generated = (generated * 255).astype(np.uint8)

    return Image.fromarray(generated)


def generate_text(text):

    images = []

    print("\nGenerating text:", text)

    for character in text:

        if character == " ":
            images.append(None)
            continue

        if character not in char_to_id:
            print("Skipping unsupported character:", character)
            continue

        generated_image = generate_character(character)
        images.append(generated_image)

    spacing = 10
    space_width = 28

    total_width = sum(
        space_width if img is None else 64 + spacing
        for img in images
    )

    total_width = max(total_width, 64)

    canvas = Image.new(
        "L",
        (total_width, 80),
        255
    )

    x_position = 0

    for img in images:

        if img is None:
            x_position += space_width
            continue

        canvas.paste(img, (x_position, 8))
        x_position += 64 + spacing

    output_path = os.path.join(
        OUTPUT_DIR,
        "generated_handwriting.png"
    )

    canvas.save(output_path)

    print("\nText generated successfully!")
    print("Saved at:", output_path)

    canvas.show()


if __name__ == "__main__":

    print("\nAvailable characters:")
    print("".join(char_to_id.keys()))

    user_text = input(
        "\nEnter the text you want to generate: "
    )

    generate_text(user_text)
