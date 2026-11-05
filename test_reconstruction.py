
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

DATASET_DIR = "dataset/english-handwritten-characters-dataset"

print("Loading models...")

encoder = tf.keras.models.load_model(
    "models/handwriting_encoder.keras",
    compile=False
)

decoder = tf.keras.models.load_model(
    "models/handwriting_model.keras",
    compile=False
)

with open("models/label_mapping.json", "r") as f:
    char_to_id = json.load(f)

# Load one actual dataset image.
df = pd.read_csv(os.path.join(DATASET_DIR, "english.csv"))

row = df.iloc[0]

image_path = os.path.join(DATASET_DIR, row["image"])

image = Image.open(image_path).convert("L")
image = image.resize((64, 64), Image.Resampling.LANCZOS)

original = np.asarray(image, dtype=np.float32) / 255.0

# Convert black strokes to foreground value 1.
processed = 1.0 - original

character = str(row["label"])
character_id = char_to_id[character]

image_input = processed[np.newaxis, :, :, np.newaxis]

character_input = np.array(
    [[character_id]], dtype=np.int32
)

# Encode the actual image.
mean, log_var = encoder.predict(
    [image_input, character_input],
    verbose=0
)

print("\nLatent mean range:", mean.min(), mean.max())
print("Latent mean average:", mean.mean())
print("Latent log variance:", log_var.mean())

# Reconstruct using the encoded latent vector.
reconstruction = decoder.predict(
    [character_input, mean],
    verbose=0
)[0, :, :, 0]

print("\nReconstruction statistics:")
print("Minimum:", reconstruction.min())
print("Maximum:", reconstruction.max())
print("Mean:", reconstruction.mean())
print("Standard deviation:", reconstruction.std())

# Display original and reconstructed images.
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original, cmap="gray")
axes[0].set_title("Original Image")

axes[1].imshow(1.0 - processed, cmap="gray")
axes[1].set_title("Processed Image")

axes[2].imshow(1.0 - reconstruction, cmap="gray")
axes[2].set_title("Reconstructed Image")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig("results/reconstruction_test.png")
plt.show()
