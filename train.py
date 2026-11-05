
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input

# -------------------------------
# CONFIGURATION
# -------------------------------

DATASET_DIR = "dataset/english-handwritten-characters-dataset"
CSV_PATH = os.path.join(DATASET_DIR, "english.csv")

MODEL_DIR = "models"
RESULT_DIR = "results"

IMAGE_SIZE = 64
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 45
LEARNING_RATE = 0.0005

BETA = 0.001
STROKE_WEIGHT = 10.0

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# -------------------------------
# LOAD DATASET
# -------------------------------

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

if not {"image", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain image and label columns.")

characters = list(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

raw_labels = df["label"].astype(str).unique().tolist()

if all(label.isdigit() for label in raw_labels):

    numeric_labels = sorted(int(label) for label in raw_labels)

    if numeric_labels != list(range(len(numeric_labels))):
        raise ValueError("Numeric labels must start at zero and be continuous.")

    if len(numeric_labels) > len(characters):
        raise ValueError("Too many character labels.")

    label_to_character = {
        str(index): characters[index]
        for index in numeric_labels
    }

else:

    label_to_character = {
        label: label for label in raw_labels
    }

    if not all(label in characters for label in raw_labels):
        raise ValueError("Unexpected character labels.")

ordered_characters = sorted(
    set(label_to_character.values()),
    key=lambda x: characters.index(x)
)

char_to_id = {
    character: index
    for index, character in enumerate(ordered_characters)
}

df["character"] = df["label"].astype(str).map(label_to_character)
df["character_id"] = df["character"].map(char_to_id)

with open(
    os.path.join(MODEL_DIR, "label_mapping.json"), "w"
) as file:
    json.dump(char_to_id, file, indent=4)

print("Total records:", len(df))
print("Number of characters:", len(char_to_id))

# -------------------------------
# LOAD AND PREPROCESS IMAGES
# -------------------------------

print("\nLoading handwritten images...")

images = []
labels = []
failed_images = 0

for _, row in df.iterrows():

    image_path = os.path.join(DATASET_DIR, str(row["image"]))

    try:

        image = Image.open(image_path).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        # Convert black strokes into foreground value 1.
        # White background becomes value 0.
        image_array = 1.0 - image_array

        images.append(image_array)
        labels.append(int(row["character_id"]))

    except Exception as error:

        failed_images += 1
        print("Could not load:", image_path, error)

X = np.array(images, dtype=np.float32)[..., np.newaxis]
y = np.array(labels, dtype=np.int32).reshape(-1, 1)

print("Successfully loaded:", len(X))
print("Failed images:", failed_images)
print("Image shape:", X.shape)

# Display preview as normal black handwriting on white paper.
fig, axes = plt.subplots(2, 8, figsize=(12, 4))

for i, ax in enumerate(axes.flat):

    if i < len(X):
        ax.imshow(1.0 - X[i].squeeze(), cmap="gray", vmin=0, vmax=1)

    ax.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "dataset_preview.png")
)
plt.close()

# -------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train), seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# -------------------------------
# BUILD CVAE ENCODER
# -------------------------------

print("\nBuilding CVAE encoder...")

num_characters = len(char_to_id)

image_input = Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    name="image_input"
)

character_input = Input(
    shape=(1,),
    dtype="int32",
    name="encoder_character_id"
)

x = layers.Conv2D(
    32, 3, strides=2, padding="same", activation="relu"
)(image_input)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    64, 3, strides=2, padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    128, 3, strides=2, padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)

character_embedding = layers.Embedding(
    num_characters, 16
)(character_input)

character_embedding = layers.Flatten()(character_embedding)

character_features = layers.Dense(
    8 * 8 * 16, activation="relu"
)(character_embedding)

character_features = layers.Reshape(
    (8, 8, 16)
)(character_features)

x = layers.Concatenate()([x, character_features])

x = layers.Conv2D(
    128, 3, padding="same", activation="relu"
)(x)

x = layers.Flatten()(x)

x = layers.Dense(256, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)

encoder = Model(
    [image_input, character_input],
    [z_mean, z_log_var],
    name="cvae_encoder"
)

# -------------------------------
# BUILD CVAE DECODER
# -------------------------------

print("Building CVAE decoder...")

decoder_character_input = Input(
    shape=(1,),
    dtype="int32",
    name="character_id"
)

latent_input = Input(
    shape=(LATENT_DIM,),
    name="latent_vector"
)

char_features = layers.Embedding(
    num_characters, 32
)(decoder_character_input)

char_features = layers.Flatten()(char_features)

char_features = layers.Dense(
    8 * 8 * 16, activation="relu"
)(char_features)

latent_features = layers.Dense(
    8 * 8 * 32, activation="relu"
)(latent_input)

char_features = layers.Reshape(
    (8, 8, 16)
)(char_features)

latent_features = layers.Reshape(
    (8, 8, 32)
)(latent_features)

x = layers.Concatenate()(
    [char_features, latent_features]
)

x = layers.Conv2D(
    128, 3, padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(
    128, 4, strides=2, padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(
    64, 4, strides=2, padding="same", activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(
    32, 4, strides=2, padding="same", activation="relu"
)(x)

output_image = layers.Conv2D(
    1, 3, padding="same", activation="sigmoid",
    name="generated_image"
)(x)

decoder = Model(
    [decoder_character_input, latent_input],
    output_image,
    name="cvae_decoder"
)

# -------------------------------
# WEIGHTED RECONSTRUCTION LOSS
# -------------------------------

def calculate_reconstruction_loss(original, reconstructed):

    epsilon = 1e-7

    reconstructed = tf.clip_by_value(
        reconstructed,
        epsilon,
        1.0 - epsilon
    )

    # Give handwriting strokes more importance.
    pixel_loss = -(
        STROKE_WEIGHT * original * tf.math.log(reconstructed)
        +
        (1.0 - original) * tf.math.log(1.0 - reconstructed)
    )

    return tf.reduce_mean(
        tf.reduce_sum(pixel_loss, axis=[1, 2, 3])
    )


# -------------------------------
# TRAINING SETUP
# -------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


@tf.function
def train_step(images, character_ids):

    with tf.GradientTape() as tape:

        mean, log_var = encoder(
            [images, character_ids],
            training=True
        )

        epsilon = tf.random.normal(tf.shape(mean))

        z = mean + tf.exp(0.5 * log_var) * epsilon

        reconstructed = decoder(
            [character_ids, z],
            training=True
        )

        reconstruction_loss = calculate_reconstruction_loss(
            images, reconstructed
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + log_var
                - tf.square(mean)
                - tf.exp(log_var),
                axis=1
            )
        )

        total_loss = reconstruction_loss + BETA * kl_loss

    variables = (
        encoder.trainable_variables
        + decoder.trainable_variables
    )

    gradients = tape.gradient(total_loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return total_loss, reconstruction_loss, kl_loss


@tf.function
def validation_step(images, character_ids):

    mean, log_var = encoder(
        [images, character_ids],
        training=False
    )

    reconstructed = decoder(
        [character_ids, mean],
        training=False
    )

    reconstruction_loss = calculate_reconstruction_loss(
        images, reconstructed
    )

    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + log_var
            - tf.square(mean)
            - tf.exp(log_var),
            axis=1
        )
    )

    return reconstruction_loss + BETA * kl_loss


# -------------------------------
# TRAINING LOOP
# -------------------------------

print("\nStarting improved CVAE training...")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Epochs:", EPOCHS)

best_val_loss = float("inf")
patience = 8
wait = 0

history = {
    "loss": [],
    "val_loss": [],
    "reconstruction_loss": [],
    "kl_loss": []
}

for epoch in range(EPOCHS):

    train_losses = []
    reconstruction_losses = []
    kl_losses = []

    for batch_images, batch_labels in train_dataset:

        total, reconstruction, kl = train_step(
            batch_images, batch_labels
        )

        train_losses.append(float(total))
        reconstruction_losses.append(float(reconstruction))
        kl_losses.append(float(kl))

    val_losses = []

    for batch_images, batch_labels in val_dataset:

        val_loss = validation_step(
            batch_images, batch_labels
        )

        val_losses.append(float(val_loss))

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    reconstruction_loss = np.mean(reconstruction_losses)
    kl_loss = np.mean(kl_losses)

    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["reconstruction_loss"].append(reconstruction_loss)
    history["kl_loss"].append(kl_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Reconstruction: {reconstruction_loss:.4f} | "
        f"KL: {kl_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        wait = 0

        decoder.save(
            os.path.join(MODEL_DIR, "handwriting_model.keras")
        )

        encoder.save(
            os.path.join(MODEL_DIR, "handwriting_encoder.keras")
        )

        print("Best model saved.")

    else:

        wait += 1

        if wait >= patience:
            print("\nEarly stopping activated.")
            break

# -------------------------------
# SAVE TRAINING HISTORY
# -------------------------------

pd.DataFrame(history).to_csv(
    os.path.join(RESULT_DIR, "training_history.csv"),
    index=False
)

plt.figure(figsize=(9, 5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved CVAE Training Performance")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(RESULT_DIR, "training_loss.png")
)

plt.close()

print("\nTraining completed successfully!")
print("Best validation loss:", best_val_loss)
print("Decoder:", MODEL_DIR + "/handwriting_model.keras")
print("Encoder:", MODEL_DIR + "/handwriting_encoder.keras")
print("Mapping:", MODEL_DIR + "/label_mapping.json")
