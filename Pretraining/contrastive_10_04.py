# ============================================================
# Contrastive Representation Learning for Satellite Imagery
# ------------------------------------------------------------
# This script trains a SimSiam-style contrastive learning model
# on Landsat-8 satellite imagery to generate high-dimensional
# environmental embeddings for malaria prediction.
#
# The learned representations are later used as covariates
# in a Bayesian malaria prevalence model.
#
# Architecture:
#   - Custom CNN encoder
#   - SimSiam self-supervised objective
#   - Symmetric contrastive prediction loss
#
# Input:
#   RGB satellite image tiles (224 × 224)
#
# Output:
#   2048-dimensional environmental embeddings
#
# ============================================================


# ============================================================
# Imports
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image


# ============================================================
# TensorFlow / GPU Configuration
# ============================================================

tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices("GPU")
print("Detected GPUs:", gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ============================================================
# Global Configuration
# ============================================================

AUTO = tf.data.AUTOTUNE

BATCH_SIZE = 64
EPOCHS = 50

IMAGE_SIZE = (224, 224)
CROP_SIZE = 224

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4

SEED = 26

# ============================================================
# File Paths
# ============================================================

IMAGES_FOLDER = "/home/Landsat8/10k_5k_PNG_all/"

PROJECTION_WEIGHTS = (
    "/home/contrastive_step_19_01/weights/"
    "projection_weights_15_05.h5"
)

PREDICTION_WEIGHTS = (
    "/home/contrastive_step_19_01/weights/"
    "prediction_weights_15_05.h5"
)

# Output visualisations
RAW_IMAGES_FIG = "/home/contrastive_step_19_01/raw_images_15_05.png"
AUGMENTATION_1_FIG = "/home/contrastive_step_19_01/aug1_15_05.png"
AUGMENTATION_2_FIG = "/home/contrastive_step_19_01/aug2_15_05.png"
LOSS_CURVE_FIG = "/home/contrastive_step_19_01/contrastive_15_05.png"


# ============================================================
# Verify Image Directory
# ============================================================

if not os.path.exists(IMAGES_FOLDER):
    raise FileNotFoundError("Satellite image directory not found.")

print("Satellite image directory found.")


# ============================================================
# Load Example Images for Visualisation
# ============================================================

image_filenames = [
    f for f in os.listdir(IMAGES_FOLDER)
    if f.endswith(".png")
]

images = []

NUM_EXAMPLE_IMAGES = 100

for i, filename in enumerate(image_filenames):

    if i >= NUM_EXAMPLE_IMAGES:
        break

    img = Image.open(os.path.join(IMAGES_FOLDER, filename))
    images.append(np.array(img))

images = np.array(images)

print("Loaded example images:", images.shape)


# ============================================================
# Visualise Example Images
# ============================================================

selected_images = images[49:]

fig, axes = plt.subplots(7, 7, figsize=(13, 13))

for i, ax in enumerate(axes.flat):
    ax.imshow(selected_images[i])
    ax.axis("off")

plt.savefig(RAW_IMAGES_FIG)
plt.show()


# ============================================================
# Data Augmentation Functions
# ============================================================

@tf.function
def adjust_brightness(image, delta=0.075):
    return tf.image.adjust_brightness(image, delta)


@tf.function
def random_crop_and_flip(image, horizontal=True):

    if horizontal:
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.random_flip_up_down(image)

    image = tf.image.random_crop(
        image,
        (CROP_SIZE, CROP_SIZE, 3)
    )

    return image


@tf.function
def rotate_image(image, k=1):
    return tf.image.rot90(image, k=k)


@tf.function
def adjust_saturation(image, factor):
    return tf.image.adjust_saturation(image, factor)


@tf.function
def adjust_contrast(image, factor):
    return tf.image.adjust_contrast(image, factor)


@tf.function
def adjust_hue(image, delta):
    return tf.image.adjust_hue(image, delta)


@tf.function
def random_apply(function, image, probability):

    if tf.random.uniform([], 0, 1) < probability:
        return function(image)

    return image


# ============================================================
# SimSiam Augmentation Pipelines
# ------------------------------------------------------------
# Two different stochastic augmentations are applied to the
# same image to create positive pairs for contrastive learning.
# ============================================================

@tf.function
def augmentation_pipeline_1(image):

    image = adjust_brightness(image)
    image = random_crop_and_flip(image, horizontal=True)

    image = random_apply(
        lambda x: rotate_image(x, k=1),
        image,
        probability=0.7
    )

    image = random_apply(
        lambda x: adjust_saturation(x, 1.25),
        image,
        probability=0.7
    )

    image = random_apply(
        lambda x: adjust_contrast(x, 1.75),
        image,
        probability=0.7
    )

    image = random_apply(
        lambda x: adjust_hue(x, -0.01),
        image,
        probability=0.7
    )

    return image


@tf.function
def augmentation_pipeline_2(image):

    image = adjust_brightness(image)
    image = random_crop_and_flip(image, horizontal=False)

    image = random_apply(
        lambda x: rotate_image(x, k=2),
        image,
        probability=0.7
    )

    image = random_apply(
        lambda x: adjust_saturation(x, 0.75),
        image,
        probability=0.7
    )

    image = random_apply(
        lambda x: adjust_contrast(x, 2.5),
        image,
        probability=0.7
    )

    image = random_apply(
        lambda x: adjust_hue(x, 0.01),
        image,
        probability=0.7
    )

    return image


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image_path, augmentation_function):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = augmentation_function(image)

    return image


# ============================================================
# Create TensorFlow Datasets
# ============================================================

image_paths = [
    os.path.join(IMAGES_FOLDER, filename)
    for filename in os.listdir(IMAGES_FOLDER)
]

dataset_one = (
    tf.data.Dataset
    .from_tensor_slices(image_paths)
    .shuffle(1024, seed=SEED)
    .map(
        lambda x: preprocess_image(
            x,
            augmentation_pipeline_1
        ),
        num_parallel_calls=AUTO
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

dataset_two = (
    tf.data.Dataset
    .from_tensor_slices(image_paths)
    .shuffle(1024, seed=SEED)
    .map(
        lambda x: preprocess_image(
            x,
            augmentation_pipeline_2
        ),
        num_parallel_calls=AUTO
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


# ============================================================
# Visualise Augmented Examples
# ============================================================

sample_images = next(iter(dataset_one))

plt.figure(figsize=(20, 20))

for n in range(16):

    ax = plt.subplot(4, 4, n + 1)

    plt.imshow(sample_images[n].numpy().astype("int"))
    plt.axis("off")

plt.savefig(AUGMENTATION_1_FIG)
plt.show()


# ============================================================
# Encoder Network
# ------------------------------------------------------------
# Custom VGG-style convolutional encoder producing
# 2048-dimensional environmental embeddings.
# ============================================================

def build_encoder():

    alpha = 0.2

    inputs = tf.keras.layers.Input(
        (224, 224, 3),
        name="Encoder_Input"
    )

    x = inputs

    # Additional blocks omitted here for brevity
    # (retain your existing architecture)

    # Final embedding
    embeddings = tf.keras.layers.Dense(
        2048,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            WEIGHT_DECAY
        ),
        name="Embedding_Layer"
    )(x)

    model = tf.keras.Model(
        inputs,
        embeddings,
        name="Encoder"
    )

    return model


# ============================================================
# Predictor Network
# ------------------------------------------------------------
# SimSiam projection/prediction head.
# ============================================================

def build_predictor():

    inputs = tf.keras.layers.Input(
        (2048,),
        name="Predictor_Input"
    )

    x = tf.keras.layers.Dense(
        512,
        activation="relu",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            WEIGHT_DECAY
        )
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(
        2048
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="Predictor"
    )

    return model


# ============================================================
# SimSiam Loss Function
# ------------------------------------------------------------
# Negative cosine similarity between prediction vectors
# and stop-gradient target embeddings.
# ============================================================

def simsiam_loss(predictions, targets):

    targets = tf.stop_gradient(targets)

    predictions = tf.math.l2_normalize(
        predictions,
        axis=1
    )

    targets = tf.math.l2_normalize(
        targets,
        axis=1
    )

    similarity = tf.reduce_sum(
        predictions * targets,
        axis=1
    )

    return -tf.reduce_mean(similarity)


# ============================================================
# Single Training Step
# ============================================================

@tf.function
def train_step(
    batch_one,
    batch_two,
    encoder,
    predictor,
    optimizer
):

    with tf.GradientTape() as tape:

        z1 = encoder(batch_one)
        z2 = encoder(batch_two)

        p1 = predictor(z1)
        p2 = predictor(z2)

        loss = (
            simsiam_loss(p1, z2) / 2
            +
            simsiam_loss(p2, z1) / 2
        )

    variables = (
        encoder.trainable_variables
        +
        predictor.trainable_variables
    )

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(
        zip(gradients, variables)
    )

    return loss


# ============================================================
# Training Loop
# ============================================================

def train_simsiam(
    encoder,
    predictor,
    dataset_one,
    dataset_two,
    optimizer,
    epochs=50
):

    epoch_losses = []

    for epoch in tqdm(range(epochs)):

        batch_losses = []

        for batch_one, batch_two in zip(
            dataset_one,
            dataset_two
        ):

            loss = train_step(
                batch_one,
                batch_two,
                encoder,
                predictor,
                optimizer
            )

            batch_losses.append(loss.numpy())

        epoch_loss = np.mean(batch_losses)

        epoch_losses.append(epoch_loss)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Loss: {epoch_loss:.4f}"
        )

    return epoch_losses


# ============================================================
# Optimizer
# ============================================================

lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=500
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_schedule,
    momentum=0.6
)


# ============================================================
# Train Model
# ============================================================

encoder = build_encoder()
predictor = build_predictor()

loss_history = train_simsiam(
    encoder,
    predictor,
    dataset_one,
    dataset_two,
    optimizer,
    epochs=EPOCHS
)


# ============================================================
# Plot Training Loss
# ============================================================

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("SimSiam Loss")

plt.grid()

plt.savefig(LOSS_CURVE_FIG)

plt.show()


# ============================================================
# Save Weights
# ============================================================

encoder.save_weights(PROJECTION_WEIGHTS)
predictor.save_weights(PREDICTION_WEIGHTS)

print("Training complete.")


# ============================================================
# Completion Marker
# ============================================================

with open("res.txt", "w+") as file:
    file.write("Contrastive training complete")
