# ============================================================
# Contrastive Representation Learning for Satellite Imagery
# ------------------------------------------------------------
# This script trains a SimSiam-style self-supervised
# contrastive learning model on Landsat-8 satellite imagery.
#
# The learned embeddings are subsequently used as
# environmental covariates in Bayesian malaria prevalence
# modelling across sub-Saharan Africa.
#
# Key Components:
#   - Custom CNN encoder
#   - SimSiam predictor network
#   - Two stochastic augmentation pipelines
#   - Contrastive cosine similarity objective
#
# Input:
#   RGB Landsat-8 image tiles (224 × 224)
#
# Output:
#   Trained encoder + predictor weights
#
# ============================================================


# ============================================================
# Standard Imports
# ============================================================

import os
import cv2
import PIL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import tensorflow as tf
import tensorflow_addons as tfa


# ============================================================
# TensorFlow / GPU Configuration
# ============================================================

tf.keras.backend.clear_session()

print("Available GPUs:")
print(tf.config.list_physical_devices('GPU'))

devices = tf.config.experimental.list_physical_devices('GPU')

for gpu in devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# Multi-GPU training strategy
mirrored_strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0", "/gpu:1"]
)

print("Number of devices:", mirrored_strategy.num_replicas_in_sync)


# ============================================================
# Global Configuration
# ============================================================

AUTO = tf.data.AUTOTUNE

BATCH_SIZE = 64
EPOCHS = 50

CROPS_TO = 224
DIM = (224, 224)

SEED = 26

LR = 0.0001
WEIGHT_DECAY = 0.0005


# ============================================================
# File Paths
# ============================================================

images_folder = '/home/Landsat8/10k_5k_PNG_all/'

projection_weights = (
    '/home/contrastive_step_19_01/weights/'
    'projection_weights_15_05.h5'
)

prediction_weights = (
    '/home/contrastive_step_19_01/weights/'
    'prediction_weights_15_05.h5'
)

imgs = '/home/contrastive_step_19_01/raw_images_15_05.png'
aug1 = '/home/contrastive_step_19_01/aug1_15_05.png'
aug2 = '/home/contrastive_step_19_01/aug2_15_05.png'

contra = (
    '/home/contrastive_step_19_01/'
    'contrastive_15_05.png'
)


# ============================================================
# Verify Image Directory
# ============================================================

if not os.path.exists(images_folder):
    print('Images not found!!')
else:
    print('Images found')


# ============================================================
# Load Example Images
# ------------------------------------------------------------
# These images are used for visualisation and quality checks.
# ============================================================

image_filenames = [
    f for f in os.listdir(images_folder)
    if f.endswith('.png')
]

images = []

num_images = 100

for i, filename in enumerate(image_filenames):

    if i == num_images:
        break

    img = Image.open(
        os.path.join(images_folder, filename)
    )

    img_array = np.array(img)

    images.append(img_array)

images = np.array(images)

print("Image array shape:", images.shape)
print("Number of images:", len(images))
print("Image type:", type(images))
print("Single image shape:", images[5].shape)


# ============================================================
# Visualise Raw Images
# ============================================================

selected_images = images[49:]

fig, axes = plt.subplots(7, 7, figsize=(13, 13))

for i, ax in enumerate(axes.flat):

    ax.imshow(selected_images[i])
    ax.axis('off')

plt.savefig(imgs)
plt.show()


# ============================================================
# Augmentation Functions
# ------------------------------------------------------------
# Two different stochastic augmentation pipelines are used
# to create positive pairs for SimSiam contrastive learning.
# ============================================================

@tf.function
def bright(image):

    image = tf.image.adjust_brightness(
        image,
        delta=0.075
    )

    return image


# ------------------------------------------------------------
# Augmentation Pipeline 1
# ------------------------------------------------------------

@tf.function
def flip_vertical_random_crop(image):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_crop(
        image,
        (CROPS_TO, CROPS_TO, 3)
    )

    return image


@tf.function
def img_rot(image):

    image = tf.image.rot90(image, k=1)

    return image


@tf.function
def saturate(image):

    image = tf.image.adjust_saturation(
        image,
        1.25
    )

    return image


@tf.function
def contrast(image):

    image = tf.image.adjust_contrast(
        image,
        1.75
    )

    return image


@tf.function
def hue(image):

    image = tf.image.adjust_hue(
        image,
        -0.01
    )

    return image


@tf.function
def random_apply(func, x, p):

    if tf.random.uniform([], 0, 1) < p:
        return func(x)

    return x


@tf.function
def custom_augment1(image):

    image = bright(image)

    image = flip_vertical_random_crop(image)

    image = random_apply(img_rot, image, p=0.7)

    image = random_apply(saturate, image, p=0.7)

    image = random_apply(contrast, image, p=0.7)

    image = random_apply(hue, image, p=0.7)

    return image


# ------------------------------------------------------------
# Augmentation Pipeline 2
# ------------------------------------------------------------

@tf.function
def flip_horizontal_random_crop(image):

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_crop(
        image,
        (CROPS_TO, CROPS_TO, 3)
    )

    return image


@tf.function
def img_rot180(image):

    image = tf.image.rot90(image, k=2)

    return image


@tf.function
def saturate2(image):

    image = tf.image.adjust_saturation(
        image,
        0.75
    )

    return image


@tf.function
def contrast2(image):

    image = tf.image.adjust_contrast(
        image,
        2.5
    )

    return image


@tf.function
def hue2(image):

    image = tf.image.adjust_hue(
        image,
        0.01
    )

    return image


@tf.function
def custom_augment2(image):

    image = bright(image)

    image = flip_horizontal_random_crop(image)

    image = random_apply(img_rot180, image, p=0.7)

    image = random_apply(saturate2, image, p=0.7)

    image = random_apply(contrast2, image, p=0.7)

    image = random_apply(hue2, image, p=0.7)

    return image


# ============================================================
# Dataset Preprocessing
# ============================================================

image_paths = [
    os.path.join(images_folder, filename)
    for filename in os.listdir(images_folder)
]


def preprocess_image_dataset1(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = custom_augment1(image)

    return image


def preprocess_image_dataset2(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = custom_augment2(image)

    return image


# ============================================================
# TensorFlow Datasets
# ============================================================

dataset_one = tf.data.Dataset.from_tensor_slices(
    image_paths
)

dataset_one = (
    dataset_one
    .shuffle(1024, seed=0)
    .map(preprocess_image_dataset1)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

dataset_two = tf.data.Dataset.from_tensor_slices(
    image_paths
)

dataset_two = (
    dataset_two
    .shuffle(1024, seed=0)
    .map(preprocess_image_dataset2)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

print(type(dataset_one))
print(type(dataset_two))


# ============================================================
# Visualise Augmented Samples
# ============================================================

sample_images_one = next(iter(dataset_one))

plt.figure(figsize=(50, 50))

for n in range(16):

    ax = plt.subplot(4, 4, n + 1)

    plt.imshow(
        sample_images_one[n].numpy().astype("int")
    )

    plt.axis("off")

plt.savefig(aug1)
plt.show()


sample_images_two = next(iter(dataset_two))

plt.figure(figsize=(50, 50))

for n in range(16):

    ax = plt.subplot(4, 4, n + 1)

    plt.imshow(
        sample_images_two[n].numpy().astype("int")
    )

    plt.axis("off")

plt.savefig(aug2)
plt.show()


# ============================================================
# Encoder Network
# ------------------------------------------------------------
# Custom VGG-style convolutional encoder producing
# 2048-dimensional environmental embeddings.
# ============================================================

def get_encoder():

    inputs = tf.keras.layers.Input(
        (224, 224, 3),
        name='Inputs_BaseEncoder'
    )

    alpha = 0.2

    # ========================================================
    # Block 1
    # ========================================================

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv1_BaseEncoder'
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder1'
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv2_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder2'
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='Pool1_BaseEncoder'
    )(x)

    # ========================================================
    # Block 2
    # ========================================================

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv3_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder3'
    )(x)

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv4_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder4'
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='Pool2_BaseEncoder'
    )(x)

    # ========================================================
    # Block 3
    # ========================================================

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv5_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder5'
    )(x)

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv6_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder6'
    )(x)

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv7_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder7'
    )(x)

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv8_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder8'
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='Pool3_BaseEncoder'
    )(x)

    # ========================================================
    # Block 4
    # ========================================================

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv9_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder9'
    )(x)

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv10_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder10'
    )(x)

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv11_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder11'
    )(x)

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv12_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder12'
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='Pool4_BaseEncoder'
    )(x)

    # ========================================================
    # Block 5
    # ========================================================

    x = tf.keras.layers.Conv2D(
        1024,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv13_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder13'
    )(x)

    x = tf.keras.layers.Conv2D(
        1024,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv14_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder14'
    )(x)

    x = tf.keras.layers.Conv2D(
        1024,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv15_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder15'
    )(x)

    x = tf.keras.layers.Conv2D(
        1024,
        (3, 3),
        padding='same',
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Conv16_BaseEncoder'
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder16'
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name='Pool5_BaseEncoder'
    )(x)

    # ========================================================
    # Dense Projection Head
    # ========================================================

    x = tf.keras.layers.GlobalAveragePooling2D(
        name='GAP_BaseEncoder'
    )(x)

    x = tf.keras.layers.Dense(
        2048,
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Dense1_BaseEncoder',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            WEIGHT_DECAY
        )
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder17'
    )(x)

    x = tf.keras.layers.Dense(
        2048,
        activation=tf.keras.layers.LeakyReLU(alpha=alpha),
        name='Dense2_BaseEncoder',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            WEIGHT_DECAY
        )
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder18'
    )(x)

    z = tf.keras.layers.Dense(
        2048,
        name='Dense3_BaseEncoder',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            WEIGHT_DECAY
        )
    )(x)

    z = tf.keras.layers.BatchNormalization(
        name='BN_BaseEncoder19'
    )(z)

    f = tf.keras.Model(
        inputs,
        z,
        name='BaseEncoder'
    )

    return f


get_encoder().summary()


# ============================================================
# Predictor Network
# ------------------------------------------------------------
# SimSiam predictor head.
# ============================================================

def get_predictor():

    inputs = tf.keras.layers.Input(
        (2048,),
        name='Input_Predictor'
    )

    x = tf.keras.layers.Dense(
        512,
        activation='relu',
        name='Dense1_Predictor',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(
            WEIGHT_DECAY
        )
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name='BN_Predictor'
    )(x)

    p = tf.keras.layers.Dense(
        2048,
        name='Dense2_Predictor'
    )(x)

    h = tf.keras.Model(
        inputs,
        p,
        name='Predictor'
    )

    return h


get_predictor().summary()


# ============================================================
# SimSiam Loss Function
# ------------------------------------------------------------
# Negative cosine similarity between prediction vectors and
# stop-gradient target embeddings.
# ============================================================

def loss_func(p, z):

    z = tf.stop_gradient(z)

    p = tf.math.l2_normalize(p, axis=1)

    z = tf.math.l2_normalize(z, axis=1)

    return -tf.reduce_mean(
        tf.reduce_sum((p * z), axis=1)
    )


# ============================================================
# Single Training Step
# ============================================================

@tf.function
def train_step(ds_one, ds_two, f, h, optimizer):

    with tf.GradientTape() as tape:

        z1, z2 = f(ds_one), f(ds_two)

        p1, p2 = h(z1), h(z2)

        loss = (
            loss_func(p1, z2) / 2
            +
            loss_func(p2, z1) / 2
        )

    learnable_params = (
        f.trainable_variables
        +
        h.trainable_variables
    )

    gradients = tape.gradient(
        loss,
        learnable_params
    )

    optimizer.apply_gradients(
        zip(gradients, learnable_params)
    )

    return loss


# ============================================================
# Training Loop
# ============================================================

def train_simsiam(
    f,
    h,
    dataset_one,
    dataset_two,
    optimizer,
    epochs=200
):

    step_wise_loss = []

    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):

        for ds_one, ds_two in zip(
            dataset_one,
            dataset_two
        ):

            loss = train_step(
                ds_one,
                ds_two,
                f,
                h,
                optimizer
            )

            step_wise_loss.append(loss)

        epoch_wise_loss.append(
            np.mean(step_wise_loss)
        )

        if epoch % 5 == 0:

            print(
                "epoch: {} loss: {:.3f}".format(
                    epoch + 1,
                    np.mean(step_wise_loss)
                )
            )

    return epoch_wise_loss, f, h


# ============================================================
# Optimizer
# ============================================================

decay_steps = 500

lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=decay_steps
)

optimizer = tf.keras.optimizers.SGD(
    lr_decayed_fn,
    momentum=0.6
)


# ============================================================
# Train SimSiam Model
# ============================================================

f = get_encoder()

h = get_predictor()

epoch_wise_loss, f, h = train_simsiam(
    f,
    h,
    dataset_one,
    dataset_two,
    optimizer,
    epochs=EPOCHS
)


# ============================================================
# Plot Training Loss
# ============================================================

plt.plot(epoch_wise_loss)

plt.grid()

plt.savefig(contra)

plt.show()


# ============================================================
# Save Trained Weights
# ============================================================

f.save_weights(projection_weights)

h.save_weights(prediction_weights)

print('Training Finished')


# ============================================================
# Completion Marker
# ============================================================

fil = open('res.txt', 'w+')

fil.write('Contrastive step done')

fil.close()
