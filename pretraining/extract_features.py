# ============================================================
# Feature Extraction from Trained Contrastive Encoder
# ------------------------------------------------------------
# This script loads a pretrained SimSiam-style encoder and
# extracts high-dimensional environmental embeddings from
# Landsat-8 satellite imagery.
#
# The extracted embeddings are later used as covariates in
# Bayesian malaria prevalence modelling.
#
# Input:
#   - PNG satellite image tiles (224 × 224 RGB)
#   - Pretrained encoder weights
#   - CSV file containing image filenames + coordinates
#
# Output:
#   - CSV file containing:
#         filename
#         latitude
#         longitude
#         learned feature embeddings
#
# ============================================================


# ============================================================
# Standard Imports
# ============================================================

import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from PIL import Image
import PIL

import sklearn


# ============================================================
# TensorFlow / GPU Configuration
# ============================================================

print("TensorFlow Version:", tf.__version__)

# Use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("Available GPUs:")
print(tf.config.list_physical_devices('GPU'))

devices = tf.config.experimental.list_physical_devices('GPU')

print(devices)

# Enable memory growth
for gpu in devices:
    tf.config.experimental.set_memory_growth(
        gpu,
        True
    )


# ============================================================
# Global Configuration
# ============================================================

BS = 64
EPOCHS = 300
STEPS_PER_EPOCH = 200

LR = 0.001
WEIGHT_DECAY = 0.0005


# ============================================================
# File Paths
# ============================================================

# CSV containing image filenames + coordinates
csv_df = (
    '/home/contrastive_step_19_01/features/'
    '5k_PNG_2905_top_latlons.csv'
)

# Folder containing PNG satellite imagery
images_path = '/home/Landsat8/10k_5k_PNG_all/'

# Trained encoder weights from contrastive learning
pretrained_weights = (
    '/home/contrastive_step_19_01/weights/'
    'projection_weights_15_05.h5'
)

# Output CSV containing extracted features
features_csv1 = (
    '/home/contrastive_step_19_01/features/'
    'full_features_5km_27_05_top.csv'
)


# ============================================================
# Verify Image Directory
# ============================================================

print('Before reading images path')

image_filenames = [
    f for f in os.listdir(images_path)
    if f.endswith('.png')
]

image_paths = [
    os.path.join(images_path, filename)
    for filename in os.listdir(images_path)
]

print('Images found')


# ============================================================
# Load Metadata CSV
# ------------------------------------------------------------
# CSV contains:
#   - image filename
#   - latitude
#   - longitude
# ============================================================

print('Before reading csv')

df = pd.read_csv(csv_df)

print(len(df.index), 'images available')

df.head()


# ============================================================
# Encoder Architecture
# ------------------------------------------------------------
# This must exactly match the architecture used during
# contrastive pretraining.
#
# The model outputs a 2048-dimensional embedding vector.
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


# ============================================================
# Load Pretrained Encoder
# ============================================================

get_encoder().summary()

projection = get_encoder()

projection.load_weights(pretrained_weights)

print(projection.layers[38])


# ============================================================
# Create Feature Extraction Model
# ------------------------------------------------------------
# We extract features from the penultimate layer rather than
# the final projection output used during SimSiam training.
# ============================================================

rn50 = tf.keras.Model(
    projection.input,
    projection.layers[38].output
)

rn50.summary()

print('Model loaded, weights attached')


# ============================================================
# Extract Features
# ============================================================

features = []

for image_name in df['filenames']:

    image = tf.image.decode_png(
        tf.io.read_file(
            os.path.join(images_path, image_name)
        ),
        channels=3
    )

    embedding = rn50(
        image[tf.newaxis, ...]
    ).numpy().flatten()

    features.append(embedding)


# ============================================================
# Inspect Extracted Features
# ============================================================

print(features[1])

print(df.head())


# ============================================================
# Save Features to CSV
# ------------------------------------------------------------
# Output columns:
#   filename | latitude | longitude | feature_0 ... feature_n
# ============================================================

images_names = df['filenames']

lat = df['lat']

lon = df['lon']


with open(features_csv1, 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(
        ['filenames', 'lat', 'lon']
        +
        [f'feature_{i}' for i in range(len(features[0]))]
    )

    for i, image_name in enumerate(images_names):

        writer.writerow(
            [image_name, lat[i], lon[i]]
            +
            list(features[i])
        )


print('Features Done')
