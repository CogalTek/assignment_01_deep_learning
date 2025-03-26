import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force le CPU uniquement
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd


# print("🖥️  Devices disponibles :")
# for device in tf.config.list_physical_devices():
#     print(device)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("✅ GPU détecté : TensorFlow peut l'utiliser !")
# else:
#     print("❌ Aucun GPU détecté, TensorFlow utilise le CPU.")


# Chargement du dataset Stanford Dogs
(train_ds_raw, val_ds_raw), ds_info = tfds.load(
    'stanford_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

# Paramètres globaux
image_size = (180, 180)
batch_size = 128
AUTOTUNE = tf_data.AUTOTUNE
num_classes = ds_info.features['label'].num_classes

# Prétraitement de base : resize des images
def format_example(image, label):
    image = tf.image.resize(image, image_size)
    return image, label

train_ds = train_ds_raw.map(format_example, num_parallel_calls=AUTOTUNE)
val_ds = val_ds_raw.map(format_example, num_parallel_calls=AUTOTUNE)

# Augmentation des données (horizontal flip + légère rotation)
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def augment(image, label):
    for layer in data_augmentation_layers:
        image = layer(image)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

# Préparation des datasets (batching + prefetch)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

# Modèle CNN inspiré du tutoriel Keras
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Pour les résidus

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Connexion résiduelle
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    # Sortie
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)  # logits

    return keras.Model(inputs, outputs)

# Création du modèle
model = make_model(input_shape=image_size + (3,), num_classes=num_classes)

# Compilation
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Entraînement
epochs = 25
callbacks = [
    keras.callbacks.ModelCheckpoint("save_stanford_at_{epoch}.keras"),
]

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks,
)

# Sauvegarde finale du modèle entraîné
model.save("stanford_dogs_model.keras")

# Export de l'historique d'entraînement
pd.DataFrame(history.history).to_csv("stanford_dogs_training_log.csv")

print("✅ Entraînement terminé, modèle et métriques sauvegardés.")
