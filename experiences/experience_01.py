# experience_01.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from load_pet_dataset import load_pet_dataset
from notify import notify_discord

# ⚙️ Paramètres
image_size = (180, 180)
batch_size = 32
epochs = 5

# 📦 Dataset : cats vs dogs (local)
train_ds, val_ds = load_pet_dataset(image_size=image_size, batch_size=batch_size)



# 🧠 Architecture CNN
def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1. / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for size in [256, 512, 728]:
        residual = x
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(residual)
        x = layers.add([x, residual])

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1)(x)  # logits pour binaire

    return keras.Model(inputs, outputs)

model = make_model(image_size + (3,))
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 🚀 Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save("cats_vs_dogs_from_scratch.keras")
pd.DataFrame(history.history).to_csv("training_log_ex1.csv")
