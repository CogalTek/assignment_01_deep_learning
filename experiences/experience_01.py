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

notify_discord("🚀 Début de l'entraînement sur le dataset local Cats vs Dogs !")

# 🧠 Architecture CNN
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

model = make_model(image_size + (3,), num_classes=2)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 🚀 Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save("cats_vs_dogs_from_scratch_V2.keras")
pd.DataFrame(history.history).to_csv("training_log_ex1_V2.csv")
notify_discord("🚀 Fin de l'entraînement sur le dataset local Cats vs Dogs !")
