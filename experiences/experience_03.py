# experience_03.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from load_pet_dataset import load_pet_dataset

# ⚙️ Paramètres
image_size = (180, 180)
batch_size = 32
epochs = 5

# 📦 Dataset Cats vs Dogs
train_ds, val_ds = load_pet_dataset(image_size=image_size, batch_size=batch_size)

# 🔁 Charger le modèle préentraîné sur Stanford Dogs
base_model = keras.models.load_model("../stanford_dogs/stanford_dogs_model.keras")

# ❄️ On gèle tous les poids
for layer in base_model.layers:
    layer.trainable = False

# 🧩 On remplace les deux premières couches convolutives
inputs = keras.Input(shape=image_size + (3,))
x = layers.Rescaling(1. / 255)(inputs)
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(256, 3, strides=1, padding="same", activation="relu")(x)

# 🔁 On reprend à partir de la 3e couche de base_model (après les deux premières conv)
for layer in base_model.layers[3:-1]:  # skip input, rescale, conv1, conv2
    x = layer(x)

# 🔄 Remplacement de la dernière couche (classification binaire)
x = layers.Dense(1)(x)
model = keras.Model(inputs, x)

# ✅ Compilation
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 🚀 Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save("../PetImages/Models/cats_vs_dogs_transfer_exp3.keras")
pd.DataFrame(history.history).to_csv("../PetImages/Models/training_log_ex3.csv")

print("✅ Expérience 3 terminée.")
