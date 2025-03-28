import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from load_pet_dataset import load_pet_dataset
from notify import notify_discord

# Paramètres
epochs = 10
batch_size = 32
image_size = (180, 180)

notify_discord("🌐 Démarrage de l'entraînement de l'expérience 3 (remplace les 2 premières convolutions)", mention=False)

# Dataset Cats vs Dogs
train_ds, val_ds = load_pet_dataset(image_size=image_size, batch_size=batch_size)

# Charger le modèle préentrainé sur Stanford Dogs
base_model = keras.models.load_model("../stanford_dogs/stanford_dogs_model_v2.keras")

# Geler toutes les couches sauf les couches conservées
for layer in base_model.layers:
    layer.trainable = False

# Nouvelles entrées + remplacement des 2 premières convolutions
inputs = keras.Input(shape=image_size + (3,))
x = layers.Rescaling(1. / 255)(inputs)
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)

# Reprendre à partir de la couche 5 (index = 5)
for layer in base_model.layers[5:-1]:
    x = layer(x)

# Nouvelle couche de sortie pour classification binaire
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save("../PetImages/Models/cats_vs_dogs_transfer_exp3.keras")
pd.DataFrame(history.history).to_csv("../PetImages/Models/training_log_ex3.csv")

notify_discord("✅ Entraînement expérience 3 terminé et sauvegardé.", mention=True)
