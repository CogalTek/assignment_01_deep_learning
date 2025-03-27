# experience_04.py

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

# 🧩 On récupère toutes les couches sauf les deux dernières convolutives + dense
# Ici, on garde jusqu'à l’avant-avant dernière grosse conv (728), on coupe avant les couches finales

# Trouve l'index de la couche à laquelle couper si besoin :
cut_index = -6  # empirique si tu veux personnaliser

inputs = keras.Input(shape=image_size + (3,))
x = layers.Rescaling(1. / 255)(inputs)

# On rejoue les premières couches de base_model sauf les deux dernières convolutives
for layer in base_model.layers[2:cut_index]:
    x = layer(x)

# On remplace les deux dernières grosses convolutions
x = layers.SeparableConv2D(1024, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(1)(x)  # logits

model = keras.Model(inputs, x)

# ✅ Compilation
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 🚀 Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save("../PetImages/Models/cats_vs_dogs_transfer_exp4.keras")
pd.DataFrame(history.history).to_csv("../PetImages/Models/training_log_ex4.csv")

print("✅ Expérience 4 terminée.")
