# experience_02.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
from load_pet_dataset import load_pet_dataset

# ⚙️ Paramètres
image_size = (180, 180)
batch_size = 32
epochs = 50

# 📦 Dataset : cats vs dogs (local)
train_ds, val_ds = load_pet_dataset(image_size=image_size, batch_size=batch_size)

# 🔁 Charger le modèle préentraîné
base_model = keras.models.load_model("../stanford_dogs/stanford_dogs_model_v3.keras")

base_model.summary()

# ✂️ Supprimer la dernière couche et en ajouter une nouvelle
x = base_model.layers[-2].output  # la couche juste avant la Dense(120)
new_output = layers.Dense(1)(x)   # binaire : 1 sortie, logits
model = keras.Model(inputs=base_model.input, outputs=new_output)

# ✅ Recompiler
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 🚀 Réentraîner
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
model.save("../PetImages/Models/cats_vs_dogs_from_transfer_exp2_v2.keras")
pd.DataFrame(history.history).to_csv("../PetImages/Models/training_log_ex2_v2.csv")

print("✅ Expérience 2 terminée.")
