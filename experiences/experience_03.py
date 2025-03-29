import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import pandas as pd
from load_pet_dataset import load_pet_dataset
from notify import notify_discord

# ⚙️ Paramètres
image_size = (180, 180)
batch_size = 32
epochs = 5

# 📦 Dataset : Cats vs Dogs (local)
train_ds, val_ds = load_pet_dataset(image_size=image_size, batch_size=batch_size)

notify_discord("📦 Expérience 3 : remplacement de la sortie + premières convolutions.")

# 🔁 Charger le modèle Stanford Dogs
base_model = keras.models.load_model("../stanford_dogs/stanford_dogs_training_log_v3.keras")

base_model.summary()

# 🔧 Recréer modèle avec les deux premières convolutions remplacées
inputs = keras.Input(shape=image_size + (3,), name="input")

# Nouvelle entrée + nouvelle première conv
x = layers.Rescaling(1. / 255, name="custom_rescale")(inputs)
x = layers.Conv2D(128, 3, strides=2, padding="same", name="custom_conv1")(x)
x = layers.BatchNormalization(name="custom_bn1")(x)
x = layers.Activation("relu", name="custom_relu1")(x)

# 🔁 Récupérer la sortie après le premier bloc résiduel du modèle original
# Dans ton architecture, c’est après le premier `add` → index = 6
block_start_index = 6
for i, layer in enumerate(base_model.layers[block_start_index:]):
    x = layer(x)

# 🔄 Changer la dernière couche
x = layers.Dense(1, activation=None, name="new_output")(x)  # binaire

# 🔧 Nouveau modèle
model = Model(inputs=inputs, outputs=x, name="exp3_model")

# 📦 Copier les poids des couches conservées
for layer in model.layers:
    if layer.name in [l.name for l in base_model.layers]:
        try:
            layer.set_weights(base_model.get_layer(layer.name).get_weights())
        except:
            pass  # On skip les couches modifiées

# ✅ Compilation
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 🚀 Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 💾 Sauvegarde
model.save("cats_vs_dogs_from_transfer_exp3.keras")
pd.DataFrame(history.history).to_csv("training_log_ex3.csv")

notify_discord("✅ Expérience 3 terminée avec succès !")
