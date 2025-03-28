import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import pandas as pd
from load_pet_dataset import load_pet_dataset
from notify import notify_discord

# ⚙️ Paramètres
image_size = (180, 180)
batch_size = 32
epochs = 25

# 📦 Dataset : Cats vs Dogs (local)
train_ds, val_ds = load_pet_dataset(image_size=image_size, batch_size=batch_size)

notify_discord("📦 Expérience 3 : remplacement de la sortie + premières convolutions.")

# 🔁 Charger le modèle Stanford Dogs
base_model = keras.models.load_model("../stanford_dogs/stanford_dogs_model_v3.keras")

# Afficher l'architecture du modèle original pour référence
base_model.summary()

# 🧠 Créer un nouveau modèle depuis zéro, mais en conservant l'architecture globale
inputs = keras.Input(shape=image_size + (3,), name="input_layer")

# Remplacer les deux premières couches convolutives
x = layers.Rescaling(1.0 / 255, name="rescaling")(inputs)
x = layers.Conv2D(128, 3, strides=2, padding="same", name="conv2d")(x)
x = layers.BatchNormalization(name="batch_normalization")(x)
x = layers.Activation("relu", name="activation")(x)

# Stocker la première activation pour la connexion résiduelle
previous_block_activation = x

# Premier bloc (corresponds au bloc avec size=256 dans l'architecture d'origine)
x = layers.Activation("relu", name="activation_1")(x)
x = layers.SeparableConv2D(256, 3, padding="same", name="separable_conv2d")(x)
x = layers.BatchNormalization(name="batch_normalization_1")(x)
x = layers.Activation("relu", name="activation_2")(x)
x = layers.SeparableConv2D(256, 3, padding="same", name="separable_conv2d_1")(x)
x = layers.BatchNormalization(name="batch_normalization_2")(x)
x = layers.MaxPooling2D(3, strides=2, padding="same", name="max_pooling2d")(x)

# Connexion résiduelle pour le premier bloc
residual = layers.Conv2D(256, 1, strides=2, padding="same", name="conv2d_1")(previous_block_activation)
x = layers.add([x, residual])
previous_block_activation = x

# À partir d'ici nous réutilisons l'architecture et les poids du modèle Stanford Dogs
# Bloc avec size=512
x = layers.Activation("relu", name="activation_3")(x)
x = layers.SeparableConv2D(512, 3, padding="same", name="separable_conv2d_2")(x)
x = layers.BatchNormalization(name="batch_normalization_3")(x)
x = layers.Activation("relu", name="activation_4")(x)
x = layers.SeparableConv2D(512, 3, padding="same", name="separable_conv2d_3")(x)
x = layers.BatchNormalization(name="batch_normalization_4")(x)
x = layers.MaxPooling2D(3, strides=2, padding="same", name="max_pooling2d_1")(x)

# Connexion résiduelle pour le bloc 512
residual = layers.Conv2D(512, 1, strides=2, padding="same", name="conv2d_2")(previous_block_activation)
x = layers.add([x, residual], name="add_1")
previous_block_activation = x

# Bloc avec size=728
x = layers.Activation("relu", name="activation_5")(x)
x = layers.SeparableConv2D(728, 3, padding="same", name="separable_conv2d_4")(x)
x = layers.BatchNormalization(name="batch_normalization_5")(x)
x = layers.Activation("relu", name="activation_6")(x)
x = layers.SeparableConv2D(728, 3, padding="same", name="separable_conv2d_5")(x)
x = layers.BatchNormalization(name="batch_normalization_6")(x)
x = layers.MaxPooling2D(3, strides=2, padding="same", name="max_pooling2d_2")(x)

# Connexion résiduelle pour le bloc 728
residual = layers.Conv2D(728, 1, strides=2, padding="same", name="conv2d_3")(previous_block_activation)
x = layers.add([x, residual], name="add_2")

# Bloc final
x = layers.SeparableConv2D(1024, 3, padding="same", name="separable_conv2d_6")(x)
x = layers.BatchNormalization(name="batch_normalization_7")(x)
x = layers.Activation("relu", name="activation_7")(x)
x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
x = layers.Dropout(0.25, name="dropout")(x)

# Remplacer la couche de sortie pour la classification binaire (chats vs chiens)
outputs = layers.Dense(1, activation=None, name="output_layer")(x)

# Créer le nouveau modèle
model = Model(inputs=inputs, outputs=outputs)

# Afficher l'architecture du nouveau modèle
model.summary()

# 📦 Copier les poids des couches correspondantes à partir du modèle Stanford Dogs
# Les deux premières convolutions et la couche de sortie sont déjà remplacées
# Nous ne transférons que les poids des couches du milieu
for layer_new in model.layers:
    # Exclure les deux premières couches conv et la couche de sortie
    if layer_new.name.startswith(("rescaling", "conv2d", "batch_normalization", "activation", "separable_conv2d", "batch_normalization_1", "activation_1", "separable_conv2d_1", "batch_normalization_2", "max_pooling2d", "conv2d_1", "output_layer")):
        continue

    # Chercher la couche correspondante dans le modèle de base
    for layer_base in base_model.layers:
        if layer_base.name == layer_new.name:
            try:
                print(f"Transfert des poids: {layer_base.name} -> {layer_new.name}")
                layer_new.set_weights(layer_base.get_weights())
                break
            except Exception as e:
                print(f"Erreur de transfert des poids pour {layer_new.name}: {e}")

# ✅ Compilation
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 🚀 Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 💾 Sauvegarde
model.save("../PetImages/Models/cats_vs_dogs_from_transfer_exp3.keras")
pd.DataFrame(history.history).to_csv("../PetImages/Models/training_log_ex3.csv")

notify_discord("✅ Expérience 3 terminée avec succès !")