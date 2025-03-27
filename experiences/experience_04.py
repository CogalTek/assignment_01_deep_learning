from load_pet_dataset import load_pet_dataset
import keras

# Charger le modèle
model = keras.models.load_model("stanford_dogs_model.keras")

# Geler toutes les couches sauf les 2 dernières conv + sortie
for layer in model.layers:
    layer.trainable = False

# Déverrouiller les 2 dernières SeparableConv2D (à la fin de la boucle)
unfrozen = 0
for layer in reversed(model.layers):
    if isinstance(layer, keras.layers.SeparableConv2D):
        layer.trainable = True
        unfrozen += 1
        if unfrozen >= 2:
            break

# Remplacer la couche de sortie
x = model.layers[-2].output
new_output = keras.layers.Dense(1, activation=None)(x)
model = keras.Model(inputs=model.input, outputs=new_output)

# Compilation
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Chargement dataset
train_ds, val_ds = load_pet_dataset()

# Entraînement
history = model.fit(train_ds, validation_data=val_ds, epochs=50)
model.save("exp4_transfer_last2_finetune.keras")
