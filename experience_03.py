from load_pet_dataset import load_pet_dataset
import keras

# Charger le modèle pré-entraîné
model = keras.models.load_model("stanford_dogs_model.keras")

# Désactiver les 2 premières couches
for layer in model.layers[:3]:  # Rescaling + Conv2D + BN
    layer.trainable = False

# Remplacer la sortie
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
model.save("exp3_transfer_first2_finetune.keras")
