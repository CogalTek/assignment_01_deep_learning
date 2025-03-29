import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import tensorflow_datasets as tfds
from notify import notify_discord

# ⚙️ Paramètres
epochs = 5
image_size = (180, 180)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

notify_discord("📊 Modèle Stanford Dogs depart !")

# 📆 Chargement du dataset Stanford Dogs
(train_ds_raw, val_ds_raw), ds_info = tfds.load(
    'stanford_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = ds_info.features['label'].num_classes

# 🔧 Prétraitement : resize des images
def format_example(image, label):
    image = tf.image.resize(image, image_size)
    return image, label

train_ds = train_ds_raw.map(format_example, num_parallel_calls=AUTOTUNE)
val_ds = val_ds_raw.map(format_example, num_parallel_calls=AUTOTUNE)

# 🔧 Data augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def augment(image, label):
    for layer in data_augmentation_layers:
        image = layer(image)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

# 🧠 Architecture compatible transfert learning

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
model = make_model(image_size + (3,), num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 🚀 Entraînement avec callback pour sauvegarde
callbacks = [
    keras.callbacks.ModelCheckpoint("stanford_dogs_model.keras"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

# 📁 Sauvegarde finale
model.save("../stanford_dogs/stanford_dogs_model_v3.keras")
pd.DataFrame(history.history).to_csv("../stanford_dogs/stanford_dogs_training_log_v3.csv")

notify_discord("📊 Modèle Stanford Dogs entraîné et sauvegardé !", mention=False)
