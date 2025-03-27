import tensorflow as tf
import keras
from keras import layers
import os
import numpy as np

# Préparation du dataset Cats vs Dogs
def load_pet_dataset(image_size=(180, 180), batch_size=128):
    # Nettoyage des images corrompues
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("../PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
    print(f"🧹 Images corrompues supprimées : {num_skipped}")

    # Chargement et prétraitement
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "../PetImages",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    def augment(img, label):
        return data_augmentation(img), label

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds
