import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# === Paramètres ===
# image_path = "coton.jpeg"
image_path = "golden.jpg"
model_path = "stanford_dogs_model.keras"
image_size = (180, 180)

# === Charger le modèle entraîné ===
model = keras.models.load_model(model_path)

# === Charger les noms de classes ===
ds_info = tfds.builder("stanford_dogs").info
label_names = ds_info.features["label"].names

# === Charger et afficher l'image ===
img = tf.keras.utils.load_img(image_path, target_size=image_size)
plt.imshow(img)
plt.title("Image testée")
plt.axis("off")
plt.show()

# === Prétraiter l'image ===
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Ajoute la dimension batch
img_array = img_array / 255.0  # Normalisation comme dans le modèle

# === Prédiction ===
predictions = model.predict(img_array)
predicted_index = tf.argmax(predictions[0]).numpy()
predicted_label = label_names[predicted_index]
confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy()

# === Résultat ===
print(f"✅ Le modèle pense que c'est : {predicted_label} ({confidence * 100:.2f}% confiance)")
