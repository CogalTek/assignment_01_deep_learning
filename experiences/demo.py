import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import keras
import matplotlib.pyplot as plt

# Charger le modèle Keras
model = None

def predict_image(image_path, value):
    img = keras.utils.load_img(image_path, target_size=(180, 180))
    plt.imshow(img)

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    scoreCat = 100 * (1 - score)
    scoreDog = 100 * score
    isCorrect = False

    if (scoreCat > scoreDog and value == "Cat") or (scoreCat < scoreDog and value == "Dog"):
        isCorrect = True

    print(f"{image_path} This image is {scoreCat:.2f}% cat and {scoreDog:.2f}% dog. IsCorrect={isCorrect}")

# Model exp 01
print("Model exp 01")
model = tf.keras.models.load_model('../PetImages/Models/cats_vs_dogs_from_scratch_V2.keras')
predict_image('../PetImages/Dog/9.jpg', "Dog")
predict_image('../PetImages/Dog/4.jpg', "Dog")
predict_image('../PetImages/Cat/0.jpg', "Cat")
predict_image('../PetImages/Cat/9.jpg', "Cat")
# Model exp 02
print("Model exp 02")
model = tf.keras.models.load_model('../PetImages/Models/cats_vs_dogs_from_transfer_exp2_v2.keras')
predict_image('../PetImages/Dog/9.jpg', "Dog")
predict_image('../PetImages/Dog/4.jpg', "Dog")
predict_image('../PetImages/Cat/0.jpg', "Cat")
predict_image('../PetImages/Cat/9.jpg', "Cat")
# Model exp 03
print("Model exp 03")
model = tf.keras.models.load_model('../PetImages/Models/cats_vs_dogs_from_transfer_exp3.keras')
predict_image('../PetImages/Dog/9.jpg', "Dog")
predict_image('../PetImages/Dog/4.jpg', "Dog")
predict_image('../PetImages/Cat/0.jpg', "Cat")
predict_image('../PetImages/Cat/9.jpg', "Cat")
# Model exp 04
print("Model exp 04")
model = tf.keras.models.load_model('../PetImages/Models/cats_vs_dogs_from_transfer_exp4.keras')
predict_image('../PetImages/Dog/9.jpg', "Dog")
predict_image('../PetImages/Dog/4.jpg', "Dog")
predict_image('../PetImages/Cat/0.jpg', "Cat")
predict_image('../PetImages/Cat/9.jpg', "Cat")

# image_path = '../PetImages/Dog/14.jpg'
# result = predict_image(image_path)
# print(f"dog L'image est un {result}.")