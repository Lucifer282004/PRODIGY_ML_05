!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition

import zipfile
zfile = zipfile.ZipFile('/content/fruit-and-vegetable-image-recognition.zip','r')
zfile.extractall('/content')
zfile.close()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_set = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/train',
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/validation',
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Define the CNN model
cnn = tf.keras.models.Sequential()

# Add layers to the CNN model
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))

# Compile the model
cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=36)

cnn.save('trained_model.h5')

import json
with open('history.json', 'w') as f:
    json.dump(training_history.history, f)

print("Validation Set Accuracy {}" .format(training_history.history['val_accuracy'][-1]*100))

cnn.save('trained_model.h5')

plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

cnn2=tf.keras.models.load_model("/content/trained_model.h5")

import cv2
image_path="/content/test/apple/Image_1.jpg"
img=cv2.imread(image_path)
plt.imshow(img)

plt.title("Test Image")

image=tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr=tf.keras.preprocessing.image.img_to_array(image)
input_arr=np.array([input_arr])
prediction=cnn2.predict(input_arr)
print(prediction)

result_index = np.argmax(prediction[0])
print("Predicted class index:", result_index)

class_names = training_set.class_names
predicted_class = class_names[result_index]
print("Predicted Item name is :- ", predicted_class)
