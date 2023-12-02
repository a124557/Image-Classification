# A simple image classification machine learning model built for learning purposes
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Loading the fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Revewing the data

# train_images.shape returns the shape of the array. index[0] is the numebr of sample iamges, index[1] rows in pixels
# index[2] represents columns in pixels
print("Training data shape:", train_images.shape)
print("Number of training labels:", len(train_labels))

# Preprocess data by scaling pixel values to range of 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
"""10 neurons that are activated based on the 'softmax' function. 'softmax' allows us to provide a
probability to each image, making it easier to interpret the model's predictions. In simpler terms, 
the 'softmax' activation helps the model say, I'm this certain amount sure it's a T-shirt, this certain 
amount sure it's a Trouser, and so on. It turns the model's raw outputs into something more understandable 
and useful for making predictions on a multi-class classification task."""
model = keras.Sequential([
    # Flatten layer convers the 2D images to a single line of 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    # 128 neurons that are activated based on the 'relu' function
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=10)

