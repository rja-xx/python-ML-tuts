import pickle

import tensorflow as tf
# from tensorflow import keras
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# numpy arrays do this
train_images = train_images/255
test_images = test_images/255

input_shape = test_images[0].shape

def save_model(model):
    # saving model
    json_model = model.to_json()
    open('resources/fashionmodel/model_architecture.json', 'w').write(json_model)
    # saving weights
    model.save_weights('resources/fashionmodel/model_weights.h5', overwrite=True)

def load_model():
    # loading model
    model = model_from_json(open('resources/fashionmodel/model_architecture.json').read())
    model.load_weights('resources/fashionmodel/model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(228, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # epochs is how many times the model sees the trainingdata
    model.fit(train_images, train_labels, epochs=8)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Tested accuracy: ", test_acc, " Tested loss: ", test_loss)
    return model


# model = build_model(input_shape)
# save_model(model)
model = load_model();
#
# print(train_images[7])
#
classes = ['tshirt', 'pants', 'sweater', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ancleboot']
assert len(classes) == 10

for i in range(10):
    prediction = model.predict(test_images)
    print(classes[np.argmax(prediction[13+i])])
    plt.imshow(test_images[13+i])
    plt.title(classes[np.argmax(prediction[13+i])])
    plt.show()