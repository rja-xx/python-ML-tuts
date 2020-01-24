import pickle

# import tensorflow as tf
# from tensorflow import keras
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json

# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

word_index = data.get_word_index()
# print(word_index)

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text])


print(decode_review(test_data[7]))
# different lenghts
print(len(test_data[7]), len(test_data[1]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

print(len(test_data[7]), len(test_data[1]))



# numpy arrays do this
# train_images = train_images/255
# test_images = test_images/255
#
# input_shape = test_images[0].shape
#


def build_model(train_data, train_labels):
    model = keras.Sequential();
    model.add(keras.layers.Embedding(10000, 16))


    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]


    model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)
    return model

def save_model(model):
    #     saving model
    json_model = model.to_json()
    open('resources/textanalysis/model_architecture.json', 'w').write(json_model)
    # saving
    # weights
    model.save_weights('resources/textanalysis/model_weights.h5', overwrite=True)


#
def load_model():
    # loading model
    model = model_from_json(open('resources/textanalysis/model_architecture.json').read())
    model.load_weights('resources/textanalysis/model_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# model = build_model(train_data, train_labels)
# save_model(model)
model = load_model();

i = 232
test_review = test_data[i]
predict = model.predict([test_review])
print(decode_review(test_review))
print(predict[i])
print(test_labels[i])

#
#
# def build_model(input_shape)
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=input_shape),
#         keras.layers.Dense(228, activation="relu"),
#         keras.layers.Dense(128, activation="relu"),
#         keras.layers.Dense(10, activation="softmax")
#     ])
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#     epochs is how many times the model sees the trainingdata
# model.fit(train_images, train_labels, epochs=8)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested accuracy: ", test_acc, " Tested loss: ", test_loss)
# return model
#

# model = build_model(input_shape)
# save_model(model)
# model = load_model();
#
# print(train_images[7])
#
# classes = ['tshirt', 'pants', 'sweater', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ancleboot']
# assert len(classes) == 10
#
# for i in range(10):
#     prediction = model.predict(test_images)
#     print(classes[np.argmax(prediction[13+i])])
#     plt.imshow(test_images[13+i])
#     plt.title(classes[np.argmax(prediction[13+i])])
#     plt.show()
