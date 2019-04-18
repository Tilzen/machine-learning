# -*- coding:utf8 -*-

# os library and limits the log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import matplotlib to visualization
import matplotlib.pyplot as plt

# helper librarys
from math import ceil
from numpy import array, argmax

# import tensorflow tools and datasets
from tensorflow import enable_eager_execution, cast, float32
from tensorflow_datasets import load
from tensorflow.nn import relu, softmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.logging import set_verbosity, ERROR

set_verbosity(ERROR)
enable_eager_execution()


dataset, metadata = load('fashion_mnist', as_supervised=True, with_info=True)

clothes_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

BATCH_SIZE = 32

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

steps = num_train_examples / BATCH_SIZE


def normalize(images, labels):
    images = cast(images, float32)
    images /= 255
    return images, labels


def show_image(image):
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def init_datasets():
    return dataset['train'], dataset['test']


def create_model():
    input_layer = Flatten(input_shape=(28, 28, 1))
    hidden_layer = Dense(units=128, activation=relu)
    output_layer = Dense(units=10, activation=softmax)

    model = Sequential([input_layer, hidden_layer, output_layer])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def configure_datasets(train_dataset, test_dataset):
    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.shuffle(num_train_examples).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset


def get_predictions(model, test_dataset):
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict([test_images])
    return predictions


def show_predictions(predictions):
    for prediction in predictions:
        i = argmax(prediction)
        print(f'The clothe is {clothes_classes[i]}')


def main():
    model = create_model()
    train_dataset, test_dataset = init_datasets()
    train_dataset, test_dataset = configure_datasets(train_dataset, test_dataset)
    model.fit(train_dataset, epochs=5, steps_per_epoch=ceil(steps))

    predictions = get_predictions(model, test_dataset)
    show_predictions(predictions)


if __name__ == '__main__':
    main()
