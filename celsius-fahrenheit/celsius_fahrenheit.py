# -*- coding: utf8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from numpy import array, round
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.logging import set_verbosity, ERROR
from argparse import ArgumentParser

set_verbosity(ERROR)

celsius_examples = [-40, -10,  0,  8, 15, 22,  38]
celsius_values = array(celsius_examples, dtype=float)

fahrenheit_examples = [-40,  14, 32, 46, 59, 72, 100]
fahrenheit_values = array(fahrenheit_examples, dtype=float)


def create_model(layer):
    model = Sequential([layer])
    model.compile(loss='mean_squared_error', optimizer=Adam(0.1))
    return model


def get_args():
    args = ArgumentParser()

    args.add_argument('-c', '--celsius',
                      help='converts the value into degrees Celsius.',
                      type=float)

    args.add_argument('-f', '--fahrenheit',
                      help='converts the value into degrees Fahrenheit.',
                      type=float)

    return vars(args.parse_args())


def main():
    layer = Dense(units=1, input_shape=[1])
    model = create_model(layer)
    history = []

    args = get_args()
    celsius = args['celsius']
    fahrenheit = args['fahrenheit']

    labels = ['{0}° Celsius is {1}° Fahrenheit.',
              '{0}° Fahrenheit is {1}° Celsius.']

    if celsius:
        history = model.fit(celsius_values, fahrenheit_values,
                            epochs=500, verbose=False)

        labels = labels[0].format(celsius,
                           round(model.predict([celsius]), 2))

        labels = labels.replace('[', '').replace(']', '')
        print(labels)


    elif fahrenheit:
        history = model.fit(fahrenheit_values, celsius_values,
                            epochs=500, verbose=False)

        labels = labels[1].format(fahrenheit,
                           round(model.predict([fahrenheit]), 2))

        labels = labels.replace('[', '').replace(']', '')
        print(labels)


if __name__ == '__main__':
    main()
