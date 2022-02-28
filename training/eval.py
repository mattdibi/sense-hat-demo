#!/usr/bin/env python3

import argparse
import os.path

import pandas as pd

import tensorflow as tf

from train import preprocessing
from train import get_options


def main():
    train_data_path, trained_model_path = get_options()

    # Preprocessing
    train_data = pd.read_csv(train_data_path)

    scaled_train_data = preprocessing(train_data)
    print(pd.DataFrame(scaled_train_data).describe())

    # Grab first sample
    print(scaled_train_data[0])
    print(scaled_train_data[0].shape)

    # Load trained model
    autoencoder_model = tf.keras.models.load_model(trained_model_path)
    autoencoder_model.summary()

    print(autoencoder_model.predict(scaled_train_data[0].reshape((1, 13))))


if __name__ == '__main__':
    main()
