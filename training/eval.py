#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Avoid AVX2 error

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
    # print(pd.DataFrame(scaled_train_data).describe())

    # Grab first sample
    input_sample = scaled_train_data[3:4].copy()
    print("Input data:")
    print(input_sample)
    print()

    # Load trained model
    autoencoder_model = tf.keras.models.load_model(trained_model_path)
    # autoencoder_model.summary()

    # Inference on input sample
    reconstructed_sample = autoencoder_model.predict(input_sample)
    print("Reconstructed data:")
    print(reconstructed_sample)
    print()


if __name__ == '__main__':
    main()
