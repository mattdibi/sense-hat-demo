#!/usr/bin/env python3

import argparse
import os.path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout


def get_options():
    DEFAULT_TRAIN_DATA_PATH = "train-raw.csv"
    DEFAULT_SAVED_MODEL_NAME = "autoencoder"

    # Get options
    parser = argparse.ArgumentParser(
            description="Training script for Kura AI Wire Component anomaly detection",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
            "-t", "--train_data_path",
            help="Path to .csv training set",
            type=str, required=False, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument(
            "-s", "--saved_model_name",
            help="Folder where the trained model will be saved to",
            type=str, required=False, default=DEFAULT_SAVED_MODEL_NAME)

    args = parser.parse_args()

    train_data_path = args.train_data_path
    trained_model_path = os.path.join("saved_model", args.saved_model_name)

    return train_data_path, trained_model_path


def preprocessing(data):
    # Select features
    features = ['MAGNET_X', ' MAGNET_Z', ' MAGNET_Y', ' ACC_Y',
                ' ACC_X', ' ACC_Z', ' PRESSURE', ' TEMP_PRESS', ' TEMP_HUM',
                ' HUMIDITY', ' GYRO_X', ' GYRO_Y', ' GYRO_Z']

    data = data[features]
    data = data.to_numpy()

    # MinMax scaling
    #         MAGNET_X   MAGNET_Z   MAGNET_Y     ACC_Y     ACC_X     ACC_Z    PRESSURE   TEMP_PRESS   TEMP_HUM   HUMIDITY    GYRO_X    GYRO_Y    GYRO_Z
    min = np.array([-1.8199598e+01, 1.3105305e+01, -3.1945429e+01, -1.0663300e-01, -3.1721205e-01, 8.5927510e-01, 9.9792896e+02, 3.4666668e+01, 3.6153427e+01, 2.0175404e+01, -2.1469840e+00, -1.3511374e+00, -5.0122184e-01])
    max = np.array([3.7974422e+00, 1.7207603e+01, -2.4966690e+01, 1.1758627e-01, 2.4764843e-01, 1.1285601e+00, 9.9812500e+02, 3.6097916e+01, 3.7271065e+01, 2.3081772e+01, 1.8424674e+00, 7.2500044e-01, 4.4410592e-01])

    scaled_train_data = (data - min) / (max - min)

    return scaled_train_data


def create_model(input_dim):
    # Latent space dimension
    latent_dim = 4

    # The encoder will consist of a number of dense layers that decrease in size
    # as we taper down towards the bottleneck of the network, the latent space
    input_data = Input(shape=(input_dim,), name='INPUT0')

    # hidden layers
    encoder = Dense(48, activation='tanh', name='encoder_1')(input_data)
    encoder = Dropout(.1)(encoder)
    encoder = Dense(16, activation='tanh', name='encoder_2')(encoder)
    encoder = Dropout(.1)(encoder)

    # bottleneck layer
    latent_encoding = Dense(latent_dim, activation='linear', name='latent_encoding')(encoder)

    # The decoder network is a mirror image of the encoder network
    decoder = Dense(16, activation='tanh', name='decoder_1')(latent_encoding)
    decoder = Dropout(.1)(decoder)
    decoder = Dense(48, activation='tanh', name='decoder_2')(decoder)
    decoder = Dropout(.1)(decoder)

    # The output is the same dimension as the input data we are reconstructing
    reconstructed_data = Dense(input_dim, activation='linear', name='OUTPUT0')(decoder)

    autoencoder_model = Model(input_data, reconstructed_data)

    return autoencoder_model


def main():
    train_data_path, trained_model_path = get_options()

    # ########
    # Preprocessing
    # ########
    train_data = pd.read_csv(train_data_path)

    scaled_train_data = preprocessing(train_data)

    x_train, x_test = train_test_split(scaled_train_data, test_size=0.15, random_state=42)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # ########
    # Model
    # ########
    autoencoder_model = create_model(x_train.shape[1])
    autoencoder_model.summary()

    # ########
    # Training
    # ########
    batch_size = 64
    max_epochs = 15
    learning_rate = .0001

    opt = optimizers.Adam(learning_rate=learning_rate)
    autoencoder_model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    autoencoder_model.fit(x_train, x_train,
                          shuffle=True,
                          epochs=max_epochs,
                          batch_size=batch_size,
                          validation_data=(x_test, x_test))

    autoencoder_model.save(trained_model_path)

    # ########
    # Postprocessing
    # ########
    x_test_recon = autoencoder_model.predict(x_test)
    reconstruction_scores = np.mean((x_test - x_test_recon)**2, axis=1)  # MSE

    anomaly_data = pd.DataFrame({'recon_score': reconstruction_scores})
    print(anomaly_data.describe())

    # Compute threshold from test set
    alpha = 1.25
    threshold = np.max(reconstruction_scores) * alpha
    print("Anomaly score threshold: %f" % threshold)


if __name__ == '__main__':
    main()
