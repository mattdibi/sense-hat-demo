#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

def main():
    # ########
    # Preprocessing
    # ########
    train_data_path = "train-raw.csv"
    train_data = pd.read_csv(train_data_path)

    features = ['MAGNET_X', ' MAGNET_Z', ' MAGNET_Y', ' ACC_Y',
       ' ACC_X', ' ACC_Z', ' PRESSURE', ' TEMP_PRESS', ' TEMP_HUM', ' HUMIDITY',
       ' GYRO_X', ' GYRO_Y', ' GYRO_Z']
    train_data = train_data[features]

    print(train_data.columns)
    print(train_data.head())
    print(train_data.describe())

    train_data = train_data.to_numpy()

    #         MAGNET_X   MAGNET_Z   MAGNET_Y     ACC_Y     ACC_X     ACC_Z    PRESSURE   TEMP_PRESS   TEMP_HUM   HUMIDITY    GYRO_X    GYRO_Y    GYRO_Z
    min = np.array([-1.8199598e+01, 1.3105305e+01, -3.1945429e+01, -1.0663300e-01, -3.1721205e-01, 8.5927510e-01, 9.9792896e+02, 3.4666668e+01, 3.6153427e+01, 2.0175404e+01, -2.1469840e+00, -1.3511374e+00, -5.0122184e-01])
    max = np.array([ 3.7974422e+00, 1.7207603e+01, -2.4966690e+01, 1.1758627e-01, 2.4764843e-01, 1.1285601e+00, 9.9812500e+02, 3.6097916e+01, 3.7271065e+01, 2.3081772e+01, 1.8424674e+00, 7.2500044e-01, 4.4410592e-01])

    scaled_train_data = (train_data - min) / (max - min)

    print(scaled_train_data.shape)
    print(pd.DataFrame(scaled_train_data).describe())

    x_train, x_test = train_test_split(scaled_train_data, test_size=0.15, random_state=42)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # ########
    # Model
    # ########
    input_dim = x_train.shape[1]

    batch_size = 64
    latent_dim = 4
    max_epochs = 15

    # The encoder will consist of a number of dense layers that decrease in size
    # as we taper down towards the bottleneck of the network, the latent space
    input_data = Input(shape=(input_dim,), name='encoder_input')

    # hidden layers
    encoder = Dense(48,activation='tanh', name='encoder_1')(input_data)
    encoder = Dropout(.1)(encoder)
    encoder = Dense(16,activation='tanh', name='encoder_2')(encoder)
    encoder = Dropout(.1)(encoder)

    # bottleneck layer
    latent_encoding = Dense(latent_dim, activation='linear', name='latent_encoding')(encoder)

    # The decoder network is a mirror image of the encoder network
    decoder = Dense(16, activation='tanh', name='decoder_1')(latent_encoding)
    decoder = Dropout(.1)(decoder)
    decoder = Dense(48, activation='tanh', name='decoder_2')(decoder)
    decoder = Dropout(.1)(decoder)

    # The output is the same dimension as the input data we are reconstructing
    reconstructed_data = Dense(input_dim, activation='linear', name='reconstructed_data')(decoder)

    autoencoder_model = Model(input_data, reconstructed_data)
    autoencoder_model.summary()

    opt = optimizers.Adam(lr=.0001)
    autoencoder_model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    train_history = autoencoder_model.fit(x_train, x_train,
        shuffle=True,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

    autoencoder_model.save("saved_model/autoencoder")

    # ########
    # Postprocessing
    # ########
    x_test_recon  = autoencoder_model.predict(x_test)
    reconstruction_scores = np.mean((x_test - x_test_recon)**2, axis=1)

    anomaly_data = pd.DataFrame({'recon_score':reconstruction_scores})
    print(anomaly_data.describe())

    # plt.xlabel('Reconstruction Score')
    # anomaly_data['recon_score'].plot.hist(bins=200, range=[.04, 1])
    # plt.show()

if __name__ == '__main__':
    main()
