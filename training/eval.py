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

    # Grab first sample
    print(scaled_train_data[0])
    print(scaled_train_data[0].shape)

    # Load trained model
    autoencoder_model = tf.keras.models.load_model("saved_model/autoencoder")

    # Check its architecture
    autoencoder_model.summary()

    print(autoencoder_model.predict(scaled_train_data[0].reshape((1, 13))))

if __name__ == '__main__':
    main()
