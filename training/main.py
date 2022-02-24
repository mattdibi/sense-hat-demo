#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
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
    # means = np.array([-6.662364, 15.270208, -27.288259, 0.030234, 0.011407, 0.992252, 998.028151, 35.338573, 36.764190, 21.211922, -0.026414, 0.008353, 0.005063])
    # std   = np.array([ 5.250766,  0.853466,   1.319709, 0.028152, 0.089823, 0.024537,   0.035404,  0.376381,  0.336227,  0.613346,  1.096187, 0.498696, 0.146555])

    min = np.array([-1.8199598e+01, 1.3105305e+01, -3.1945429e+01, -1.0663300e-01, -3.1721205e-01, 8.5927510e-01, 9.9792896e+02, 3.4666668e+01, 3.6153427e+01, 2.0175404e+01, -2.1469840e+00, -1.3511374e+00, -5.0122184e-01])
    max = np.array([ 3.7974422e+00, 1.7207603e+01, -2.4966690e+01, 1.1758627e-01, 2.4764843e-01, 1.1285601e+00, 9.9812500e+02, 3.6097916e+01, 3.7271065e+01, 2.3081772e+01, 1.8424674e+00, 7.2500044e-01, 4.4410592e-01])

    scaled_train_data = (train_data - min) / (max - min)

    print(scaled_train_data.shape)
    print(pd.DataFrame(scaled_train_data).describe())

    # scaler = MinMaxScaler().fit(train_data)
    # scaled_data = scaler.transform(train_data)
    # print(pd.DataFrame(scaled_data).describe())


if __name__ == '__main__':
    main()
