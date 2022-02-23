#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    #         MAGNET_X   MAGNET_Z   MAGNET_Y     ACC_Y     ACC_X     ACC_Z    PRESSURE   TEMP_PRESS   TEMP_HUM   HUMIDITY    GYRO_X    GYRO_Y    GYRO_Z
    means = [-6.662364, 15.270208, -27.288259, 0.030234, 0.011407, 0.992252, 998.028151, 35.338573, 36.764190, 21.211922, -0.026414, 0.008353, 0.005063]
    std   = [ 5.250766,  0.853466,   1.319709, 0.028152, 0.089823, 0.024537,   0.035404,  0.376381,  0.336227,  0.613346,  1.096187, 0.498696, 0.146555]

    train_data = (train_data - means)/std
    print(train_data.describe())

    # scaler = StandardScaler().fit(train_data)

    # print(scaler.mean_)
    # print(scaler.scale_)
    # scaled_data = scaler.transform(train_data)

    # print(pd.DataFrame(scaled_data).describe())



if __name__ == '__main__':
    main()
