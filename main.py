# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ~~~ Practice Basic Linear Regression - Start ~~~ #

    # Create input and output tensors for linear regression
    array1 = tf.range(0, 25, 1)
    array2 = tf.range(0, 50, 2)

    # Define sequential model (no activation as relationship is linear so non-linearity is not needed)
    regression_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile model with standard loss, optimizer
    regression_model.compile(loss=tf.keras.losses.mae,
                             optimizer=tf.keras.optimizers.SGD(),
                             metrics=["mae"])

    # Fit the model to learn from the dummy data created
    regression_history = regression_model.fit(array1, array2, epochs=500)

    # Create test data to test how the model has learned
    test_array = np.random.randint(0, 200, 10)

    predictions = regression_model(test_array)

    # Visualise the data
    plt.scatter(array1, array2, c="b")
    plt.scatter(test_array, (2 * test_array), c="g")
    plt.scatter(test_array, predictions, c="r")
    plt.show()

    # ~~~ Practice Basic Regression - End ~~~ #

    # ~~~ Practice regression - Insurance START ~~~ #

    # Fetch dataset
    insurance = pd.read_csv(
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

    # Define transformer that will be used to normalise data and make it easier
    # for the model to learn
    column_transformer = make_column_transformer(
        (MinMaxScaler(), ["age", "bmi", "children"]),
        (OneHotEncoder(), ["sex", "smoker", "region"])
    )

    # Define input and output data
    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]

    # Split the dataset into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Apply transformer to data
    column_transformer.fit(X_train)

    X_train_normal = column_transformer.transform(X_train)
    X_test_normal = column_transformer.transform(X_test)

    # Create dataset by slicing input data and associating it with corresponding output data
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_normal, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_normal, y_test))

    # Shuffle and batch data for optimised learning
    train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # Define model
    regression_model = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(96, activation="sigmoid"),
        layers.Dense(72, activation="tanh"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    # Define loss, optimizer, and metrics to measure how well it learns
    regression_model.compile(loss="mae",
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=["mae"])

    # Fit the model to the data
    regression_model.fit(train_dataset,
                         epochs=500,
                         validation_data=test_dataset)

    # ~~~ Practice regression - Insurance END ~~~ #

    # ~~~ Practice regression - Boston Housing START ~~~ #

    #Load dataset and prepare the data (similar to insurance data above)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2)

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train_normal = scaler.transform(X_train)
    X_test_normal = scaler.transform(X_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_normal, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_normal, y_test))

    train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    regression_model = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(96, activation="sigmoid"),
        layers.Dense(72, activation="tanh"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    regression_model.compile(loss="mae",
                             optimizer=tf.keras.optimizers.SGD(),
                             metrics=["mae"])

    # Attempt to improve learning and minimize loss by adjusting LR to
    # prevent plateauing
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    patience=25,
                                                    factor=0.5,
                                                    min_lr=0.00001)

    save_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath= os.environ['USERPROFILE'] + "\\Model_checkpoints\\model_1",
        save_best_only=True)

    regression_model.fit(train_dataset,
                         epochs=500,
                         validation_data=test_dataset)

    # ~~~ Practice regression - Boston Housing END ~~~ #

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
