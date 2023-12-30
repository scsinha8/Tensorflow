import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons

if __name__ == '__main__':
    # ~~~ Practice Basic Binary classification - Start ~~~ #

    # Create dataset
    X, y = make_circles(1000,
                        noise=0.05)

    # Visualise data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

    # Split into training and validation sets
    X_train = X[:800]
    X_test = X[800:]
    y_train = y[:800]
    y_test = y[800:]

    # Define model (sigmoid activation in final layer for binary classification)
    classification_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # BinaryCrossentropy loss for Binary classification data
    classification_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                 optimizer=tf.keras.optimizers.Adam(),
                                 metrics=["accuracy"])

    classification_history = classification_model.fit(X_train, y_train, epochs=100,
                                                      validation_data=(X_test, y_test))

    # Evaluate model on new data
    X_test, y_true = make_circles(150, noise=0.05)

    y_preds = classification_model.evaluate(X_test, y_true)

    y_preds = tf.squeeze(tf.round(y_preds))

    # ~~~ Practice Basic Binary classification - End ~~~ #

    # ~~~ Practice Basic classification (Make Moons) - Start ~~~ #

    # Create dataset
    X, y = make_moons(1000, noise=0.2)

    # Visualise data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

    X_test, y_test = make_moons(200, noise=0.2)

    moons_model = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    moons_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])

    moons_model.fit(X, y, epochs=200, validation_data=(X_test, y_test))
