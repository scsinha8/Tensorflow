import pathlib
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # ~~~ Practice Basic classification Multi - Start ~~~ #

    # Load training data and get labels from folder names
    data_dir = pathlib.Path("food_data/test/")
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

    train_path = "food_data/train"
    test_path = "food_data/test"

    # Use imagedatagenerator to augment data
    # Variety of data should provide better learning and prevent overfitting
    train_datagen = ImageDataGenerator(rescale=(1./255),
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       height_shift_range=0.2,
                                       width_shift_range=0.2,
                                       shear_range=0.2)

    # Normalise pixel data
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create datasets from directory of images
    # Images are augmented, batched, and resized
    train_data = train_datagen.flow_from_directory(directory=train_path,
                                                   batch_size=32,
                                                   target_size=(224, 224),
                                                   shuffle=True,
                                                   class_mode="categorical")

    test_data = test_datagen.flow_from_directory(directory=test_path,
                                                 batch_size=32,
                                                 target_size=(224, 224),
                                                 class_mode="categorical")

    # Define CNN model. Softmax for multi classification
    category_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64,
                               kernel_size=2,
                               padding="valid",
                               activation="relu"),
                               # input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(64, 2, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(32, 2, activation="relu"),
        tf.keras.layers.Conv2D(32, 2, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # categorical_crossentropy for multi classification
    category_model.compile(loss="categorical_crossentropy",
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=["accuracy"])

    # Only 2 epochs as currently no access to GPU for training
    category_model.fit(train_data, epochs=2, steps_per_epoch=len(train_data), validation_data=test_data,
                       validation_steps=0.2*len(test_data))

    # ~~~ Practice Basic classification Multi - End ~~~ #

    # ~~~ Classification with transfer learning - Start ~~~ #
    # Use same data as above model

    # Define base model, downloading pre-trained model
    # Model is not trainable, feature extractor will be defined on top of the base model
    base_model = tf.keras.applications.EfficientNetB7(include_top=False)
    base_model.trainable = False

    # Use Functional API for model definition for greater flexibility of defining complex models
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="softmax")(x)

    model = tf.keras.Model(inputs, output)

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    model.fit(train_data,
              epochs=2,
              validation_data=test_data)

    # ~~~ Classification with transfer learning - Start ~~~ #
