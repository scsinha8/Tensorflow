import zipfile
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()




if __name__ == "__main__":

    # ~~~ NLP Binary classification - Start ~~~ #

    # Objective is to perform binary classification on tweet data
    # Tweet data is labeled as 'real disaster(1)' or 'not real disaster(0)'
    # This should serve to perform sentiment analysis on the tweets using binary classification
    unzip_data("nlp_getting_started.zip")

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_df_shuffled = train_df.sample(frac=1, random_state=8)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_df_shuffled["text"].to_numpy(),
        train_df_shuffled["target"].to_numpy(),
        test_size=0.1,
        random_state=8
    )

    # Create a text vectorizer to represent the text data as a vector that can be learned
    # This can then be represented in an embedding layer, where words of similar sentiment are
    # close to each other in the vector space
    text_vectorizer = TextVectorization(max_tokens=10000,
                                        output_mode="int",
                                        output_sequence_length=15)

    text_vectorizer.adapt(train_sentences)

    embedding = tf.keras.layers.Embedding(input_dim=10000,
                                          output_dim=128,
                                          embeddings_initializer="uniform",
                                          input_length=15)

    # A number of models will be tested out to see which performs best
    # LSTM model

    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model_1 = tf.keras.Model(inputs, outputs, name="model_1_LSTM")

    model_1.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    model_1.fit(train_sentences,
                train_labels,
                validation_data=(val_sentences, val_labels),
                epochs=10)

    model_1_probs = model_1.predict(val_sentences)

    model_1_preds = tf.squeeze(tf.round(model_1_probs))

    model_1_results = calculate_results(y_true=val_labels,
                                        y_pred=model_1_preds)

    # GRU model

    model_2 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), dtype="string"),
        text_vectorizer,
        embedding,
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model_2.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    model_2.fit(train_sentences,
                train_labels,
                validation_data=(val_sentences, val_labels),
                epochs=10)

    model_2_probs = model_2.predict(val_sentences)

    model_2_preds = tf.squeeze(tf.round(model_2_probs))

    model_2_results = calculate_results(y_true=val_labels,
                                        y_pred=model_2_preds)

    # Bidirectional RNN

    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model_3 = tf.keras.Model(inputs, outputs, name="model_3_Bidir_LSTM")

    model_3.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    model_3.fit(train_sentences,
                train_labels,
                validation_data=(val_sentences, val_labels),
                epochs=10)

    model_3_probs = model_3.predict(val_sentences)

    model_3_preds = tf.squeeze(tf.round(model_3_probs))

    model_3_results = calculate_results(val_labels, model_3_preds)

    # CNN

    inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalMaxPool1D(keepdims=True)(x)
    x = tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model_4 = tf.keras.Model(inputs, outputs, name="model_4_CNN")

    model_4.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    model_4.fit(train_sentences,
                train_labels,
                validation_data=(val_sentences, val_labels),
                epochs=10)

    model_4_probs = model_4.predict(val_sentences)

    model_4_preds = tf.squeeze(tf.round(model_4_probs))

    model_4_results = calculate_results(val_labels, model_4_preds)

    # Transfer learning - Pretrained embeddings

    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sentence_encode_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                           dtype="string",
                                           name="USE")

    inputs = tf.keras.layers.Input(shape=[], dtype="string")
    x = sentence_encode_layer(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model_5 = tf.keras.Model(inputs, outputs, name="model_5_USE")

    model_5.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    model_5.fit(train_sentences,
                train_labels,
                validation_data=(val_sentences, val_labels),
                epochs=10)

    model_5_probs = model_5.predict(val_sentences)

    model_5_preds = tf.squeeze(tf.round(model_5_probs))

    model_5_results = calculate_results(val_labels, model_5_preds)

    # ~~~ NLP Binary classification - End ~~~ #
