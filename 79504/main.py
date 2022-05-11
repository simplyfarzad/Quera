import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import Fast
from VADER import sentiment_vader

FAST = True  # To run the sentiment analysis in parallel
SENTIMENT = True  # Has the sentiment been extracted from the data?

# LOAD THE DATA
train_data_uri = "Data\\train_sentiment.csv" if SENTIMENT else "Data\\train.csv"
test_data_uri = "Data\\test_sentiment.csv" if SENTIMENT else "Data\\test.csv"

train_data = pd.read_csv(train_data_uri, index_col=['Date'], parse_dates=True)
test_data = pd.read_csv(test_data_uri, index_col=['Date'], parse_dates=True)

# PREPROCESS
X_train = train_data.drop(['Label'], axis=1)
y_train = train_data['Label']
X_test = test_data

# Fill NAN values with '0', since the `sentiment_vader` cannot process NAN or numeric values
X_train.fillna('0', inplace=True)

if not SENTIMENT:
    # EXTRACT SENTIMENT FROM THE NEWS AND SAVE IT
    if FAST:
        # Use the year as a measure for grouping the data, so we can parallelize the `applymap` function
        X_train['year'] = X_train.index.year
        X_test['year'] = X_test.index.year


        def foo(single_year: pd.DataFrame):
            return single_year.drop(['year'], axis=1).applymap(sentiment_vader)


        X_train = Fast.parallel_apply(X_train.groupby(['year']), foo)
        X_test = Fast.parallel_apply(X_test.groupby(['year']), foo)
    else:
        X_train = X_train.applymap(sentiment_vader)
        X_test = X_test.applymap(sentiment_vader)

    X_train_to_save = X_train.copy()
    X_train_to_save['Label'] = y_train
    X_train_to_save.to_csv("Data\\train_sentiment.csv")

    X_test.to_csv("Data\\test_sentiment.csv")

# Add sequence
seq_len = 10
X_train_seq = np.empty(shape=(X_train.shape[0], seq_len, X_train.shape[1]), dtype=np.float64)
X_test_seq = np.empty(shape=(X_test.shape[0], seq_len, X_test.shape[1]), dtype=np.float64)

for i in range(seq_len):
    X_train_seq[:, i, :] = X_train.shift(-i)
    X_test_seq[:, i, :] = X_test.shift(-i)

# Drop the final NANs
X_train_seq = X_train_seq[:-9, :, :]
y_train = y_train[9:]
X_test_seq = X_test_seq[:-9, :, :]

# SPLIT THE TRAIN AND VALIDATION DATA
np.random.choice(range(X_train_seq.shape[0]), X_train_seq.shape[0])

# The Model
model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
)

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_seq, y_train, epochs=50, batch_size=64)

# Since we are using a sequence to train our model, the first `seq_len - 1` observations will be omitted.
# We compensate for that by adding `seq_len - 1` random (0 or 1) predictions to the final result.
initial_random_predictions = np.random.choice([0, 1], seq_len - 1)
predictions = np.round(model.predict(X_test_seq))
predictions = np.append(initial_random_predictions, predictions)

np.savetxt("Data\\output.csv", predictions.astype(int), fmt='%i', delimiter='\r\n')
