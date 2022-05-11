# تحلیل بورسی

[The Question](https://quera.org/problemset/79504/)

Steps we took:
1. Encode the data using VADER sentiment analyser.
2. Create a sequence-based dataset (`seq_len = 10`).
3. Train the model using a simple stacked LSTM model.
4. Save the predictions.

NOTE: since we are using a sequence-based dataset, the first `seq_len - 1` observations
will be omitted. So, we insert `seq_len - 1` random predictions in the beginning. 