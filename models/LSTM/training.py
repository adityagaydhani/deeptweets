"""
Code for LSTM text generation using Tensorflow
Reference: tensorflow documentation
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


handle = sys.argv[1]

df = pd.read_csv(f'../../data/{handle}.csv')

text = '<|endoftext|>'
text += '<|endoftext|>'.join(df.tweet) + '<|endoftext|>'

print(f'{len(text)} characters in total.')
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence you want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


print(model.summary())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = f'./training_checkpoints_{handle}'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    monitor='loss',
    save_best_only=True)

EPOCHS = 15
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

print(tf.train.latest_checkpoint(checkpoint_dir))

with open(f'params_{handle}.pkl', 'wb') as f:
    pickle.dump([vocab_size, embedding_dim, rnn_units, char2idx, idx2char], f)
