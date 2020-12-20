"""
Code for running inference on text generation models
Reference: tensorflow documentation
"""

import pickle
import sys

import tensorflow as tf

tf.compat.v1.enable_eager_execution()


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


def generate_text(model, start_string):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 0.5

    model.reset_states()
    while ''.join(text_generated[-13:]) != '<|endoftext|>':
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string[13:] + ''.join(text_generated[:-13]))


handle = sys.argv[1]
prompt = sys.argv[2]
checkpoint_dir = f'./training_checkpoints_{handle}'

with open(f'params_{handle}.pkl', 'rb') as f:
    vocab_size, embedding_dim, rnn_units, char2idx, idx2char = pickle.load(f)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

for i in range(3):
    gen = generate_text(model, start_string=f'<|endoftext|>{prompt} ')
    print(f'* Generated #{i+1}: {gen}')
