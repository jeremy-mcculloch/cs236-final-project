import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Masking, Layer
from keras import Model
import keras
import tensorflow as tf


class Encoder(Layer):
    def __init__(self, seq_length, x_dim, y_dim, latent_dim):
        super().__init__()
        self.mask_layer = Masking(mask_value=0., input_shape=(seq_length, x_dim + y_dim))
        self.rnn_layer = LSTM(latent_dim, activation='relu', return_sequences=False)
        self.repeat_layer = RepeatVector(seq_length)
    def call(self, inputs, *args, **kwargs):
        masked = self.mask_layer(inputs)
        last_latent = self.rnn_layer(masked)
        output = self.repeat_layer(last_latent)

        return output
    def get_config(self):
        return super(Encoder, self).get_config()

class Decoder(Layer):
    def __init__(self, seq_length, x_dim, y_dim, latent_dim):
        super().__init__()
        self.rnn_layer = LSTM(latent_dim, activation='relu', return_sequences=True)
        self.dense = TimeDistributed(Dense(y_dim))

    def call(self, inputs, *args, **kwargs):
        latents = self.rnn_layer(inputs)
        output = self.dense(latents)
        return output

    def get_config(self):
        return super(Decoder, self).get_config()


class RNNVAE(Model):
    def __init__(self, seq_length, x_dim, y_dim, latent_dim):
        super().__init__()
        self.seq_length = seq_length
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.enc = Encoder(seq_length, x_dim, y_dim, latent_dim)
        self.dec = Decoder(seq_length, x_dim, y_dim, latent_dim)

    def call(self, inputs):
        encoded = self.enc(inputs)
        x = inputs[:, :, 0:self.x_dim]
        dec_inp = tf.concat((x, encoded), axis=-1)
        yhat = self.dec(dec_inp)
        return yhat

    def encode(self, inputs):
        latent = self.enc(inputs)
        return latent[:, 0, :]

    def decode(self, x, latent):
        seq_len = x.shape[1]
        latent_repeated = np.tile(latent, (1, seq_len, 1))
        dec_inp = np.concatenate((x, latent_repeated), axis=-1)
        yhat = self.dec(dec_inp)
        return yhat


    def train(self, xs, ys, epochs=10, batch_size=32, validation_split=0.2):
        # Create a mask for padded values (1 for valid values, 0 for padded values)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
        input_data = np.concatenate((xs, ys), axis=-1)
        mask = np.where(np.sum(input_data, axis=-1) != 0, 1, 0)
        self.fit(input_data, ys, epochs=epochs, batch_size=batch_size, validation_split=validation_split, sample_weight=mask, callbacks=callback)

# data is list of
def train_vae(Cs, Ss, epochs=5000, should_save=True):
    # Hack since dims aren't right
    Cs = [C.reshape((-1, 1)) for C in Cs]
    Ss = [S.reshape((-1, 1)) / 1000 for S in Ss]

    # Find the maximum sequence length
    max_len = max(C.shape[0] for C in Cs)

    # Pad sequences with zeros to make them equal length
    padded_C = np.array([np.pad(C, ((0, max_len - len(C)), (0, 0)), 'constant') for C in Cs])
    padded_S = np.array([np.pad(S, ((0, max_len - len(S)), (0, 0)), 'constant') for S in Ss])

    # Create model
    rnnvae = RNNVAE(max_len, Cs[0].shape[1], Cs[0].shape[1], latent_dim=64)
    optimizer = keras.optimizers.Adam(lr=0.0001, clipvalue=0.1)


    rnnvae.compile(optimizer=optimizer, loss='mse')
    # rnnvae.summary()
    rnnvae.train(padded_C, padded_S, epochs=epochs, batch_size=30, validation_split=0)
    if should_save:
        rnnvae.save_weights('./rnnvae.h5')
    return rnnvae

def load_model(Cs, Ss):
    rnnvae = train_vae(Cs, Ss, epochs=1, should_save=False)
    rnnvae.load_weights('./rnnvae.h5')
    return rnnvae
