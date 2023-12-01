import numpy as np
import tensorflow as tf
from keras import layers
import keras
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# custom RNN cell to learn Prony series parameters
# Modified so tau parameters are taken as input from latent space
class ViscRNNCellGen(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.state_size = [tf.TensorShape([1]), tf.TensorShape([units])]
        self.units = units
        super(ViscRNNCellGen, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):

        scaleFactor = tf.constant([1000.])  # scale factor can be adjusted to see if training improves

        (sig0_prev, h_prev) = states  # stored from previous time step

        dt, sig0, tau = tf.split(inputs, num_or_size_splits=[1, 1, self.units], axis=1)

        tauPos = tf.nn.relu(tau) + tf.constant([1e-5])  # constrain parameters to be positive

        a = tf.math.exp(tf.math.divide(tf.math.multiply(tf.constant([-1.]), dt), tauPos * scaleFactor))

        b = tf.math.exp(tf.math.divide(tf.math.multiply(tf.constant([-1.]), dt),
                                       tf.math.multiply(tf.constant([2.]), tauPos * scaleFactor)))

        dsig = tf.math.subtract(sig0, sig0_prev)

        h = tf.math.add(tf.math.multiply(a, h_prev), tf.math.multiply(b, dsig))

        output = tf.concat([sig0, h], 1)

        return output, (sig0, h)

# %%%% Invariant-based model %%%%

# exponential activation function
def activation_Exp(x):
    return 1.0 * (tf.math.exp(x) - 1.0)


# logarithmic activation function
def activation_ln(x):
    return -1.0 * tf.math.log(1.0 - (x))

# gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]


# assembles invariant-based terms for the initial stored energy function
#  I_ref: either I1-3 or I2-3
#  L2: regularization strength
def SingleInvNet6(I_ref, theta_raw): # Assume theta > 0
    theta = tf.exp(theta_raw)
    theta_split = tf.split(theta, num_or_size_splits=10, axis=2)
    exp_scale = 0.00001
    log_scale = 0.00001
    # linear terms
    # Differentiate by hand
    I_w11 = theta_split[0]
    I_w21 = theta_split[1] * exp_scale * theta_split[6] * tf.math.exp(exp_scale * theta_split[6] * I_ref)
    I_w31 = theta_split[2] / (1 - log_scale * theta_split[7] * I_ref) * log_scale * theta_split[7]
    I_w41 = theta_split[3] * I_ref * 2
    I_w51 = theta_split[4] * exp_scale * theta_split[8] * 2 * I_ref * tf.math.exp(exp_scale * theta_split[8] * I_ref ** 2)
    I_w61 = theta_split[5] / (1 - log_scale * theta_split[9] * I_ref ** 2) * 2 * log_scale * theta_split[9] * I_ref
    collect = [I_w11, I_w21, I_w31, I_w41, I_w51, I_w61]
    collect_out = tf.keras.layers.concatenate(collect)
    return collect_out



# Calculation of Cauchy stress for uniaxial tension/compression only
#  inputs: (dPsidI1, dPsidI2, Stretch, I1)
#   dPsidI1: partial derivative of the initial stored energy function w/ respect to the first invariant I1
#   dPsidI2: partial derivative of the initial stored energy function w/ respect to I2
#   Stretch: the current axial stretch
#   I1: the first invariant
def Stress_calc_UT(inputs):

    (dPsidI1, dPsidI2, Stretch, I1) = inputs

    one = tf.constant(1.0, dtype='float32')
    two = tf.constant(2.0, dtype='float32')

    minus = two * (dPsidI1 * one / tf.math.square(Stretch) + dPsidI2 * one / tf.math.pow(Stretch, 3))
    P = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus

    sig = P * Stretch

    return sig


# build the invariant-based RNN model
#  n: number of terms in the relaxation function
#  l2: regularization weight for the initial stored energy function
#  rp: regularization weight for the prony series relaxation function
def build_inv(n):
    cell = ViscRNNCellGen(n)  # viscoelastic model RNN cell

    stretch = keras.layers.Input(shape=(None, 1), name='input_stretch')  # input: axial stretch
    dt = keras.layers.Input(shape=(None, 1), name='time_step')  # input: time step
    all_params = keras.layers.Input(shape=(None, 21 + 2 * n), name='all_params')
    theta_I1, theta_I2, tau, alpha = tf.split(all_params, num_or_size_splits=[10, 10, n, n + 1], axis=2)

    # calculate the invariants
    I1 = keras.layers.Lambda(lambda x: x ** 2 + 2.0 / x, name='I1')(stretch)
    I2 = keras.layers.Lambda(lambda x: 2.0 * x + 1 / x ** 2, name='I2')(stretch)


    I1_ref = keras.layers.Lambda(lambda x: (x - 3.0), name='I1_ref')(I1)
    I2_ref = keras.layers.Lambda(lambda x: (x - 3.0), name='I2_ref')(I2)

    dPsidI1 = tf.reduce_sum(SingleInvNet6(I1_ref, theta_I1), axis=-1, keepdims=True)
    dPsidI2 = tf.reduce_sum(SingleInvNet6(I2_ref, theta_I2), axis=-1, keepdims=True)

    # normal initial Cauchy stress in axial direction
    sig0 = keras.layers.Lambda(function=Stress_calc_UT, name='init_Stress')([dPsidI1, dPsidI2, stretch, I1])


    merge = keras.layers.Concatenate(name='rnn_input')([dt, sig0, 10 * tf.exp(tau)])
    # the relaxation function RNN
    rnnOutput = keras.layers.RNN(cell, return_sequences=True, name='relax_function')(merge)

    # calculating the total Cauchy stress
    out = tf.reduce_sum(rnnOutput * tf.nn.softmax(alpha), axis=-1, keepdims=True)

    model = keras.models.Model(inputs=[stretch, dt, all_params], outputs=[out, I1])

    return model

def build_decoder(n_tau, z_dim, seq_len):
    stretch = keras.layers.Input(shape=(seq_len, 1), name='input_stretch')  # input: axial stretch
    dt = keras.layers.Input(shape=(seq_len, 1 ), name='time_step')  # input: time step
    latent = keras.layers.Input(shape=(z_dim,), name='latent')

    vcann = build_inv(n_tau)

    param_model = keras.Sequential(
    [
        layers.Dense(100, activation="relu", name="layer1"),
        layers.Dense(100, activation="relu", name="layer2"),
        layers.Dense(100, activation="relu", name="layer3"),
        layers.Dense(2 * n_tau + 21, name="layer4"),
        layers.RepeatVector(seq_len)
    ])

    all_params = param_model(latent)
    stress, sig0 = vcann([stretch, dt, all_params])
    return keras.models.Model(inputs=[stretch, dt, latent], outputs=[stress])

def build_encoder(z_dim, seq_len):
    stretch = keras.layers.Input(shape=(seq_len, 1), name='input_stretch')  # input: axial stretch
    stress = keras.layers.Input(shape=(seq_len, 1), name='input_stress')  # input: axial stress
    dt = keras.layers.Input(shape=(seq_len, 1), name='time_step')  # input: time step
    input = layers.Concatenate()([stress, stress, dt])
    cnn = keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=(seq_len, 3)),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(16, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2 * z_dim)
    ])
    z_params = cnn(input)
    z_mean, z_log_var = tf.split(z_params, num_or_size_splits=[z_dim, z_dim], axis=1)

    z = Sampling()([z_mean, z_log_var])

    return keras.models.Model(inputs=[stretch, stress, dt], outputs=[z_mean, z_log_var, z])

class VAE(keras.Model):
    def __init__(self, n_tau, z_dim, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_encoder(z_dim, seq_len)
        self.decoder = build_decoder(n_tau, z_dim, seq_len)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.test_tracker = keras.metrics.Mean(name="test")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.test_tracker
        ]

    def train_step(self, data):
        stretch, stress, dt = tf.split(data, num_or_size_splits=3, axis=2)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([stretch, stress, dt])
            stress_pred = self.decoder([stretch, dt, z])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(stress, stress_pred), axis=1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss * 0.1
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()

        }

    def call(self, inputs, training=None, mask=None):
        return self.reconstruction_test(inputs)
    def reconstruction_test(self, data):
        stretch, stress, dt = tf.split(data, num_or_size_splits=3, axis=2)
        z_mean, z_log_var, z = self.encoder([stretch, stress, dt])
        stress_pred = self.decoder([stretch, dt, z_mean])
        return stress_pred

    def prediction_test(self, data):
        stretch = data[0:1, :, 0]
        stress = data[0:1, :, 1]
        dt = data[0:1, :, 2]
        z_mean, z_log_var, z = self.encoder([stretch, stress, dt])
        stretch_new = data[1:, :, 0]
        dt_new = data[1:, :, 2]
        stress_pred = self.decoder([stretch_new, dt_new, z_mean])
        return stress_pred
# # %%%% vanilla RNN model %%%%
#
# # build the vanilla RNN model with 1 LSTM layer of 8 hidden units
# def build_rnn():
#     visible1 = layers.Input(shape=(None, 2))
#     hidden1 = layers.LSTM(8, return_sequences=True)(visible1)
#     outputLayer = layers.TimeDistributed(layers.Dense(1))(hidden1)
#
#     model = keras.Model(inputs=visible1, outputs=outputLayer)
#
#     return model