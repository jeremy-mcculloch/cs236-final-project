import numpy as np
import matplotlib.pyplot as plt
import scipy
from dataset import generate_data_with_params, generate_data, generate_test_data, pad_values
# from models import *
import pickle
import keras
from vcann import VAE
import tensorflow as tf
from vcann_baseline import build_inv

def main():
    # # Generate Data
    # Cs, Ss, dts = generate_data(1000)
    #
    # # Save Data
    # input_data = {"stretches": Cs, "stresses": Ss, "dts": dts}
    # with open(f'./training_data.pickle', 'wb') as handle:
    #     pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load Data
    with open(f'./training_data.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    Cs = input_data["stretches"]
    Ss = input_data["stresses"]
    dts = input_data["dts"]

    # dt_orig = 0.01
    # dts = np.array([(C.shape[0] - 1) * dt_orig / 100.0 for C in Cs])
    # samples = np.arange(100) / 100.0
    # Cs = np.array([np.interp(samples, np.arange(C.shape[0]) / (C.shape[0] - 1), C) for C in Cs])
    # Ss = np.array([np.interp(samples, np.arange(S.shape[0]) / (S.shape[0] - 1), S) for S in Ss])


    # # Plot data to verify it looks good
    plot_raw_data = False
    if plot_raw_data:
        i = 2
        time = np.arange(Cs[i, :].shape[0]) * dts[i]
        # plt.plot(time, Cs[i])
        plt.plot(time, Ss[i, :])
        plt.xlabel("Time [s]")
        plt.ylabel("Stress [Pa]")
        plt.show()
        plt.plot(time, Cs[i, :])
        plt.xlabel("Time [s]")
        plt.ylabel("Stretch [-]")
        plt.show()

    ## Preprocess Data
    Cs_mean = np.mean(Cs)
    Ss_mean = np.mean(Ss)
    Cs_std = np.std(Cs)
    Ss_std = np.std(Ss)
    print(Ss.mean(axis=1))
    # Cs_scaled = (Cs - Cs_mean) / Cs_std
    Ss_scaled = (Ss) / Ss_std

    # Train VAE
    n_tau = 10
    z_dim = 32
    seq_len = 100
    vae = VAE(n_tau, z_dim, seq_len)
    dts_tiled = dts[:, np.newaxis].repeat(seq_len, axis=1)
    inputs = np.concatenate([Cs[:, :, np.newaxis], Ss_scaled[:, :, np.newaxis], dts_tiled[:, :, np.newaxis]], axis=2)
    # vae.compile(optimizer=keras.optimizers.Adam())
    # vae.fit(inputs, epochs=10000, batch_size=200)
    # vae.save_weights('./cannvae.h5')
    vae.build(input_shape=(None, seq_len, 3))
    vae.load_weights('./cannvae.h5')

    ## p(z), p(z|x1), p(z|x2)

    # p(z|x1, x2) = p(z, x1, x2) / p(x1, x2)
    # = p(z) p(x1|z) p(x2|z) / p(x1)p(x2)
    # = p(x1, z) p(x2, z) / p(x1)p(x2)p(z)
    # = p(z|x1) p(z|x2) / p(z)
    # = N(u1, s1) N(u2, s2) / N(0, 1)
    # = 1 / sqrt(2 * pi * s1^2 s2^2) * exp(-0.5 * ((x-u1)^2 / s1^2 + (x-u2)^2 / s2^2 - x^2))
    # = 1 / sqrt(2 * pi * s1^2 s2^2) * exp(-0.5 * (x^2 * (1 / s1^2 + 1 / s2^2 - 1) - x * (2u1 / s1^2 + 2u2 / s2^2) + u1^2 / s1^2 + u2^2 / s2^2)
    # = 1 / sqrt(2 * pi * s1^2 s2^2) * exp(-0.5 * ((s1^2 + s2^2 - s1^2 s2^2) / (s1^2 s2^2) * (x^2) - 2 * x * (u1s2^2 + u2s1^2) / (s1^2 + s2^2 - s1^2 s2^2)
    #           +


    # Reconstruction test on training data
    plotting = False
    if plotting:
        i = 1
        Ss_predict_scaled = vae.reconstruction_test(inputs[:, :, :])
        print(Ss_predict_scaled.shape)

        Ss_predict = Ss_std * Ss_predict_scaled

        # Plotting
        time = np.arange(seq_len) * dts[i]
        plt.plot(time, Ss[i, :], label='Training Data')
        plt.plot(time, tf.squeeze(Ss_predict[i, :]), label='Model Prediction')
        plt.xlabel('Time [s]')
        plt.ylabel('Stress [kPa]')
        plt.legend()
        plt.title("Training Set")

        plt.show()

    n_trails = 20
    all_mse_model = []
    all_mse_bl = []
    plot_model = True
    plot_bl = True
    for i in range(n_trails):
        # Reconstruction test
        Cs_test, Ss_test, dts_test = generate_test_data()

        # Cs_test_scaled = (Cs_test - Cs_mean) / Cs_std
        Ss_test_scaled = (Ss_test) / Ss_std
        dts_test_tiled = dts_test[:, np.newaxis].repeat(seq_len, axis=1)

        inputs_test = np.concatenate([Cs_test[:, :, np.newaxis], Ss_test_scaled[:, :, np.newaxis],
                                      dts_test_tiled[:, :, np.newaxis]], axis=2)
        Ss_predict_test_scaled = vae.reconstruction_test(inputs_test[:, :, :])
        Ss_predict_test = Ss_std * Ss_predict_test_scaled[0, :]

        # Plotting
        if plot_model:
            time = np.arange(seq_len) * dts_test[0]

            plt.plot(time, Ss_test[0, :], label='Data')
            plt.plot(time, Ss_predict_test, label='Model Prediction')
            plt.xlabel('Time [s]')
            plt.ylabel('Stress [Pa]')
            plt.legend()
            plt.title("VAE Reconstruction")

            plt.show()

        # Novel Data test
        Ss_predict_novel_scaled = vae.prediction_test(inputs_test)
        Ss_predict_novel = Ss_std * Ss_predict_novel_scaled[0, :]

        # Plotting
        if plot_model:
            time = np.arange(seq_len) * dts_test[1]
            plt.plot(time, Ss_test[1, :], label='Data')
            plt.plot(time, Ss_predict_novel, label='Model Prediction')
            plt.xlabel('Time [s]')
            plt.ylabel('Stress [Pa]')
            plt.legend()
            plt.title("VAE Prediction")
            plt.show()

        novel_data = Ss_test[1, :]
        print(novel_data[:].shape)
        print(Ss_predict_novel[:].shape)
        mse = tf.keras.losses.mean_squared_error(tf.squeeze(novel_data), tf.squeeze(Ss_predict_novel)).numpy()
        all_mse_model.append(mse)
        print("MSE: ", mse)


        ## baseline
        # Train on first, test on second
        inputs_train = [Cs_test[0, np.newaxis, :, np.newaxis], np.repeat(dts_test[0, np.newaxis, np.newaxis, np.newaxis], 100, axis=1)]
        outputs_train = Ss_test[0, np.newaxis, :]
        inputs_test = [Cs_test[1, np.newaxis, :, np.newaxis], np.repeat(dts_test[1, np.newaxis, np.newaxis, np.newaxis], 100, axis=1)]
        outputs_test = Ss_test[1, np.newaxis, :]

        # Train and predict
        L2 = 0.00001
        model = build_inv(5, L2, 0.000001)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
        model.fit(inputs_train, outputs_train, epochs=10000, batch_size=1)
        output_pred_test = model.predict(inputs_test)[:, :, 0]
        output_pred_train = model.predict(inputs_train)[:, :, 0]

        if plot_bl:
            # Plot results
            time = np.arange(Cs_test.shape[1]) * dts_test[0]
            plt.plot(time, Ss_test[0, :], label='Data')
            plt.plot(time, output_pred_train[0, :], label='Model Prediction')
            plt.xlabel('Time [s]')
            plt.ylabel('Stress [Pa]')
            plt.legend()
            plt.title("Baseline Training Set")
            plt.show()

            # Plot results
            time = np.arange(Cs_test.shape[1]) * dts_test[1]
            plt.plot(time, Ss_test[1, :], label='Data')
            plt.plot(time, output_pred_test[0, :], label='Model Prediction')
            plt.xlabel('Time [s]')
            plt.ylabel('Stress [Pa]')
            plt.legend()
            plt.title("Baseline Test Set")
            plt.show()

        mse = tf.keras.losses.mean_squared_error(outputs_test, output_pred_test).numpy()
        all_mse_bl.append(mse)



    print(all_mse_model)
    print(all_mse_bl)
    #
    # ## Baseline
    # # Train on one test on another
    # n_trials = 5
    # for i in range(n_trials):
    #     Cs, Ss, dts = generate_test_data()
    #
    #     # Train on first, test on second
    #     inputs_train = [Cs[0, np.newaxis, :, np.newaxis], np.repeat(dts[0, np.newaxis, np.newaxis, np.newaxis], 100, axis=1)]
    #     outputs_train = Ss[0, np.newaxis, :]
    #     inputs_test = [Cs[1, np.newaxis, :, np.newaxis], np.repeat(dts[1, np.newaxis, np.newaxis, np.newaxis], 100, axis=1)]
    #     outputs_test = Ss[1, np.newaxis, :]
    #
    #     # Train and predict
    #     L2 = 0.00001
    #     model = build_inv(5, L2, 0.000001)
    #     model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    #     model.fit(inputs_train, outputs_train, epochs=10000, batch_size=1)
    #     output_pred_test = model.predict(inputs_test)[:, :, 0]
    #     output_pred_train = model.predict(inputs_train)[:, :, 0]
    #
    #     # Plot results
    #     time = np.arange(Cs.shape[1]) * dts[0]
    #     plt.plot(time, Ss[0, :], label='Training Data')
    #     plt.plot(time, output_pred_train[0, :], label='Model Prediction')
    #     plt.xlabel('Time')
    #     plt.ylabel('Stress [Pa]')
    #     plt.legend()
    #     plt.title("Training Data Baseline")
    #     plt.show()
    #
    #     # Plot results
    #     time = np.arange(Cs.shape[1]) * dts[1]
    #     plt.plot(time, Ss[1, :], label='Test Data')
    #     plt.plot(time, output_pred_test[0, :], label='Model Prediction')
    #     plt.xlabel('Time')
    #     plt.ylabel('Stress [Pa]')
    #     plt.legend()
    #     plt.title("Test Data Baseline")
    #     plt.show()
    #
    #     mse = tf.keras.losses.mean_squared_error(outputs_test, output_pred_test).numpy()
    #     all_mse.append(mse)
    #     print("MSE: ", mse)
    # print(all_mse)
main()
