import numpy as np
import matplotlib.pyplot as plt
import scipy
from dataset import generate_data_with_params, generate_data, generate_test_data, pad_values
from models import *
import pickle

def main_plotting():

    alpha = 0.005
    gamma0 = 1e-4
    mu0 = 20.
    lambdal = 1.09
    sigma0 = 25.
    n = 3.0
    G0 = 4500.
    Ginf = 600.
    eta = 60000.
    mat_params = (G0, Ginf, eta, gamma0, alpha, sigma0, n, mu0, lambdal)

    # t = 10.
    # dt = 0.001
    # stretch_max = 1.5
    # stretch_rate = 1.
    #
    # C, S = generate_uniaxial_stress_relaxation(stretch_rate, stretch_max, t, mat_params, dt)
    dt = 0.01

    loading_params = {
        "loading_mode": "uniaxial",
        "peak_stretch": 1.5,

    }
    experiment_params = {
        "exp_type": "cl",
        "n_cycles": 3,
        "rise_time": 5.0,
    }
    # experiment_params = {
    #     "exp_type": "fs",
    #     "duration": 20.0,
    #     "f_initial": 1.5,
    #     "f_final": 15.0
    # }
    C, S = generate_data_with_params(loading_params, experiment_params, mat_params, dt)
    # C, S = generate_uniaxial_stress_cycle(stretch_rate, stretch_max, n_cycles, mat_params, dt)
    axial_stress = S[:, 0, 0] - S[:, 1, 1]
    stretch = np.sqrt(C[:, 0, 0])
    ts = np.arange(0, C.shape[0]) * dt
    # plt.plot(ts, axial_stress)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Stress [Pa]")
    # plt.show()

    plt.plot(ts, stretch)
    plt.xlabel("Time [s]")
    plt.ylabel("Stretch [-]")
    plt.show()

def main():
    # Generate Data
    # Cs, Ss = generate_data(20)
    #
    # # Save Data
    # input_data = {"Cs": Cs, "Ss": Ss}
    # with open(f'./training_data.pickle', 'wb') as handle:
    #     pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'./training_data.pickle', 'rb') as handle:
        input_data = pickle.load(handle)
    Cs = input_data["Cs"]
    Ss = input_data["Ss"]

    Cs = [C[::100] for C in Cs]
    Ss = [S[::100] for S in Ss]

    print(sum([sum(np.isinf(C)) for C in Cs]))
    print(sum([sum(np.isinf(S)) for S in Ss]))

    # Plot data to verify it looks good
    dt = 0.01
    # time = np.arange(Cs[10].shape[0]) * dt
    # plt.plot(time, Ss[10])
    # plt.xlabel("Time [s]")
    # plt.ylabel("Stress [-]")
    # plt.show()
    # input("Enter to continue")

    # Train VAE
    # vae = train_vae(Cs, Ss)
    vae = load_model(Cs, Ss)

    # Reconstruction test on training data
    max_len = vae.seq_length
    i = 1
    Cs_train = pad_values(Cs[i][np.newaxis, :, np.newaxis], max_len)
    Ss_train = pad_values(Ss[i][np.newaxis, :, np.newaxis], max_len) / 1000

    inputs = np.concatenate((Cs_train, Ss_train), axis=-1)
    print(inputs)
    print(inputs.shape)
    Ss_predict = vae.call(inputs)
    Ss_predict = Ss_predict[0, 0:Cs[i].shape[0], 0]

    # Plotting
    time = np.arange(Cs[i].shape[0]) * dt * 100
    plt.plot(time, Ss[i] / 1000, label='Training Data')
    plt.plot(time, Ss_predict, label='Model Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stress [kPa]')
    plt.legend()
    plt.show()

    # Reconstruction test
    Cs_test, Ss_test = generate_test_data()
    Cs_test = [C[::100] for C in Cs_test]
    Ss_test = [S[::100]/1000 for S in Ss_test]
    max_len = vae.seq_length
    Cs_test0 = pad_values(Cs_test[0][np.newaxis, :, np.newaxis], max_len)
    Cs_test1 = pad_values(Cs_test[1][np.newaxis, :, np.newaxis], max_len)
    Ss_test0 = pad_values(Ss_test[0][np.newaxis, :, np.newaxis], max_len)

    inputs = np.concatenate((Cs_test0, Ss_test0), axis=-1)
    Ss_predict = vae.call(inputs)
    Ss_predict = Ss_predict[0, 0:Cs_test[0].shape[0], 0]

    # Plotting
    time = np.arange(Cs_test[0].shape[0]) * dt * 100
    plt.plot(time, Ss_test[0], label='Test Data')
    plt.plot(time, Ss_predict, label='Model Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stress [Pa]')
    plt.legend()
    plt.show()

    # Novel Data test
    latent = vae.encode(inputs)
    Ss_predict = vae.decode(Cs_test1, latent)
    Ss_predict = Ss_predict[0, 0:Cs_test[1].shape[0], 0]

    # Plotting
    time = np.arange(Cs_test[1].shape[0]) * dt * 100
    plt.plot(time, Ss_test[1], label='Test Data')
    plt.plot(time, Ss_predict, label='Model Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stress [Pa]')
    plt.legend()
    plt.show()
    input("Enter to continue")

main()
