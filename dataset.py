import numpy as np
import scipy

def pad_values(inp, max_len):
    padded = np.array([np.pad(X, ((0, max_len - X.shape[0]), (0, 0)), 'constant') for X in inp])
    return padded

def generate_mat_params():
    alpha = 0.005
    gamma0 = 1e-4
    mu0 = 10 ** np.random.uniform(1, 2)
    lambdal = np.random.uniform(1.05, 1.15)
    sigma0 = 10 ** np.random.uniform(1, 2)
    n = np.random.uniform(2, 4)
    G0 = 10 ** np.random.uniform(3, 4)
    Ginf = 10 ** np.random.uniform(0, 1) * G0
    eta = 10 ** np.random.uniform(4, 5)
    roots = np.roots(np.array([1.0, 0.0, -3.0 * lambdal ** 2, 2.0]))
    roots = np.sort(roots)
    lam_a_min = roots[1]
    lam_a_max = roots[2]
    mat_params = (G0, Ginf, eta, gamma0, alpha, sigma0, n, mu0, lambdal, lam_a_min, lam_a_max)
    return mat_params

def generate_loading_params():
    loading_params = {
        "loading_mode": "uniaxial",
        "peak_stretch": np.random.uniform(1.2, 1.5),

    }
    return loading_params

def generate_exp_params(i=None):
    i = np.random.uniform(0, 2) if i is None else i
    if i < 1:
        experiment_params = {
            "exp_type": "cl",
            "n_cycles": int(np.random.uniform(2, 5)),
            "rise_time": 10 ** np.random.uniform(1, 1.5),
        }
    else:
        experiment_params = {
            "exp_type": "sr",
            "duration": np.random.uniform(50, 100),
            "rise_time": np.random.uniform(1.0, 2.0),
        }
    return experiment_params


def generate_data(n_experiments):
    all_stretches = np.zeros((n_experiments, 100))
    all_stresses = np.zeros((n_experiments, 100))
    dts = np.zeros((n_experiments,))
    for i in range(n_experiments):
        # Randomize parameters
        mat_params = generate_mat_params()
        loading_params = generate_loading_params()
        exp_params = generate_exp_params()
        stretches, stresses, dt = generate_data_with_params(loading_params, exp_params, mat_params)
        all_stretches[i, :] = stretches
        all_stresses[i, :] = stresses
        dts[i] = dt
    return all_stretches, all_stresses, dts

def generate_test_data():
    all_stretches = np.zeros((2, 100))
    all_stresses = np.zeros((2, 100))
    dts = np.zeros((2,))
    mat_params = generate_mat_params()
    seed = np.random.uniform(0, 2)

    for i in range(2):
        print(i)
        # Randomize parameters
        loading_params = generate_loading_params()
        exp_params = generate_exp_params(seed if i == 0 else 2 - seed) # Ensure 1st and second exp are different types
        stretches, stresses, dt = generate_data_with_params(loading_params, exp_params, mat_params)
        all_stretches[i, :] = stretches
        all_stresses[i, :] = stresses
        dts[i] = dt
    return all_stretches, all_stresses, dts

def generate_data_with_params(loading_params, experiment_params, mat_params):
    if experiment_params["exp_type"] == "sr":
        duration = experiment_params["duration"]
        rise_time = experiment_params["rise_time"]
        progress = lambda t: min(t / rise_time, 1.0)
        tcrits = np.array([0])
    elif experiment_params["exp_type"] == "cl":
        rise_time = experiment_params["rise_time"]
        n_cycles = experiment_params["n_cycles"]
        duration = rise_time * 2 * n_cycles
        progress = lambda t: np.abs(((t + rise_time) % (2 * rise_time)) - rise_time) / rise_time
        tcrits = np.arange(2 * n_cycles) * rise_time
    elif experiment_params["exp_type"] == "fs":
        duration = experiment_params["duration"]
        w0 = experiment_params["f_initial"] * 2 * np.pi
        w1 = experiment_params["f_final"] * 2 * np.pi
        k = np.log(w1 / w0) / duration
        progress = lambda t: (1 - np.cos(w0 / k * (np.exp(k * t) - 1))) / 2
    else:
        raise NotImplementedError("Experiment type not implemented")


    if loading_params["loading_mode"] == "uniaxial":
        stretch_max = loading_params["peak_stretch"]
        stretch = lambda t: progress(t) * (stretch_max - 1) + 1
        ts = np.arange(100) / 99.0 * duration
        dt = ts[1] - ts[0]
        def update(t, y):
            stretch_b_dot, stretch_d_dot, sigma = compute_derivatives_uniaxial(stretch(t), y[0], y[1], mat_params)
            return np.array([stretch_b_dot, stretch_d_dot])

        y0 = np.array([1.0, 1.0])
        # ys = scipy.integrate.odeint(update, y0, ts, h0=1e-8, tcrit=tcrits, )
        ode = scipy.integrate.ode(update)
        ode.set_integrator('vode', method='bdf')
        ode.set_initial_value(y0, 0.0)
        ys = np.array([y0] + [ode.integrate(ode.t+dt) for i in range(99)])

        stretches = np.array([stretch(ts[i]) for i in range(ts.shape[0])])
        Ps = np.zeros_like(ts)
        for i in range(Ps.shape[0]):
            _, _, Ps[i] = compute_derivatives_uniaxial(stretches[i], ys[i, 0], ys[i, 1], mat_params)

        if np.isnan(np.mean(Ps)):
            print(experiment_params["exp_type"])
            print(ts[np.sum(1 - np.isnan(Ps))])
            print(experiment_params["rise_time"])

    elif loading_params["loading_mode"] == "biaxial":
        pass
        # stretch_max_x = loading_params["peak_stretch_x"]
        # stretch_max_y = loading_params["peak_stretch_y"]
        # stretch_x = progress * (stretch_max_x - 1) + 1
        # stretch_y = progress * (stretch_max_y - 1) + 1
        # for i in range(nts):
        #     C[i, :, :] = np.diag(np.array([stretch_x[i] ** 2, stretch_y[i] ** 2, 1 / stretch_x[i] ** 2 / stretch_y[i] ** 2]))
    elif loading_params["loading_mode"] == "shear":
        pass
        # shear_max = loading_params["peak_stretch"]
        # shear = progress * shear_max
        # for i in range(nts):
        #     F = np.eye(3)
        #     F[0, 1] = shear[i]
        #     C[i, :, :] = F.T @ F
    else:
        raise NotImplementedError("Loading mode not implemented")

    return stretches, Ps, dt


# # assume constant stretch rate
# def generate_uniaxial_stress_relaxation(stretch_rate, stretch_max, duration, mat_params, dt):
#     nts = round(duration / dt) + 1
#     t_rise = abs(stretch_max - 1) / stretch_rate
#     C = np.zeros((nts, 3, 3))
#     for i in range(nts):
#         t = i * dt
#         progress = min(t / t_rise, 1.0)
#         stretch = 1 + progress * (stretch_max - 1)
#         C[i, :, :] = np.diag(np.array([stretch ** 2, 1 / stretch, 1 / stretch]))
#     S = compute_stresses(C, mat_params, dt)
#     return C, S
#
# def generate_uniaxial_stress_cycle(stretch_rate, stretch_max, n_cycles, mat_params, dt):
#     t_rise = abs(stretch_max - 1) / stretch_rate
#     nts = int(t_rise / dt)
#     C = np.zeros((nts, 3, 3))
#     for i in range(nts):
#         t = i * dt
#         progress = min(t / t_rise, 1.0)
#         stretch = 1 + progress * (stretch_max - 1)
#         C[i, :, :] = np.diag(np.array([stretch ** 2, 1 / stretch, 1 / stretch]))
#     C = np.concatenate((C, np.flip(C, axis=0)))
#     C = np.tile(C, (n_cycles, 1, 1))
#     print(C.shape)
#     S = compute_stresses(C, mat_params, dt)
#     return C, S

def sigmoid(x):
    return 1/ (1 + np.exp(-x))
def compute_derivatives_uniaxial(stretch, stretch_b, stretch_d, mat_params):

    G0, Ginf, eta, gamma0, alpha, sigma0, n, mu0, lambdal, lam_a_min, lam_a_max = mat_params

    # Assume inputs are numpy
    stretch_a = stretch / stretch_b
    stretch_c = stretch_b / stretch_d
    # SLS model
    Sc = 2 * G0 * np.log(stretch_c)
    Se = 2 * Ginf * np.log(stretch_d)
    stretch_d_dot = stretch_d / np.sqrt(2) / eta * (Sc - Se)

    # Nonlinear elasticity
    lam = np.sqrt((stretch_a ** 2 + 2 / stretch_a)/ 3)
    Linv = lambda x: x * (3 - x ** 2) / (1 - x ** 2)
    # print("a")
    if np.abs(lam / lambdal) > 1:
        print(lam / lambdal)
        print("Numerical instability detected, stretch increased too rapidly")
        lam = 0.999 * lambdal * np.sign(lam)
        # assert False
    Ta = mu0 * lambdal / lam * Linv(lam / lambdal) * 2/3 * (stretch_a ** 2 - 1 / stretch_a)
    sigma = Ta * 3/2 / stretch

    # Nonlinear viscosity
    Tb = Ta - Sc / 3 * (2 * stretch_a ** 2 + 1 / stretch_a)
    sqrtTb2Trace = Tb * np.sqrt(3/2)
    fR = alpha ** 2 / (alpha + np.sqrt((stretch_b ** 2 + 2 / stretch_b) / 3) - 1) ** 2
    stretch_b_dot = gamma0 * fR * np.sign(sqrtTb2Trace) * np.abs(sqrtTb2Trace / (sigma0 * np.sqrt(2.))) ** n * np.sqrt(2/3) * stretch_b

    return stretch_b_dot, stretch_d_dot, sigma

# Compute gradient
def compute_derivatives(C, Fb, Fd, mat_params):
    # Assume inputs are numpy
    G0, Ginf, eta, gamma0, alpha, sigma0, n, mu0, lambdal = mat_params
    Ca = np.linalg.inv(Fb.T) @ C @ np.linalg.inv(Fb)

    # SLS model
    Fc = Fb @ np.linalg.inv(Fd)
    Sc = G0 * scipy.linalg.logm(Fc @ Fc.T)
    Se = Ginf * scipy.linalg.logm(Fd @ Fd.T)
    Sd = Sc - Fc @ Se @ Fc.T
    Sdprime = Sd - np.eye(3) * np.trace(Sd) / 3

    Fddot = 1 / eta / np.sqrt(2) * np.linalg.inv(Fc) @ Sdprime @ Fb

    # Nonlinear elasticity
    lam = np.sqrt(np.trace(Ca) / 3)
    Linv = lambda x: x * (3 - x ** 2) / (1 - x ** 2)
    if np.abs(lam / lambdal) > 1:
        print(lam)
        print("Numerical instability detected, stretch increased too rapidly")
        assert False
    Sa = mu0 * lambdal / lam * Linv(lam / lambdal) * (np.eye(3) - lam ** 2 * np.linalg.inv(Ca))

    # Nonlinear viscosity
    Sb = Sa - Sc
    Sb = Sb - np.linalg.inv(Ca) / 3 * np.trace(Ca @ Sb)
    trSb2 = np.trace(Sb @ Sb.T)
    fR = alpha ** 2 / (alpha + np.sqrt(np.trace(Fb @ Fb.T) / 3) - 1) ** 2
    gammaBdot = gamma0 * fR * (np.sqrt(trSb2)) ** (n - 1) / (sigma0 * np.sqrt(2.)) ** n
    Fbdot = gammaBdot * Sb @ Ca @ Fb
    errorb = np.trace(Sb @ Ca)
    errord = np.trace(Sc)
    # print("ace")
    # print(errorb)
    # print(errord)
    # print(G0 * 2 * np.log(np.linalg.det(Fc)))
    # print(np.trace(Se))
    return Fbdot, Fddot, Sa

# Generate Data
def compute_stresses(Cs, mat_params, dt):
    nts = Cs.shape[0]
    Fb = np.eye(3)
    Fd = np.eye(3)
    S = np.zeros_like(Cs)

    stress = 0
    for i in range(nts):
        C = Cs[i, :, :]
        Fbdot1, Fddot1, _ = compute_derivatives(C, Fb, Fd, mat_params)
        Fb2 = Fb + dt / 2 * Fbdot1
        Fd2 = Fd + dt / 2 * Fddot1
        Fbdot2, Fddot2, _ = compute_derivatives(C, Fb2, Fd2, mat_params)
        Fb3 = Fb + dt / 2 * Fbdot2
        Fd3 = Fd + dt / 2 * Fddot2
        Fbdot3, Fddot3, _ = compute_derivatives(C, Fb3, Fd3, mat_params)
        Fb4 = Fb + dt * Fbdot3
        Fd4 = Fd + dt * Fddot3
        Fbdot4, Fddot4, _ = compute_derivatives(C, Fb4, Fd4, mat_params)
        Fb = Fb + dt / 6 * (Fbdot1 + 2 * Fbdot2 + 2 * Fbdot3 + Fbdot4)
        Fd = Fd + dt / 6 * (Fddot1 + 2 * Fddot2 + 2 * Fddot3 + Fddot4)
        Fb = Fb * np.linalg.det(Fb) ** (-1 / 3)
        Fd = Fd * np.linalg.det(Fd) ** (-1 / 3)
        _, _, Sa = compute_derivatives(C, Fb, Fd, mat_params)
        S[i, :, :] = Sa

        new_stress = Sa[0, 0] - Sa[1, 1]
        stress = new_stress
        # if i % 100 == 0:
        #     print(i)
            # print(np.linalg.det(Fb))
            # print(np.linalg.det(Fd))

    return S