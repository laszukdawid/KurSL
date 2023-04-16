import numpy as np

from kursl import KurSL


def test_kuramoto_ODE_generator_no_coupling():
    from scipy.integrate import ode

    W = np.array([12.0, 15.0])
    K = np.array([0, 0]).reshape((1, 1, 2))
    t = np.linspace(0, np.pi, 100)
    S = W[:, None] * t

    kODE = ode(KurSL.kuramoto_ODE)
    kODE.set_initial_value(S[:, 0], t[0])
    kODE.set_f_params((W, K))

    phase = np.zeros((S.shape))

    for idx, _t in enumerate(t[1:]):
        phase[:, idx] = kODE.y
        kODE.integrate(_t)

    phase[:, -1] = kODE.y

    assert np.allclose(phase[0], S[0]), "Phase for W=12 oscillator"
    assert np.allclose(phase[1], S[1]), "Phase for W=15 oscillator"
