import numpy as np
import unittest

from kursl import KurSL


class KurslGeneralTest(unittest.TestCase):
    def test_correct_parameter_initiation(self):
        oscN = 3
        nH = 2

        #    W     Ph    A|    K1   |   K2
        P = [
            [11, 0, 1, 1, -1, 0.1, 2.3],
            [20, 0.2, 2, 1, 2, 0.5, 3.1],
            [29, 1.5, 2.5, 0.1, -1.7, -10, 0],
        ]
        P = np.array(P)
        kursl = KurSL(P)

        self.assertEqual(kursl.oscN, oscN, "Number of oscillators")
        self.assertEqual(kursl.nH, nH, "Order of model")

    def test_kuramoto_ODE_generator_no_coupling(self):
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

        self.assertTrue(np.allclose(phase[0], S[0]), "Phase for W=12 oscillator")
        self.assertTrue(np.allclose(phase[1], S[1]), "Phase for W=15 oscillator")
