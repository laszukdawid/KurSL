import numpy as np
import unittest

from kursl import ModelWrapper
from kursl import KurSL


class WrapperTest(unittest.TestCase):
    @staticmethod
    def generate_params(oscN, nH):
        W = np.random.rand(oscN, 1) * 20 + 10
        R = np.random.rand(oscN, 1) * 10
        P = np.random.rand(oscN, 1)
        K = (np.random.rand(oscN, nH * (oscN - 1)) - 0.5) * 5
        params = np.hstack((W, P, R, K))
        return params

    def test_default_values(self):
        model = ModelWrapper()
        self.assertEqual(model.MIN_R, 0)
        self.assertEqual(model.MAX_R, 1e10)
        self.assertEqual(model.MIN_W, 0)
        self.assertEqual(model.MAX_W, 1e5)
        self.assertEqual(model.THRESHOLD, 0.1)

    def test_default_distrubitions(self):
        model = ModelWrapper()
        test_val = np.array([-1, -0.5, 0, 0.5, 1])
        exp_val = np.array([0, 0, 0, 0, 0])
        self.assertTrue(np.all(model.dist_W(test_val) == exp_val))
        self.assertTrue(np.all(model.dist_R(test_val) == exp_val))
        self.assertTrue(np.all(model.dist_K(test_val) == exp_val))
        self.assertTrue(np.all(model.dist_ph(test_val) == exp_val))

    def test_initiation_with_model(self):
        oscN = 3
        nH = 1
        params = self.generate_params(oscN, nH)
        kursl = KurSL(params)
        model = ModelWrapper(kursl)

        self.assertEqual(model.model, kursl)
        self.assertEqual(model.oscN, oscN)
        self.assertEqual(model.nH, nH)

    def test_set_model_after_initiation(self):
        model = ModelWrapper()

        oscN = 3
        nH = 1
        params = np.random.random(oscN * (3 + (oscN - 1) * nH)) + 1
        params = params.reshape((oscN, -1))
        kursl = KurSL(params)
        model.set_model(kursl)

        self.assertEqual(model.model, kursl)
        self.assertEqual(model.oscN, oscN)
        self.assertEqual(model.nH, nH)

    def test_method_access_generate(self):
        oscN = 3
        nH = 1
        params = self.generate_params(oscN, nH)
        kursl = KurSL(params)
        model = ModelWrapper(kursl)

        t = np.linspace(0, 1, 200)
        s_kursl = np.array(kursl.generate(t))
        s_model = np.array(model.generate(t))
        self.assertTrue(np.all(s_kursl == s_model), "Both calls should return the same")

    def test_method_access_default_call(self):
        oscN = 3
        nH = 1
        params = self.generate_params(oscN, nH)
        t = np.linspace(0, 1, 200)
        model = ModelWrapper(KurSL(params))

        kursl = KurSL()
        s_kursl = np.array(kursl(t, params))
        s_model = np.array(model(t, params))
        self.assertTrue(np.all(s_kursl == s_model), "Both calls should return the same")
