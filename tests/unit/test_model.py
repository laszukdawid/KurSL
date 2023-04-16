import numpy as np

from kursl import ModelWrapper
from kursl import KurSL


def generate_params(oscN, nH):
    W = np.random.rand(oscN, 1) * 20 + 10
    R = np.random.rand(oscN, 1) * 10
    P = np.random.rand(oscN, 1)
    K = (np.random.rand(oscN, nH * (oscN - 1)) - 0.5) * 5
    params = np.hstack((W, P, R, K))
    return params


def test_default_values():
    model = ModelWrapper()
    assert model.MIN_R == 0
    assert model.MAX_R == 1e10
    assert model.MIN_W == 0
    assert model.MAX_W == 1e5
    assert model.THRESHOLD == 0.1


def test_default_distrubitions():
    model = ModelWrapper()
    test_val = np.array([-1, -0.5, 0, 0.5, 1])
    exp_val = np.array([0, 0, 0, 0, 0])
    assert np.all(model.dist_W(test_val) == exp_val)
    assert np.all(model.dist_R(test_val) == exp_val)
    assert np.all(model.dist_K(test_val) == exp_val)
    assert np.all(model.dist_ph(test_val) == exp_val)


def test_initiation_with_model():
    oscN = 3
    nH = 1
    params = generate_params(oscN, nH)
    kursl = KurSL(params)
    model = ModelWrapper(kursl)

    assert model.model == kursl
    assert model.oscN == oscN
    assert model.nH == nH


def test_set_model_after_initiation():
    model = ModelWrapper()

    oscN = 3
    nH = 1
    params = np.random.random(oscN * (3 + (oscN - 1) * nH)) + 1
    params = params.reshape((oscN, -1))
    kursl = KurSL(params)
    model.set_model(kursl)

    assert model.model == kursl
    assert model.oscN == oscN
    assert model.nH == nH


def test_method_access_generate():
    oscN = 3
    nH = 1
    params = generate_params(oscN, nH)
    kursl = KurSL(params)
    model = ModelWrapper(kursl)

    t = np.linspace(0, 1, 200)
    s_kursl = np.array(kursl.generate(t))
    s_model = np.array(model.generate(t))
    assert np.all(s_kursl == s_model), "Both calls should return the same"


def test_method_access_default_call():
    oscN = 3
    nH = 1
    params = generate_params(oscN, nH)
    t = np.linspace(0, 1, 200)
    model = ModelWrapper(KurSL(params))

    kursl = KurSL()
    s_kursl = np.array(kursl(t, params))
    s_model = np.array(model(t, params))
    assert np.all(s_kursl == s_model), "Both calls should return the same"
