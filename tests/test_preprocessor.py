import pytest
import numpy as np

from kursl import Preprocessor


def peak_norm(t, t0, amp, s):
    _t = (t - t0) / s
    return amp * np.exp(-_t * _t)


def peak_triangle(t, t0, amp, s):
    x = amp * (1 - np.abs((t - t0) / s))
    x[x < 0] = 0
    return x


def peak_lorentz(t, t0, amp, s):
    _t = (t - t0) / (0.5 * s)
    return amp / (_t * _t + 1)


def _cos(t, f, a, ph):
    return a * np.cos(f * 2 * np.pi * t + ph)


def test_remove_peak_norm():
    """Test removing single peak as a Gaussian peak"""
    # If too small segment then it doesn't converge properly
    t = np.arange(0, 2, 0.001)

    param_in = np.array([1.2, 0.4, 0.1])
    peak_in = peak_norm(t, param_in[0], param_in[1], param_in[2])

    peak_out, param_out = Preprocessor._remove_peak(t, peak_in, ptype="norm")
    assert np.allclose(param_in, param_out), "Parameters should be the same"
    assert np.allclose(peak_in, peak_out), "Peaks should be the same"


def test_remove_peak_triangle():
    """Test removing single peak as a triangle peak"""
    t = np.arange(0, 2, 0.001)

    param_in = np.array([0.5, 1.5, 0.3])
    peak_in = peak_triangle(t, param_in[0], param_in[1], param_in[2])

    peak_out, param_out = Preprocessor._remove_peak(t, peak_in, ptype="triang")
    assert np.allclose(param_in, param_out), "Parameters should be the same"
    assert np.allclose(peak_in, peak_out), "Peaks should be the same"


def test_remove_peak_lorentz():
    """Test removing single peak as a Lorentzian peak"""
    t = np.arange(0, 3, 0.001)
    param_in = np.array([0.8, 2.5, 0.2])
    peak_in = peak_lorentz(t, param_in[0], param_in[1], param_in[2])

    peak_out, param_out = Preprocessor._remove_peak(t, peak_in, ptype="lorentz")
    assert np.allclose(param_in, param_out), "Parameters should be the same"
    assert np.allclose(peak_in, peak_out), "Peaks should be the same"


def test_remove_peak_unknown_value():
    """Test removing single peak with unknown peak name.
    It should throw ValueError exception."""

    t = np.arange(0, 2, 0.001)
    peak_in = np.random.random(t.size)
    with pytest.raises(ValueError) as context:
        Preprocessor._remove_peak(t, peak_in, ptype="unknown")
        assert "Incorrect ptype value" in str(context.exception)
    with pytest.raises(ValueError) as context:
        Preprocessor._remove_peak(t, peak_in, ptype="other")
        assert "Incorrect ptype value" in str(context.exception)


def test_remove_energy_default():
    t = np.arange(0, 5, 0.001)
    p1 = [1.0, 2.0, 0.4]
    p2 = [3.5, 1.5, 0.3]
    p3 = [2.0, 1.0, 0.2]

    peak1 = peak_norm(t, p1[0], p1[1], p1[2])  # Biggest
    peak2 = peak_lorentz(t, p2[0], p2[1], p2[2])  # Middle
    peak3 = peak_triangle(t, p3[0], p3[1], p3[2])  # Smallest
    s = peak1 + peak2 + peak3

    preprocessor = Preprocessor()
    assert preprocessor.energy_ratio == 0.1, "Default energy ratio is 0.1"
    _, p_out = preprocessor.remove_energy(t, s)

    # Check number of peaks
    assert len(p_out) == 3, "Expected 3 peaks by default"

    # Checking location
    msg = "Expected: {:.3}, Receive: {:.3}"
    assert abs(1.00 - p_out[0, 0]) < 0.01, msg.format(1.00, p_out[0, 0])
    assert abs(3.50 - p_out[1, 0]) < 0.01, msg.format(3.50, p_out[1, 0])
    assert abs(2.00 - p_out[2, 0]) < 0.01, msg.format(2.00, p_out[2, 0])

    # Checking amplitude
    assert abs(1.99 - p_out[0, 1]) < 0.01, msg.format(1.99, p_out[0, 1])
    assert abs(1.33 - p_out[1, 1]) < 0.01, msg.format(1.33, p_out[1, 1])
    assert abs(0.93 - p_out[2, 1]) < 0.01, msg.format(0.93, p_out[2, 1])

    # Checking width
    assert abs(0.40 - p_out[0, 2]) < 0.01, msg.format(0.40, p_out[0, 2])
    assert abs(0.23 - p_out[1, 2]) < 0.01, msg.format(0.23, p_out[1, 2])
    assert abs(0.12 - p_out[2, 2]) < 0.01, msg.format(0.12, p_out[2, 2])


def test_remove_energy_triangle():
    t = np.arange(0, 5, 0.001)
    p1 = [1.0, 2.0, 0.4]
    p2 = [3.5, 1.5, 0.3]
    p3 = [2.0, 1.0, 0.2]

    peak1 = peak_norm(t, p1[0], p1[1], p1[2])  # Biggest
    peak2 = peak_lorentz(t, p2[0], p2[1], p2[2])  # Middle
    peak3 = peak_triangle(t, p3[0], p3[1], p3[2])  # Smallest
    s = peak1 + peak2 + peak3

    preprocessor = Preprocessor()
    assert preprocessor.energy_ratio == 0.1, "Default energy ratio is 0.1"
    _, p_out = preprocessor.remove_energy(t, s, ptype="triang")

    # Check number of peaks
    assert len(p_out) == 3, "Expected 3 peaks by default"

    # Checking location
    msg = "Expected: {:.3}, Receive: {:.3}"
    assert abs(p1[0] - p_out[0, 0]) < 0.01, msg.format(p1[0], p_out[0, 0])
    assert abs(p2[0] - p_out[1, 0]) < 0.01, msg.format(p2[0], p_out[1, 0])
    assert abs(p3[0] - p_out[2, 0]) < 0.01, msg.format(p3[0], p_out[2, 0])

    # Checking amplitude
    assert abs(2.16 - p_out[0, 1]) < 0.02, msg.format(p1[1], p_out[0, 1])
    assert abs(1.43 - p_out[1, 1]) < 0.02, msg.format(p2[1], p_out[1, 1])
    assert abs(1.01 - p_out[2, 1]) < 0.02, msg.format(p3[1], p_out[2, 1])

    # Checking width
    assert abs(0.65 - p_out[0, 2]) < 0.02, msg.format(0.65, p_out[0, 2])
    assert abs(0.37 - p_out[1, 2]) < 0.02, msg.format(0.37, p_out[1, 2])
    assert abs(0.20 - p_out[2, 2]) < 0.02, msg.format(0.20, p_out[2, 2])


def test_remove_energy_max_peaks():
    """Test remove_energy with limited number of peaks"""
    t = np.arange(0, 5, 0.001)
    p1 = [1.0, 2.0, 0.4]
    p2 = [3.5, 1.5, 0.3]
    p3 = [2.0, 1.0, 0.2]

    peak1 = peak_norm(t, p1[0], p1[1], p1[2])  # Biggest
    peak2 = peak_lorentz(t, p2[0], p2[1], p2[2])  # Middle
    peak3 = peak_triangle(t, p3[0], p3[1], p3[2])  # Smallest
    s = peak1 + peak2 + peak3

    max_peaks = 2
    preprocessor = Preprocessor()
    _, p_out = preprocessor.remove_energy(t, s, max_peaks=max_peaks)

    # Check number of peaks
    assert len(p_out) == 2, "Expected 2 peaks because of flag"


def test_remove_energy_different_energy_ratio():
    """Test remove_energy after changing energy_ratio to different value"""
    t = np.arange(0, 5, 0.001)
    p1 = [1.0, 2.0, 0.4]
    p2 = [3.5, 1.5, 0.3]
    p3 = [2.0, 1.0, 0.2]

    peak1 = peak_norm(t, p1[0], p1[1], p1[2])  # Biggest
    peak2 = peak_lorentz(t, p2[0], p2[1], p2[2])  # Middle
    peak3 = peak_triangle(t, p3[0], p3[1], p3[2])  # Smallest
    s = peak1 + peak2 + peak3

    energy_ratio = 0.7
    preprocessor = Preprocessor()
    _, p_out = preprocessor.remove_energy(t, s, energy_ratio=energy_ratio)

    # Check number of peaks
    assert len(p_out) == 1, "One peak expected with large energy ratio"


def test_determine_params_default():
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1)
    S = c1 + c2

    preprocessor = Preprocessor()
    params = preprocessor.determine_params(t, S)
    # Testing for number
    assert params.shape == (2, 4), "Two oscillators with (W, A, s, ph)"
    # Testing for frequency
    assert abs(params[0, 0] - 5) < 0.001, "First oscillator is with 5 Hz"
    assert abs(params[1, 0] - 2) < 0.001, "Second oscillator is with 2 Hz"
    # Testing for phase
    assert abs(params[0, 3] - 1) < 0.001, "First oscillator has phase 1"
    assert abs(params[1, 3] - 0) < 0.001, "Second oscillator has phase 0"


def test_determine_params_energy_ratio():
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1.5)
    c3 = _cos(t, 8, 6, np.pi)  # Highest amplitude
    c4 = _cos(t, 15, 3, 0)
    S = c1 + c2 + c3 + c4

    preprocessor = Preprocessor()

    high_energy = 0.8
    params = preprocessor.determine_params(t, S, energy_ratio=high_energy)
    assert len(params) == 1, "Only one oscillator"
    assert abs(params[0, 0] - 8) < 0.01, "Highest amp for osc with 8 Hz"

    mid_energy = 0.5
    params = preprocessor.determine_params(t, S, energy_ratio=mid_energy)
    assert len(params) == 2, "Two oscillators identified"
    assert abs(params[0, 0] - 15) < 0.01, "First osc with 15 Hz"
    assert abs(params[1, 0] - 8) < 0.01, "Second osc with 8 Hz"

    low_energy = 0.1
    params = preprocessor.determine_params(t, S, energy_ratio=low_energy)
    assert len(params) == 4, "All oscillators identified"
    assert abs(params[0, 0] - 15) < 0.01, "First osc with 15 Hz"
    assert abs(params[1, 0] - 8) < 0.01, "Second osc with 8 Hz"
    assert abs(params[2, 0] - 5) < 0.01, "Third osc with 5 Hz"
    assert abs(params[3, 0] - 2) < 0.01, "Fourth osc with 2 Hz"


def test_determine_params_max_oscillators():
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1.5)
    c3 = _cos(t, 8, 6, np.pi)  # Highest amplitude
    c4 = _cos(t, 15, 3, 0)
    S = c1 + c2 + c3 + c4

    preprocessor = Preprocessor()

    max_peaks_1 = 1
    params = preprocessor.determine_params(t, S, max_peaks=max_peaks_1)
    assert len(params) == max_peaks_1, "Only one oscillator identified"

    max_peaks_2 = 2
    params = preprocessor.determine_params(t, S, max_peaks=max_peaks_2)
    assert len(params) == max_peaks_2, "Two oscillators identified"

    max_peaks_3 = 3
    params = preprocessor.determine_params(t, S, max_peaks=max_peaks_3)
    assert len(params) == max_peaks_3, "Three oscillators identified"


def test_determine_params_ptype():
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1.5)
    c3 = _cos(t, 8, 6, np.pi)  # Highest amplitude
    S = c1 + c2 + c3

    preprocessor = Preprocessor()
    norm_params = preprocessor.determine_params(t, S, energy_ratio=0.20, ptype="norm")
    lorentz_params = preprocessor.determine_params(t, S, energy_ratio=0.20, ptype="lorentz")

    # Expecting similar freqs and amps
    assert norm_params.shape == lorentz_params.shape, "Expecting same number of oscillators"
    assert np.allclose(norm_params[:, 0], lorentz_params[:, 0], atol=1e-1), "Expecting similar frequencies,\n" + str(
        norm_params[:, 0] - lorentz_params[:, 0]
    )
    assert np.allclose(norm_params[:, 1], lorentz_params[:, 1], atol=1e-1), "Expecting similar amplitudes,\n" + str(
        norm_params[:, 1] - lorentz_params[:, 1]
    )


def test_compute_prior_default():
    "Currently almost copy of test_determine_params_*"
    pi2 = np.pi * 2
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1)
    S = c1 + c2

    preprocessor = Preprocessor()
    params = preprocessor.compute_prior(t, S)
    params[:, 1] = (params[:, 1] + pi2) % pi2

    # Testing for number
    assert params.shape == (2, 4), "Two oscillators (W, ph, A, K)"

    # Testing for frequency
    assert abs(params[0, 0] - 5 * pi2) < 0.05, "Expected {} rad/s, Got {} [rad/s]".format(5 * pi2, params[0, 0])
    assert abs(params[1, 0] - 2 * pi2) < 0.05, "Expected {} rad/s, Got {} [rad/s]".format(2 * pi2, params[1, 0])

    # Testing for phase
    assert abs(params[0, 1] - 1) < 0.001, "Expected phase {}, Got {}.".format(1, params[0, 1])
    assert abs(params[1, 1] - 0) < 0.001, "Expected phase {}, Got {}.".format(0, params[1, 1])

    # Testing for amplitude
    assert abs(params[0, 2] - 1.1) < 0.1, "Expected amp {}, Got {}.".format(1.1, params[0, 2])
    assert abs(params[1, 2] - 2) < 0.1, "Expected amp {}, Got {}.".format(2, params[1, 2])

    # Testing for coupling
    assert params[0, 3] == 0, "First->Second coupling should be 0"
    assert params[1, 3] == 0, "Second->First coupling should be 0"


def test_compute_prior_custom_nH():
    "Currently almost copy of test_determine_params_*"
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1)
    S = c1 + c2

    preprocessor = Preprocessor(nH=3)
    params = preprocessor.compute_prior(t, S)
    # Testing for number
    assert params.shape == (2, 6), "Two oscillators (W, ph, A, K1, K2, K3)"
    # Testing for coupling
    assert np.all(params[:, 3:] == 0), "All couplings should be zero"


def test_compute_prior_energy_ratio():
    "Currently almost copy of test_determine_params_*"
    pi2 = np.pi * 2
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1.5)
    c3 = _cos(t, 8, 6, np.pi)  # Highest amplitude
    c4 = _cos(t, 15, 3, 0)
    S = c1 + c2 + c3 + c4

    mid_energy = 0.5
    preprocessor = Preprocessor(energy_ratio=mid_energy)
    params = preprocessor.compute_prior(t, S)
    assert len(params) == 2, "Two oscillators identified"
    assert abs(params[0, 0] - 15 * pi2) < 0.05, "Expected {} rad/s, Got {} [rad/s]: ".format(
        15 * pi2, params[0, 0] * pi2
    )
    assert abs(params[1, 0] - 8 * pi2) < 0.05, "Expected {} rad/s, Got {} [rad/s]: ".format(8 * pi2, params[1, 0] * pi2)

    low_energy = 0.1
    preprocessor = Preprocessor(energy_ratio=low_energy)
    params = preprocessor.compute_prior(t, S)
    assert len(params) == 4, "All oscillators identified"
    assert abs(params[0, 0] - 15 * pi2) < 0.05, "First osc with 15 Hz"
    assert abs(params[1, 0] - 8 * pi2) < 0.05, "Second osc with 8 Hz"
    assert abs(params[2, 0] - 5 * pi2) < 0.05, "Third osc with 5 Hz"
    assert abs(params[3, 0] - 2 * pi2) < 0.05, "Fourth osc with 2 Hz"


def test_compute_prior_max_oscillators():
    "Currently almost copy of test_determine_params_*"
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1.5)
    c3 = _cos(t, 8, 6, np.pi)  # Highest amplitude
    c4 = _cos(t, 15, 3, 0)
    S = c1 + c2 + c3 + c4

    max_osc_2 = 2
    preprocessor = Preprocessor(max_osc=max_osc_2)
    params = preprocessor.compute_prior(t, S)
    assert len(params) == max_osc_2, "Two oscillators identified"

    max_osc_3 = 3
    preprocessor = Preprocessor(max_osc=max_osc_3)
    params = preprocessor.compute_prior(t, S)
    assert len(params) == max_osc_3, "Three oscillators identified"


def test_compute_prior_1_oscillator():
    "Currently almost copy of test_determine_params_*"
    t = np.arange(0, 5, 0.001)
    c1 = _cos(t, 2, 2, 0)
    c2 = _cos(t, 5, 1.1, 1.5)
    S = c1 + c2

    max_osc = 1
    preprocessor = Preprocessor(max_osc=max_osc)
    with pytest.raises(ValueError) as context:
        preprocessor.compute_prior(t, S)

    assert "Single oscillator detected" in str(context)
