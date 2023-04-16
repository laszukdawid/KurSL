import numpy as np

from kursl import Preprocessor
from utils import peak_norm, peak_triangle, peak_lorentz, cosine


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


def test_determine_params_energy_ratio():
    t = np.arange(0, 5, 0.001)
    c1 = cosine(t, 2, 2, 0)
    c2 = cosine(t, 5, 1.1, 1.5)
    c3 = cosine(t, 8, 6, np.pi)  # Highest amplitude
    c4 = cosine(t, 15, 3, 0)
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


def test_compute_prior_energy_ratio():
    "Currently almost copy of test_determine_params_*"
    pi2 = np.pi * 2
    t = np.arange(0, 5, 0.001)
    c1 = cosine(t, 2, 2, 0)
    c2 = cosine(t, 5, 1.1, 1.5)
    c3 = cosine(t, 8, 6, np.pi)  # Highest amplitude
    c4 = cosine(t, 15, 3, 0)
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
