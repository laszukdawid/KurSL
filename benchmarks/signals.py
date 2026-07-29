from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KnownMCMCSignal:
    name: str
    time: np.ndarray
    theta: np.ndarray
    theta_init: np.ndarray
    theta_std: np.ndarray
    values: np.ndarray
    nwalkers: int
    measured_iterations: int
    target_iterations: int
    seed: int
    frequency_tolerance: float
    phase_tolerance: float
    amplitude_tolerance: float
    coupling_tolerance: float
    max_relative_error: float


def _two_oscillator_signal() -> KnownMCMCSignal:
    time = np.linspace(0.0, 1.0, 100)
    theta = np.array(
        [
            [15.0, 0.2, 1.0, 0.0],
            [35.0, 2.0, 3.0, 0.0],
        ]
    )
    theta_init = theta + np.array(
        [
            [0.8, 0.1, 0.4, 0.0],
            [0.8, 0.1, 0.4, 0.0],
        ]
    )
    theta_std = np.array(
        [
            [1.0, 0.1, 0.5, 0.01],
            [1.0, 0.1, 0.5, 0.01],
        ]
    )
    components = theta[:, 2, None] * np.cos(theta[:, 0, None] * time[:-1] + theta[:, 1, None])

    return KnownMCMCSignal(
        name="smoke-2x100",
        time=time,
        theta=theta,
        theta_init=theta_init,
        theta_std=theta_std,
        values=np.sum(components, axis=0),
        nwalkers=20,
        measured_iterations=20,
        target_iterations=20,
        seed=42,
        frequency_tolerance=0.6,
        phase_tolerance=0.3,
        amplitude_tolerance=0.35,
        coupling_tolerance=0.03,
        max_relative_error=0.05,
    )


def _epilepsy_scale_signal() -> KnownMCMCSignal:
    sample_count = 4_097
    time = np.linspace(0.0, sample_count / 173.61, sample_count)
    frequencies = 2 * np.pi * np.array([2.0, 3.5, 5.0, 7.0])
    phases = np.array([0.2, 0.8, 1.4, 2.0])
    amplitudes = np.array([1.0, 1.5, 2.0, 2.5])
    coupling = np.zeros((4, 6))
    theta = np.column_stack((frequencies, phases, amplitudes, coupling))
    theta_std = np.column_stack(
        (
            np.full(4, 1e-4),
            np.full(4, 1e-3),
            np.full(4, 1e-2),
            np.full((4, 6), 1e-3),
        )
    )
    components = amplitudes[:, None] * np.cos(frequencies[:, None] * time[:-1] + phases[:, None])

    return KnownMCMCSignal(
        name="epilepsy-scale-4x4097",
        time=time,
        theta=theta,
        theta_init=theta.copy(),
        theta_std=theta_std,
        values=np.sum(components, axis=0),
        nwalkers=72,
        measured_iterations=1,
        target_iterations=100,
        seed=42,
        frequency_tolerance=1e-3,
        phase_tolerance=0.01,
        amplitude_tolerance=0.03,
        coupling_tolerance=0.01,
        max_relative_error=0.01,
    )


KNOWN_MCMC_SIGNALS = (_two_oscillator_signal(), _epilepsy_scale_signal())
