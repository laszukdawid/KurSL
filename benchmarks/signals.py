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


def _known_signal(
    name: str,
    oscillator_count: int,
    harmonic_order: int,
    sample_count: int,
    duration: float,
    seed: int,
    target_iterations: int = 100,
) -> KnownMCMCSignal:
    time = np.linspace(0.0, duration, sample_count)
    frequencies = 2 * np.pi * (1.5 + 1.25 * np.arange(oscillator_count))
    phases = 0.1 + np.linspace(0.0, 1.8, oscillator_count)
    amplitudes = 0.75 + np.linspace(0.0, 1.5, oscillator_count)
    coupling = np.zeros((oscillator_count, harmonic_order * (oscillator_count - 1)))
    theta = np.column_stack((frequencies, phases, amplitudes, coupling))
    theta_std = np.full(theta.shape, 1e-9)
    components = amplitudes[:, None] * np.cos(frequencies[:, None] * time[:-1] + phases[:, None])

    return KnownMCMCSignal(
        name=name,
        time=time,
        theta=theta,
        theta_init=theta.copy(),
        theta_std=theta_std,
        values=np.sum(components, axis=0),
        nwalkers=2 * theta.size,
        measured_iterations=1,
        target_iterations=target_iterations,
        seed=seed,
        frequency_tolerance=1e-6,
        phase_tolerance=1e-6,
        amplitude_tolerance=1e-6,
        coupling_tolerance=1e-6,
        max_relative_error=1e-10,
    )


_CASE_SPECS = (
    (2, 1, 64, 0.5),
    (2, 1, 96, 1.0),
    (2, 2, 128, 1.5),
    (2, 3, 160, 2.0),
    (3, 1, 72, 0.5),
    (3, 1, 112, 1.0),
    (3, 2, 144, 1.5),
    (3, 2, 192, 2.0),
    (3, 3, 224, 2.5),
    (4, 1, 80, 0.5),
    (4, 1, 120, 1.0),
    (4, 2, 168, 1.5),
    (4, 2, 216, 2.0),
    (4, 3, 256, 2.5),
    (5, 1, 96, 0.75),
    (5, 2, 144, 1.25),
    (5, 2, 200, 1.75),
    (5, 3, 272, 2.25),
    (6, 2, 320, 2.5),
)

KNOWN_MCMC_SIGNALS = tuple(
    _known_signal(
        name=f"known-{index:02d}-o{oscillators}-h{harmonics}-s{samples}",
        oscillator_count=oscillators,
        harmonic_order=harmonics,
        sample_count=samples,
        duration=duration,
        seed=41 + index,
    )
    for index, (oscillators, harmonics, samples, duration) in enumerate(_CASE_SPECS, start=1)
) + (
    _known_signal(
        name="known-20-epilepsy-scale-o4-h2-s4097",
        oscillator_count=4,
        harmonic_order=2,
        sample_count=4_097,
        duration=4_097 / 173.61,
        seed=61,
    ),
)
