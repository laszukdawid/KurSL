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
    niter: int
    seed: int


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
        name="two-oscillators-2x100",
        time=time,
        theta=theta,
        theta_init=theta_init,
        theta_std=theta_std,
        values=np.sum(components, axis=0),
        nwalkers=20,
        niter=20,
        seed=42,
    )


KNOWN_MCMC_SIGNALS = (_two_oscillator_signal(),)
