"""Collection of shared fixtures"""
from functools import partial
from typing import Tuple
import numpy as np
import pytest

DEFAULT_SEED = 8966433580120847635


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help=(
            "Seed(s) to use for random number generator fixture rng in tests. If "
            "multiple seeds are passed tests depending on rng will be run for all "
            "seeds specified."
        ),
    )


def pytest_generate_tests(metafunc):
    if "seed" in metafunc.fixturenames:
        metafunc.parametrize("seed", metafunc.config.getoption("seed"))


def generate_f_wav_scal(
    rng: np.random.Generator,
    L: int,
    N: int,
    J_min: int,
    lam: float,
    spin: int = 0,
    sampling: str = "mw",
) -> Tuple[np.ndarray, np.ndarray]:
    from s2wav import shapes

    f_wav_shape = shapes.f_wav(L, N, J_min, lam)
    f_scal_shape = shapes.f_scal(L)

    f_wav = rng.uniform(size=f_wav_shape) + 1j * rng.uniform(size=f_wav_shape)
    f_scal = rng.uniform(size=f_scal_shape) + 1j * rng.uniform(size=f_scal_shape)

    return f_wav, f_scal


@pytest.fixture
def rng(seed):
    # Import numpy locally to avoid `RuntimeWarning: numpy.ndarray size changed`
    # when importing at module level
    import numpy as np

    return np.random.default_rng(seed)


@pytest.fixture
def wavelet_generator(rng):
    return partial(generate_f_wav_scal, rng)
