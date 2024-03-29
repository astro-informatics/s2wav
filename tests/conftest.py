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
    sampling: str = "mw",
    reality: bool = False,
    multiresolution: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from s2wav.utils import shapes
    from s2fft import base_transforms as base

    J = shapes.j_max(L, lam)
    flmn = shapes.construct_flmn(L, N, J_min, lam, multiresolution)

    f_wav = []
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = shapes.LN_j(L, j, N, lam, multiresolution)

        for n in range(-Nj + 1, Nj, 2):
            for el in range(max(abs(n), L0j), Lj):
                for m in range(-el, el + 1):
                    flmn[j - J_min][Nj - 1 + n, el, Lj - 1 + m] = (
                        rng.uniform() + 1j * rng.uniform()
                    )
        f_wav.append(base.wigner.inverse(flmn[j - J_min], Lj, Nj, 0, sampling, reality))

    L_s = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = np.zeros((L_s, 2 * L_s - 1), dtype=np.complex128)
    for el in range(L_s):
        for m in range(-el, el + 1):
            flm[el, L_s - 1 + m] = rng.uniform() + 1j * rng.uniform()

    f_scal = base.spherical.inverse(flm, L_s, 0, sampling, reality)

    return (
        f_wav,
        f_scal,
        s2wav_to_s2let(f_wav, L, N, J_min, lam, multiresolution),
        f_scal.flatten("C"),
    )


def s2wav_to_s2let(
    f_wav: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> int:
    from s2wav.utils.shapes import j_max

    J = j_max(L, lam)
    f_wav_s2let = np.zeros(
        n_wav(L, N, J_min, lam, multiresolution), dtype=np.complex128
    )
    offset = 0
    for j in range(J_min, J + 1):
        entries = f_wav[j - J_min].flatten("C")
        f_wav_s2let[offset : offset + len(entries)] = entries
        offset += len(entries)
    return f_wav_s2let


def n_wav(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    multiresolution: bool = False,
    sampling: str = "mw",
) -> int:
    from s2wav.utils import shapes
    from s2fft.sampling import so3_samples

    J = shapes.j_max(L, lam)
    count = 0
    for j in range(J_min, J + 1):
        Lj = shapes.wav_j_bandlimit(L, j, lam, multiresolution)
        count += np.prod(list(so3_samples.f_shape(Lj, N, sampling)))

    return count


@pytest.fixture
def rng(seed):
    # Import numpy locally to avoid `RuntimeWarning: numpy.ndarray size changed`
    # when importing at module level
    import numpy as np

    return np.random.default_rng(seed)


@pytest.fixture
def f_wav_converter():
    return s2wav_to_s2let


@pytest.fixture
def wavelet_generator(rng):
    return partial(generate_f_wav_scal, rng)


@pytest.fixture
def flm_generator(rng):
    # Import s2fft (and indirectly numpy) locally to avoid
    # `RuntimeWarning: numpy.ndarray size changed` when importing at module level
    import s2fft

    return partial(s2fft.utils.signal_generator.generate_flm, rng)
