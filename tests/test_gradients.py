import pytest
import numpy as np

from s2wav.transforms import jax_wavelets
from s2wav.filter_factory import filters
from s2wav.utils import shapes
import jax.numpy as jnp
from jax.test_util import check_grads
import s2fft

L_to_test = [8]
N_to_test = [3]
J_min_to_test = [2]
multiresolution = [False, True]
reality = [False, True]
sampling_to_test = ["mw", "mwss", "dh"]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
def test_jax_synthesis_gradients(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    multiresolution: bool,
    reality: bool,
):
    J = shapes.j_max(L)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    # Generate wavelet filters
    filter = filters.filters_directional_vectorised(L, N, J_min)

    # Generate random signal
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.inverse_jax(flm, L)
    f_wav, f_scal = jax_wavelets.analysis(
        f,
        L,
        N,
        J_min,
        multiresolution=multiresolution,
        reality=reality,
        filters=filter,
    )

    # Generate target signal
    flm_target = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f_target = s2fft.inverse_jax(flm_target, L)

    def func(f_wav, f_scal):
        f = jax_wavelets.synthesis(
            f_wav,
            f_scal,
            L,
            N,
            J_min,
            multiresolution=multiresolution,
            reality=reality,
            filters=filter,
        )
        return jnp.sum(jnp.abs(f - f_target) ** 2)

    check_grads(
        func,
        (
            f_wav,
            f_scal,
        ),
        order=1,
        modes=("rev"),
    )


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
def test_jax_analysis_gradients(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    multiresolution: bool,
    reality: bool,
):
    J = shapes.j_max(L)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    # Generate wavelet filters
    filter = filters.filters_directional_vectorised(L, N, J_min)

    # Generate random signal
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.inverse_jax(flm, L)

    # Generate target signal
    flm_target = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f_target = s2fft.inverse_jax(flm_target, L)
    f_wav_target, f_scal_target = jax_wavelets.analysis(
        f_target,
        L,
        N,
        J_min,
        multiresolution=multiresolution,
        reality=reality,
        filters=filter,
    )

    def func(f):
        f_wav, f_scal = jax_wavelets.analysis(
            f,
            L,
            N,
            J_min,
            multiresolution=multiresolution,
            reality=reality,
            filters=filter,
        )
        loss = jnp.sum(jnp.abs(f_scal - f_scal_target) ** 2)
        for j in range(J - J_min):
            loss += jnp.sum(jnp.abs(f_wav[j - J_min] - f_wav_target[j - J_min]) ** 2)
        return loss

    check_grads(func, (f,), order=1, modes=("rev"))
