import pytest
import jax.numpy as jnp
from jax.test_util import check_grads
import s2fft
from s2wav.transforms import rec_wav_jax, pre_wav_jax, construct
from s2wav import filters, samples

L_to_test = [8]
N_to_test = [3]
J_min_to_test = [2]
multiresolution = [False, True]
reality = [False, True]
sampling_to_test = ["mw", "mwss", "dh"]
recursive_transform = [False, True]

@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("multiresolution", multiresolution)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
def test_jax_synthesis_gradients(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    multiresolution: bool,
    reality: bool,
    recursive: bool
):
    J = samples.j_max(L)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    # Generate wavelet filters
    filter = filters.filters_directional_vectorised(L, N, J_min)
    generator = construct.generate_wigner_precomputes if recursive else construct.generate_full_precomputes
    synthesis = rec_wav_jax.synthesis if recursive else pre_wav_jax.synthesis
    precomps = generator(
        L,
        N,
        J_min,
        forward=True,
        reality=reality,
        multiresolution=multiresolution,
    )

    # Generate random signal
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.inverse_jax(flm, L)
    f_wav, f_scal = rec_wav_jax.analysis(
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
        f = synthesis(
            f_wav,
            f_scal,
            L,
            N,
            J_min,
            multiresolution=multiresolution,
            reality=reality,
            filters=filter,
            precomps=precomps,
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
@pytest.mark.parametrize("recursive", recursive_transform)
def test_jax_analysis_gradients(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    multiresolution: bool,
    reality: bool,
    recursive: bool
):
    J = samples.j_max(L)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    # Generate wavelet filters
    filter = filters.filters_directional_vectorised(L, N, J_min)
    generator = construct.generate_wigner_precomputes if recursive else construct.generate_full_precomputes
    analysis = rec_wav_jax.analysis if recursive else pre_wav_jax.analysis
    precomps = generator(
        L,
        N,
        J_min,
        forward=False,
        reality=reality,
        multiresolution=multiresolution,
    )

    # Generate random signal
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.inverse_jax(flm, L)

    # Generate target signal
    flm_target = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f_target = s2fft.inverse_jax(flm_target, L)
    f_wav_target, f_scal_target = rec_wav_jax.analysis(
        f_target,
        L,
        N,
        J_min,
        multiresolution=multiresolution,
        reality=reality,
        filters=filter,
    )

    def func(f):
        f_wav, f_scal = analysis(
            f,
            L,
            N,
            J_min,
            multiresolution=multiresolution,
            reality=reality,
            filters=filter,
            precomps=precomps
        )
        loss = jnp.sum(jnp.abs(f_scal - f_scal_target) ** 2)
        for j in range(J - J_min):
            loss += jnp.sum(jnp.abs(f_wav[j - J_min] - f_wav_target[j - J_min]) ** 2)
        return loss

    check_grads(func, (f,), order=1, modes=("rev"))
