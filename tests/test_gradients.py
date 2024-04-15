import jax

jax.config.update("jax_enable_x64", True)
import pytest
import jax.numpy as jnp
from jax.test_util import check_grads
import s2fft
from s2wav.transforms import wavelet, wavelet_c, wavelet_precompute, construct
from s2wav import filters, samples

L_to_test = [8]
N_to_test = [3]
J_min_to_test = [2]
reality = [False, True]
recursive_transform = [False, True]
using_c_backend = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("using_c_backend", using_c_backend)
def test_jax_synthesis_gradients(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    reality: bool,
    recursive: bool,
    using_c_backend: bool,
):
    J = samples.j_max(L)

    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")
    if not recursive and using_c_backend:
        pytest.skip("Precompute transform not supported from C backend libraries.")
    if reality and using_c_backend:
        pytest.skip("Hermitian symmetry for C backend gradients currently conflicts.")

    # Generate random signal
    f_wav, f_scal, _, _ = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=2, reality=reality
    )

    # Generate wavelet filters
    filter = filters.filters_directional_vectorised(L, N, J_min, 2)
    generator = (
        None
        if using_c_backend
        else (
            construct.generate_wigner_precomputes
            if recursive
            else construct.generate_full_precomputes
        )
    )
    synthesis = (
        (wavelet_c.synthesis if using_c_backend else wavelet.synthesis)
        if recursive
        else wavelet_precompute.synthesis
    )

    precomps = (
        None
        if using_c_backend
        else generator(L, N, J_min, 2, forward=True, reality=reality)
    )

    args = {"precomps": precomps} if not using_c_backend else {}

    def func(f_wav, f_scal):
        f = synthesis(
            f_wav,
            f_scal,
            L,
            N,
            J_min,
            reality=reality,
            filters=filter,
            **args,
        )
        return jnp.sum(jnp.abs(f) ** 2)

    check_grads(func, (f_wav, f_scal), order=1, modes=("rev"))


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("using_c_backend", using_c_backend)
def test_jax_analysis_gradients(
    flm_generator,
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    reality: bool,
    recursive: bool,
    using_c_backend: bool,
):
    J = samples.j_max(L)
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")
    if not recursive and using_c_backend:
        pytest.skip("Precompute transform not supported from C backend libraries.")
    if reality and using_c_backend:
        pytest.skip("Hermitian symmetry for C backend gradients currently conflicts.")

    # Generate random signal
    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = s2fft.inverse_jax(flm, L)

    # Generate target signal
    f_wav_target, f_scal_target, _, _ = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=2, reality=reality
    )

    # Generate wavelet filters
    filter = filters.filters_directional_vectorised(L, N, J_min)
    generator = (
        None
        if using_c_backend
        else (
            construct.generate_wigner_precomputes
            if recursive
            else construct.generate_full_precomputes
        )
    )
    analysis = (
        (wavelet_c.analysis if using_c_backend else wavelet.analysis)
        if recursive
        else wavelet_precompute.analysis
    )
    precomps = (
        None
        if using_c_backend
        else generator(L, N, J_min, forward=False, reality=reality)
    )

    args = {"precomps": precomps} if not using_c_backend else {}

    def func(f):
        f_wav, f_scal = analysis(
            f, L, N, J_min, reality=reality, filters=filter, **args
        )
        loss = jnp.sum(jnp.abs(f_scal - f_scal_target) ** 2)
        for j in range(J - J_min):
            loss += jnp.sum(jnp.abs(f_wav[j - J_min] - f_wav_target[j - J_min]) ** 2)
        return loss

    check_grads(func, (f.real if reality else f,), order=1, modes=("rev"))
