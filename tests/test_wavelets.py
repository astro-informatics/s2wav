import pytest
import numpy as np
import torch
import pys2let as s2let
from s2fft import base_transforms as sht_base
from s2wav.transforms import (
    wavelet,
    wavelet_precompute,
    wavelet_precompute_torch,
    construct,
)
from s2wav import filters, samples

L_to_test = [8]
N_to_test = [2, 3]
J_min_to_test = [2]
lam_to_test = [2, 3]
reality = [False, True]
sampling_to_test = ["mw", "mwss", "dh", "gl"]
recursive_transform = [False, True]
using_torch_frontend = [False, True]
using_c_backend = [False, True]
_ssht_backends = [0, 1]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("using_torch", using_torch_frontend)
@pytest.mark.parametrize("using_c_backend", using_c_backend)
@pytest.mark.parametrize("_ssht_backend", _ssht_backends)
def test_synthesis(
    wavelet_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    reality: bool,
    recursive: bool,
    using_torch: bool,
    using_c_backend: bool,
    _ssht_backend: int,
):
    J = samples.j_max(L, lam)

    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")
    if recursive and using_torch:
        pytest.skip("Recursive transform not yet available for torch frontend.")
    if not recursive and using_c_backend:
        pytest.skip("Precompute transform not supported from C backend libraries.")

    f_wav, f_scal, f_wav_s2let, f_scal_s2let = wavelet_generator(
        L=L, N=N, J_min=J_min, lam=lam, reality=reality, using_torch=using_torch
    )

    f = s2let.synthesis_wav2px(
        f_wav_s2let,
        f_scal_s2let,
        lam,
        L,
        J_min,
        N,
        spin=0,
        upsample=False,
    )

    filter = filters.filters_directional_vectorised(
        L, N, J_min, lam, using_torch=using_torch
    )
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
        wavelet.synthesis
        if recursive
        else (
            wavelet_precompute_torch.synthesis
            if using_torch
            else wavelet_precompute.synthesis
        )
    )
    precomps = (
        None
        if using_c_backend
        else generator(
            L, N, J_min, lam, forward=True, reality=reality, using_torch=using_torch
        )
    )
    args = (
        {"use_c_backend": using_c_backend, "_ssht_backend": _ssht_backend}
        if using_c_backend
        else {}
    )

    f_check = synthesis(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        reality=reality,
        filters=filter,
        precomps=precomps,
        **args,
    )

    if using_torch:
        f_check = f_check.resolve_conj().numpy()

    f = np.real(f) if reality else f
    np.testing.assert_allclose(f, f_check.flatten("C"), atol=1e-14)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("using_torch", using_torch_frontend)
@pytest.mark.parametrize("using_c_backend", using_c_backend)
@pytest.mark.parametrize("_ssht_backend", _ssht_backends)
def test_analysis(
    flm_generator,
    f_wav_converter,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    reality: bool,
    recursive: bool,
    using_torch: bool,
    using_c_backend: bool,
    _ssht_backend: int,
):
    J = samples.j_max(L, lam)

    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")
    if recursive and using_torch:
        pytest.skip("Recursive transform not yet available for torch frontend")
    if not recursive and using_c_backend:
        pytest.skip("Precompute transform not supported from C backend libraries.")

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = sht_base.spherical.inverse(flm, L, reality=reality)

    f_wav, f_scal = s2let.analysis_px2wav(
        f.flatten("C").astype(np.complex128), lam, L, J_min, N, spin=0, upsample=False
    )
    filter = filters.filters_directional_vectorised(
        L, N, J_min, lam, using_torch=using_torch
    )
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
        wavelet.analysis
        if recursive
        else (
            wavelet_precompute_torch.analysis
            if using_torch
            else wavelet_precompute.analysis
        )
    )
    precomps = (
        None
        if using_c_backend
        else generator(
            L, N, J_min, lam, forward=False, reality=reality, using_torch=using_torch
        )
    )

    args = (
        {"use_c_backend": using_c_backend, "_ssht_backend": _ssht_backend}
        if using_c_backend
        else {}
    )

    f_wav_check, f_scal_check = analysis(
        torch.from_numpy(f) if using_torch else f,
        L,
        N,
        J_min,
        lam,
        reality=reality,
        filters=filter,
        precomps=precomps,
        **args,
    )

    f_wav_check = f_wav_converter(f_wav_check, L, N, J_min, lam, using_torch)

    np.testing.assert_allclose(f_wav, f_wav_check, atol=1e-14)
    np.testing.assert_allclose(
        f_scal,
        f_scal_check.resolve_conj().numpy().flatten("C")
        if using_torch
        else f_scal_check.flatten("C"),
        atol=1e-14,
    )


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("lam", lam_to_test)
@pytest.mark.parametrize("reality", reality)
@pytest.mark.parametrize("sampling", sampling_to_test)
@pytest.mark.parametrize("using_c_backend", using_c_backend)
@pytest.mark.parametrize("_ssht_backend", _ssht_backends)
def test_round_trip(
    flm_generator,
    L: int,
    N: int,
    J_min: int,
    lam: int,
    reality: bool,
    sampling: str,
    using_c_backend: bool,
    _ssht_backend: int,
):
    J = samples.j_max(L, lam)

    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = flm_generator(L=L, L_lower=0, spin=0, reality=reality)
    f = sht_base.spherical.inverse(flm, L, reality=reality, sampling=sampling)
    filter = filters.filters_directional_vectorised(L, N, J_min, lam)

    args = (
        {"use_c_backend": using_c_backend, "_ssht_backend": _ssht_backend}
        if using_c_backend
        else {}
    )

    f_wav, f_scal = wavelet.analysis(
        f,
        L,
        N,
        J_min,
        lam,
        reality=reality,
        sampling=sampling,
        filters=filter,
        **args,
    )
    f_check = wavelet.synthesis(
        f_wav,
        f_scal,
        L,
        N,
        J_min,
        lam,
        sampling=sampling,
        reality=reality,
        filters=filter,
        **args,
    )

    np.testing.assert_allclose(f, f_check, atol=1e-14)
