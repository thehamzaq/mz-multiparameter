"""
SU(2)-Schwinger toolkit for two-mode interferometric quantum metrology.

Conventions:
- N photons live in the (N+1)-dim symmetric subspace, j = N/2.
- |j, m⟩ basis indexed i = 0..N where m = -j + i, so i = (photons in mode 0).
- Beam splitter: B(θ) = exp(-i θ Jx).  50:50 ↔ θ = π/2.
- Phase shift in arm 0 vs arm 1: P(φ) = exp(-i φ Jz).
- Mach-Zehnder with first BS reflectivity θ, phase φ, controllable BS Θ:
    U_MZ(θ, Θ, φ) = B(Θ) P(φ) B(θ).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.special import comb


def angular_momentum(N: int):
    j = N / 2.0
    m = np.arange(-j, j + 1)
    Jz = np.diag(m).astype(complex)
    Jp = np.zeros((N + 1, N + 1), dtype=complex)
    for i in range(N):
        mi = m[i]
        Jp[i + 1, i] = np.sqrt(j * (j + 1) - mi * (mi + 1))
    Jm = Jp.conj().T
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2j)
    return Jx, Jy, Jz


def sine_state(N):
    """Berry-Wiseman / Wiseman-Killip optimal phase probe."""
    n = np.arange(N + 1)
    psi = np.sqrt(2 / (N + 2)) * np.sin((n + 1) * np.pi / (N + 2))
    return psi.astype(complex)


def noon_state(N):
    psi = np.zeros(N + 1, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return psi


def twin_fock(N):
    """|N/2, N/2⟩ — Holland-Burnett state. Requires N even."""
    if N % 2:
        raise ValueError("twin_fock requires even N")
    psi = np.zeros(N + 1, dtype=complex)
    psi[N // 2] = 1.0
    return psi


def coherent_spin(N, theta=np.pi / 2, phi=0.0):
    j = N / 2
    m = np.arange(-j, j + 1)
    c = np.zeros(N + 1, dtype=complex)
    for i, mi in enumerate(m):
        binom = comb(N, j + mi, exact=False)
        c[i] = np.sqrt(binom) * np.cos(theta / 2) ** (j + mi) \
            * np.sin(theta / 2) ** (j - mi) * np.exp(-1j * (j + mi) * phi)
    return c / np.linalg.norm(c)


def bs(theta, Jx):
    return expm(-1j * theta * Jx)


def phase_shift(phi, Jz):
    return expm(-1j * phi * Jz)


def mzi(theta, Theta, phi0, Jx, Jz):
    return bs(Theta, Jx) @ phase_shift(phi0, Jz) @ bs(theta, Jx)


def expval(op, psi):
    return np.real(np.conj(psi) @ op @ psi)


def variance(op, psi):
    return expval(op @ op, psi) - expval(op, psi) ** 2


def qfi_pure(state, generator):
    return 4 * variance(generator, state)


def qfi_reflectivity_analytic(N, chi):
    """χ = state between BS1 and phase imprint. F(θ) = 4 Var_χ(Jx).
    Holds because Jx commutes with B(θ) so the variance is θ-independent."""
    Jx, _, _ = angular_momentum(N)
    return qfi_pure(chi, Jx)


def qfi_phase_analytic(N, chi):
    """F(φ) = 4 Var_χ(Jz)."""
    _, _, Jz = angular_momentum(N)
    return qfi_pure(chi, Jz)


def qfi_two_param_inbetween(N, chi):
    """2x2 SLD QFI matrix for (θ, φ) on in-between state χ. Off-diag from {Jx,Jz}."""
    Jx, _, Jz = angular_momentum(N)
    Fxx = 4 * variance(Jx, chi)
    Fzz = 4 * variance(Jz, chi)
    anticom = Jx @ Jz + Jz @ Jx
    Fxz = 2 * np.real(np.conj(chi) @ anticom @ chi - 2 * expval(Jx, chi) * expval(Jz, chi))
    return np.array([[Fxx, Fxz], [Fxz, Fzz]])


def compatibility_inbetween(N, chi):
    """Pure-state QCRB saturability: |Im⟨Δθ ψ | Δφ ψ⟩| = |⟨Jy⟩_χ|.
    Zero ⟹ joint estimation saturates the SLD bound (Matsumoto's condition)."""
    _, Jy, _ = angular_momentum(N)
    return abs(expval(Jy, chi))


def qfi_two_param(N, psi_in, theta, Theta, phi0, h=1e-5):
    """2x2 SLD QFI matrix for (θ, φ) at fixed Θ. Indexing: row/col 0=θ, 1=φ."""
    Jx, _, Jz = angular_momentum(N)
    psi0 = mzi(theta, Theta, phi0, Jx, Jz) @ psi_in
    d_th = (mzi(theta + h, Theta, phi0, Jx, Jz) - mzi(theta - h, Theta, phi0, Jx, Jz)) @ psi_in / (2 * h)
    d_ph = (mzi(theta, Theta, phi0 + h, Jx, Jz) - mzi(theta, Theta, phi0 - h, Jx, Jz)) @ psi_in / (2 * h)

    def F(a, b):
        return 4 * np.real(np.conj(a) @ b - (np.conj(a) @ psi0) * (np.conj(psi0) @ b))

    return np.array([[F(d_th, d_th), F(d_th, d_ph)],
                     [F(d_ph, d_th), F(d_ph, d_ph)]])


def compatibility(N, psi_in, theta, Theta, phi0, h=1e-5):
    """Pure-state saturability: |Im ⟨∂_i ψ | ∂_j ψ⟩|. Zero ⇒ QCRB saturable."""
    Jx, _, Jz = angular_momentum(N)
    psi0 = mzi(theta, Theta, phi0, Jx, Jz) @ psi_in
    d_th = (mzi(theta + h, Theta, phi0, Jx, Jz) - mzi(theta - h, Theta, phi0, Jx, Jz)) @ psi_in / (2 * h)
    d_ph = (mzi(theta, Theta, phi0 + h, Jx, Jz) - mzi(theta, Theta, phi0 - h, Jx, Jz)) @ psi_in / (2 * h)
    P = np.eye(N + 1) - np.outer(psi0, psi0.conj())
    return float(np.imag(np.conj(d_th) @ P @ d_ph))


def output_probs(theta, Theta, phi0, psi_in, Jx, Jz):
    return np.abs(mzi(theta, Theta, phi0, Jx, Jz) @ psi_in) ** 2


def cfi_reflectivity(theta, Theta, phi0, psi_in, h=1e-5):
    Jx, _, Jz = angular_momentum(len(psi_in) - 1)
    p_plus = output_probs(theta + h, Theta, phi0, psi_in, Jx, Jz)
    p_minus = output_probs(theta - h, Theta, phi0, psi_in, Jx, Jz)
    p0 = output_probs(theta, Theta, phi0, psi_in, Jx, Jz)
    dp = (p_plus - p_minus) / (2 * h)
    mask = p0 > 1e-14
    return float(np.sum(dp[mask] ** 2 / p0[mask]))


def cfi_phase(theta, Theta, phi0, psi_in, h=1e-5):
    Jx, _, Jz = angular_momentum(len(psi_in) - 1)
    p_plus = output_probs(theta, Theta, phi0 + h, psi_in, Jx, Jz)
    p_minus = output_probs(theta, Theta, phi0 - h, psi_in, Jx, Jz)
    p0 = output_probs(theta, Theta, phi0, psi_in, Jx, Jz)
    dp = (p_plus - p_minus) / (2 * h)
    mask = p0 > 1e-14
    return float(np.sum(dp[mask] ** 2 / p0[mask]))


def lossy_block_dim(N):
    """Total dim of ⊕_{N'=0}^{N} sym_{N'} = (N+1)(N+2)/2."""
    return (N + 1) * (N + 2) // 2


def block_offsets(N):
    """Index offsets: block N' starts at offset[N']. block N' has dim N'+1."""
    off = [0]
    for Np in range(N + 1):
        off.append(off[-1] + (Np + 1))
    return off


def kraus_loss(N, eta):
    """Kraus operators for symmetric photon loss on two modes, mapping from
    N-photon symmetric subspace (dim N+1, basis |n, N-n⟩, index = n)
    to ⊕_{N'≤N} sym_{N'} (full Fock-truncated space).
    Returns one big Kraus matrix K[(N'+1)(N'+2)/2 + n', n] for each (l_a, l_b).
    """
    big_dim = lossy_block_dim(N)
    off = block_offsets(N)
    Ks = []
    for l_a in range(N + 1):
        for l_b in range(N + 1 - l_a):
            Np = N - l_a - l_b
            K = np.zeros((big_dim, N + 1), dtype=complex)
            for n in range(N + 1):
                m0, m1 = n, N - n
                if m0 < l_a or m1 < l_b:
                    continue
                np_in_0 = m0 - l_a
                np_in_1 = m1 - l_b
                if np_in_0 + np_in_1 != Np:
                    continue
                amp = (np.sqrt(comb(m0, l_a, exact=False) * comb(m1, l_b, exact=False))
                       * (1 - eta) ** ((l_a + l_b) / 2) * eta ** ((m0 - l_a + m1 - l_b) / 2))
                K[off[Np] + np_in_0, n] = amp
            if np.any(K != 0):
                Ks.append(K)
    return Ks


def lossy_state_pre_phase(N, psi_in, eta, theta):
    """ρ after BS1 with reflectivity θ then symmetric photon loss with η.
    Returned in the big block-diagonal basis."""
    Jx, _, _ = angular_momentum(N)
    psi_after_bs1 = bs(theta, Jx) @ psi_in
    Ks = kraus_loss(N, eta)
    big_dim = lossy_block_dim(N)
    rho = np.zeros((big_dim, big_dim), dtype=complex)
    for K in Ks:
        v = K @ psi_after_bs1
        rho += np.outer(v, v.conj())
    return rho


def apply_phase_and_bs2_blockwise(rho, N, Theta, phi0):
    """Apply P(φ) then B(Θ) within each photon-number block separately."""
    off = block_offsets(N)
    out = rho.copy()
    for Np in range(N + 1):
        if Np == 0:
            continue
        Jx_p, _, Jz_p = angular_momentum(Np)
        U = bs(Theta, Jx_p) @ phase_shift(phi0, Jz_p)
        s, e = off[Np], off[Np + 1]
        out[s:e, :] = U @ out[s:e, :]
        out[:, s:e] = out[:, s:e] @ U.conj().T
    return out


def lossy_density_matrix(N, psi_in, eta, theta, Theta, phi0):
    rho = lossy_state_pre_phase(N, psi_in, eta, theta)
    return apply_phase_and_bs2_blockwise(rho, N, Theta, phi0)


def _sld_qfi_pair(rho, drho_a, drho_b, rel_eps=1e-7):
    """SLD QFI inner product Tr[L_a · drho_b] = sum 2 Re(M_a M_b†)/(w_i+w_k).
    Filters eigenvalue pairs with small w_i+w_k relative to max(w)."""
    w, V = np.linalg.eigh(rho)
    w = np.real(w)
    wmax = max(w.max(), 1e-30)
    Ma = V.conj().T @ drho_a @ V
    Mb = V.conj().T @ drho_b @ V
    sums = w[:, None] + w[None, :]
    keep = sums > rel_eps * wmax
    contrib = np.where(keep, 2 * np.real(Ma * Mb.conj()) / np.where(keep, sums, 1.0), 0.0)
    return float(np.sum(contrib))


def sld_qfi(rho, drho, rel_eps=1e-7):
    return _sld_qfi_pair(rho, drho, drho, rel_eps=rel_eps)


def block_diagonal_J(N):
    """Block-diagonal Jx, Jy, Jz on ⊕_{N'=0}^{N} sym_{N'}. Each block uses
    angular_momentum(N') in its own (N'+1)-dim subspace."""
    big = lossy_block_dim(N)
    off = block_offsets(N)
    Jx_big = np.zeros((big, big), dtype=complex)
    Jy_big = np.zeros((big, big), dtype=complex)
    Jz_big = np.zeros((big, big), dtype=complex)
    for Np in range(N + 1):
        if Np == 0:
            continue
        Jx, Jy, Jz = angular_momentum(Np)
        s, e = off[Np], off[Np + 1]
        Jx_big[s:e, s:e] = Jx
        Jy_big[s:e, s:e] = Jy
        Jz_big[s:e, s:e] = Jz
    return Jx_big, Jy_big, Jz_big


def mom_fisher_matrix(rho_func, params, observables, h=1e-4, eps=1e-12):
    """Method-of-moments classical Fisher matrix.
    rho_func: callable rho_func(*params) -> rho (DxD complex Hermitian).
    params: tuple of current parameter values (θ_1, ..., θ_K).
    observables: list of K_obs Hermitian DxD matrices.
    Returns K x K Fisher matrix.
    """
    rho0 = rho_func(*params)
    K = len(params)
    M = len(observables)
    means = np.array([float(np.real(np.trace(O @ rho0))) for O in observables])
    cov = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            anti = (observables[i] @ observables[j] + observables[j] @ observables[i]) / 2
            cov[i, j] = float(np.real(np.trace(anti @ rho0))) - means[i] * means[j]
    cov_inv = np.linalg.pinv(cov, rcond=eps)
    grad = np.zeros((M, K), dtype=float)
    for k in range(K):
        p_plus = list(params); p_plus[k] += h
        p_minus = list(params); p_minus[k] -= h
        rho_p = rho_func(*p_plus)
        rho_m = rho_func(*p_minus)
        for i, O in enumerate(observables):
            mp = float(np.real(np.trace(O @ rho_p)))
            mm = float(np.real(np.trace(O @ rho_m)))
            grad[i, k] = (mp - mm) / (2 * h)
    F = grad.T @ cov_inv @ grad
    return F


def lossy_qfi_matrix(N, psi_in, eta, theta, Theta, phi0, h=1e-4):
    """2x2 SLD QFI matrix for (θ, φ) under symmetric photon loss."""
    rho_th_p = lossy_density_matrix(N, psi_in, eta, theta + h, Theta, phi0)
    rho_th_m = lossy_density_matrix(N, psi_in, eta, theta - h, Theta, phi0)
    rho_ph_p = lossy_density_matrix(N, psi_in, eta, theta, Theta, phi0 + h)
    rho_ph_m = lossy_density_matrix(N, psi_in, eta, theta, Theta, phi0 - h)
    rho0 = lossy_density_matrix(N, psi_in, eta, theta, Theta, phi0)
    d_th = (rho_th_p - rho_th_m) / (2 * h)
    d_ph = (rho_ph_p - rho_ph_m) / (2 * h)
    F_thth = sld_qfi(rho0, d_th)
    F_phph = sld_qfi(rho0, d_ph)
    F_thph = _sld_qfi_pair(rho0, d_th, d_ph)
    return np.array([[F_thth, F_thph], [F_thph, F_phph]])
