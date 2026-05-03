"""
Experiment 05: Exact F_θθ invariances under two noise channels.

Numerical observation (this code): for the rotated-twin-Fock probe at the
operating point (θ_op, φ_op, Θ) = (π/2, π/2, π/2), the reflectivity QFI

    F_θθ  =  N(N+2)/2

is preserved EXACTLY (to floating-point precision) under either of:

    (a) Jz-dephasing of arbitrary strength γ ∈ [0, 1].
        ρ_ij in Jz basis  →  ρ_ij · exp(-γ · (m_i - m_j)²)

    (b) One-arm photon loss with η_a ∈ [0.1, 1] and η_b = 1.
        Mode-b photon counts preserved; mode-a photons lost with probability 1-η_a.

In both cases F_φφ degrades normally (collapses to ~0 for large γ; degrades
proportionally to η_a·η_b under loss).

These exact invariances are striking. We have not yet found a published
theorem that derives them, but the structural reason is clear:
both noise channels preserve the mode-b photon-number stabilizer, and the
rotated twin Fock + Jx generator + this stabilizer combine to give a
decoherence-free subspace for the θ-parameter.

Symbolic proof: open. The numerical evidence is unambiguous.
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (angular_momentum, twin_fock, bs, phase_shift,
                 sld_qfi, _sld_qfi_pair, lossy_density_matrix,
                 lossy_block_dim, block_offsets, lossy_qfi_matrix)
from src.core import comb


def dephased_state(N, psi_in, gamma, theta, Theta, phi0):
    Jx, _, Jz = angular_momentum(N)
    psi_after_bs1 = bs(theta, Jx) @ psi_in
    rho = np.outer(psi_after_bs1, psi_after_bs1.conj())
    j = N / 2
    m = np.arange(-j, j + 1)
    decoh = np.exp(-gamma * (m[:, None] - m[None, :]) ** 2)
    rho = rho * decoh
    U2 = bs(Theta, Jx) @ phase_shift(phi0, Jz)
    return U2 @ rho @ U2.conj().T


def dephased_qfi(N, psi_in, gamma, theta, Theta, phi0, h=1e-4):
    rho_th_p = dephased_state(N, psi_in, gamma, theta + h, Theta, phi0)
    rho_th_m = dephased_state(N, psi_in, gamma, theta - h, Theta, phi0)
    rho_ph_p = dephased_state(N, psi_in, gamma, theta, Theta, phi0 + h)
    rho_ph_m = dephased_state(N, psi_in, gamma, theta, Theta, phi0 - h)
    rho0 = dephased_state(N, psi_in, gamma, theta, Theta, phi0)
    d_th = (rho_th_p - rho_th_m) / (2 * h)
    d_ph = (rho_ph_p - rho_ph_m) / (2 * h)
    F = np.array([[sld_qfi(rho0, d_th), _sld_qfi_pair(rho0, d_th, d_ph)],
                  [_sld_qfi_pair(rho0, d_th, d_ph), sld_qfi(rho0, d_ph)]])
    return F


def kraus_loss_asymm(N, eta_a, eta_b):
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
                       * (1 - eta_a) ** (l_a / 2) * eta_a ** ((m0 - l_a) / 2)
                       * (1 - eta_b) ** (l_b / 2) * eta_b ** ((m1 - l_b) / 2))
                K[off[Np] + np_in_0, n] = amp
            if np.any(K != 0):
                Ks.append(K)
    return Ks


def lossy_state_asymm(N, psi_in, eta_a, eta_b, theta):
    Jx, _, _ = angular_momentum(N)
    psi_after_bs1 = bs(theta, Jx) @ psi_in
    Ks = kraus_loss_asymm(N, eta_a, eta_b)
    big_dim = lossy_block_dim(N)
    rho = np.zeros((big_dim, big_dim), dtype=complex)
    for K in Ks:
        v = K @ psi_after_bs1
        rho += np.outer(v, v.conj())
    return rho


def lossy_density_asymm(N, psi_in, eta_a, eta_b, theta, Theta, phi0):
    from src.core import apply_phase_and_bs2_blockwise
    rho = lossy_state_asymm(N, psi_in, eta_a, eta_b, theta)
    return apply_phase_and_bs2_blockwise(rho, N, Theta, phi0)


def lossy_qfi_asymm(N, psi_in, eta_a, eta_b, theta, Theta, phi0, h=1e-4):
    rho_th_p = lossy_density_asymm(N, psi_in, eta_a, eta_b, theta + h, Theta, phi0)
    rho_th_m = lossy_density_asymm(N, psi_in, eta_a, eta_b, theta - h, Theta, phi0)
    rho_ph_p = lossy_density_asymm(N, psi_in, eta_a, eta_b, theta, Theta, phi0 + h)
    rho_ph_m = lossy_density_asymm(N, psi_in, eta_a, eta_b, theta, Theta, phi0 - h)
    rho0 = lossy_density_asymm(N, psi_in, eta_a, eta_b, theta, Theta, phi0)
    d_th = (rho_th_p - rho_th_m) / (2 * h)
    d_ph = (rho_ph_p - rho_ph_m) / (2 * h)
    F = np.array([[sld_qfi(rho0, d_th), _sld_qfi_pair(rho0, d_th, d_ph)],
                  [_sld_qfi_pair(rho0, d_th, d_ph), sld_qfi(rho0, d_ph)]])
    return F


def main():
    print("=" * 80)
    print("Part A: F_θθ invariance under Jz-dephasing")
    print("=" * 80)
    print(f"\n{'N':>4}{'γ':>10}{'F_θθ':>10}{'F_φφ':>10}{'F_θθ/N(N+2)/2':>18}")
    dephasing_rows = []
    for N in [4, 8, 14]:
        Jx, _, _ = angular_momentum(N)
        chi = bs(np.pi / 2, Jx) @ twin_fock(N)
        psi_in = bs(-np.pi / 2, Jx) @ chi
        F_max = N * (N + 2) / 2
        for gamma in [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 1.0]:
            F = dephased_qfi(N, psi_in, gamma,
                             np.pi / 2 + 0.005, np.pi / 2, np.pi / 2 + 0.005)
            ratio = F[0, 0] / F_max
            dephasing_rows.append({
                "N": N, "gamma": gamma,
                "F_theta_theta": float(F[0, 0]),
                "F_phi_phi": float(F[1, 1]),
                "F_theta_theta_ratio": float(ratio),
            })
            print(f"{N:>4}{gamma:>10.4f}{F[0,0]:>10.4f}{F[1,1]:>10.4f}{ratio:>18.6f}")
        print()

    print("=" * 80)
    print("Part B: F_θθ invariance under one-arm loss (η_b = 1)")
    print("=" * 80)
    print(f"\n{'N':>4}{'η_a':>8}{'F_θθ(η_a, 1)':>16}{'F_θθ(1, 1)':>14}{'ratio':>12}")
    asymm_rows = []
    for N in [4, 6, 8, 12, 16]:
        Jx, _, _ = angular_momentum(N)
        chi = bs(np.pi / 2, Jx) @ twin_fock(N)
        psi_in = bs(-np.pi / 2, Jx) @ chi
        F_at_1 = lossy_qfi_asymm(N, psi_in, 1.0, 1.0,
                                  np.pi / 2, np.pi / 2, np.pi / 2)
        f1 = F_at_1[0, 0]
        for eta_a in [1.0, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1]:
            F = lossy_qfi_asymm(N, psi_in, eta_a, 1.0,
                                np.pi / 2, np.pi / 2, np.pi / 2)
            ratio = F[0, 0] / f1
            asymm_rows.append({
                "N": N, "eta_a": eta_a, "eta_b": 1.0,
                "F_theta_theta_loss": float(F[0, 0]),
                "F_theta_theta_lossless": float(f1),
                "ratio": float(ratio),
            })
            print(f"{N:>4}{eta_a:>8.2f}{F[0,0]:>16.6f}{f1:>14.4f}{ratio:>12.6f}")
        print()

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "05_invariances.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "description": __doc__,
            "dephasing": dephasing_rows,
            "asymmetric_loss": asymm_rows,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
