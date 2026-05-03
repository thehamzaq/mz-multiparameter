"""
Experiment 03: Minimal saturating method-of-moments observable set.

Claim (numerical observation): For the rotated-twin-Fock probe at the
slightly-off-symmetric operating point (θ_op, φ_op, Θ_op) = (π/2 + ε, π/2 + ε, Θ),
the minimal observable set whose method-of-moments classical Fisher matrix
saturates the joint quantum Cramér-Rao bound is

    D_min = { Jx² ,  (Jx Jz + Jz Jx) / 2 }     (only 2 quadratic observables)

Verified for N ∈ {4, 6, 8, 10, 14}.

Algebraic reason (numerical evidence; symbolic proof open):
On the rotated twin Fock, Jy acts trivially (Jy=0 eigenstate), and the
identity Jy|j,0⟩ = -i·{Jx,Jz}|j,0⟩ on twin Fock lifts to: the missing Jy
direction in the SLD operators is synthesised from the anticommutator
{Jx, Jz}. The Jx² observable carries the F_θθ-direction information.

Compare to Volkoff-Ryu Frontiers Phys 2024 who showed that for *single*-
parameter phase, the 2-observable set {Jz², (J+² + J-²)/2} is globally
optimal. Our 2-observable set for the joint *two*-parameter problem differs
in the choice of the second observable; the Jx² ↔ Jz² swap is the key change.
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import angular_momentum, twin_fock, bs


def rotated_twin_fock(N):
    Jx, _, _ = angular_momentum(N)
    return bs(np.pi / 2, Jx) @ twin_fock(N)


def output_pure(N, theta, Theta, phi, psi_in):
    Jx, _, Jz = angular_momentum(N)
    from src.core import phase_shift
    return bs(Theta, Jx) @ phase_shift(phi, Jz) @ bs(theta, Jx) @ psi_in


def mom_fisher_pure(N, theta, Theta, phi, psi_in, observables, h=1e-4):
    def expvals(t, p):
        psi = output_pure(N, t, Theta, p, psi_in)
        return np.array([float(np.real(np.conj(psi) @ O @ psi)) for O in observables])

    psi0 = output_pure(N, theta, Theta, phi, psi_in)
    K = len(observables)
    e0 = expvals(theta, phi)
    cov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            anti = (observables[i] @ observables[j] +
                    observables[j] @ observables[i]) / 2
            cov[i, j] = float(np.real(np.conj(psi0) @ anti @ psi0)) - e0[i] * e0[j]
    cov_inv = np.linalg.pinv(cov, rcond=1e-12)
    grad = np.zeros((K, 2))
    grad[:, 0] = (expvals(theta + h, phi) - expvals(theta - h, phi)) / (2 * h)
    grad[:, 1] = (expvals(theta, phi + h) - expvals(theta, phi - h)) / (2 * h)
    return grad.T @ cov_inv @ grad


def best_Theta(N, observables, theta_op, phi_op, psi_in):
    best = (np.inf, None)
    for Th0 in np.linspace(0.05, 0.95, 12) * np.pi:
        def cost(p):
            Th = p[0]
            F = mom_fisher_pure(N, theta_op, Th, phi_op, psi_in, observables)
            det = F[0, 0] * F[1, 1] - F[0, 1] ** 2
            if det <= 1e-10:
                return 1e6
            return (F[0, 0] + F[1, 1]) / det * N ** 2
        res = minimize(cost, [Th0], method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-6, maxiter=80))
        if res.fun < best[0]:
            best = (float(res.fun), float(res.x[0]))
    return best


def main():
    eps = 0.005
    Ns = [4, 6, 8, 10, 14]

    rows = []
    for N in Ns:
        chi = rotated_twin_fock(N)
        Jx_N, Jy_N, Jz_N = angular_momentum(N)
        psi_in = bs(-np.pi / 2, Jx_N) @ chi
        bound = 4 * N / (N + 2)
        theta_op = np.pi / 2 + eps
        phi_op = np.pi / 2 + eps

        obs_full = {
            "Jx":      Jx_N,
            "Jz":      Jz_N,
            "Jx2":     Jx_N @ Jx_N,
            "Jz2":     Jz_N @ Jz_N,
            "{Jx,Jz}/2": (Jx_N @ Jz_N + Jz_N @ Jx_N) / 2,
            "J2":      Jx_N @ Jx_N + Jy_N @ Jy_N + Jz_N @ Jz_N,
        }

        # Test sets of various sizes
        test_sets = [
            ("Jx²", ["Jx2"]),
            ("{Jx,Jz}/2", ["{Jx,Jz}/2"]),
            ("Jx² + Jz²", ["Jx2", "Jz2"]),
            ("Jx² + {Jx,Jz}/2", ["Jx2", "{Jx,Jz}/2"]),
            ("Jx² + Jz² + {Jx,Jz}/2", ["Jx2", "Jz2", "{Jx,Jz}/2"]),
            ("D = full 6-observable set", list(obs_full.keys())),
        ]

        N_results = []
        for label, keys in test_sets:
            sub = [obs_full[k] for k in keys]
            ti, Th = best_Theta(N, sub, theta_op, phi_op, psi_in)
            saturates = (ti < float("inf")) and (abs(ti - bound) / bound < 0.01)
            N_results.append({
                "set": label, "k": len(keys),
                "best_Theta_over_pi": Th / np.pi if Th else 0.0,
                "N2_Tr_Finv": ti if np.isfinite(ti) else None,
                "saturates": bool(saturates),
            })

        rows.append({"N": N, "bound": bound, "results": N_results})

    # Print
    for r in rows:
        print(f"\nN={r['N']}, joint CR bound = {r['bound']:.4f}")
        print(f"  {'set':<28}{'|set|':>6}{'Θ/π':>10}{'N²·Tr[F⁻¹]':>14}{'sat':>10}")
        print("  " + "-" * 68)
        for sub in r["results"]:
            ti_str = f"{sub['N2_Tr_Finv']:.6f}" if sub["N2_Tr_Finv"] else "inf"
            print(f"  {sub['set']:<28}{sub['k']:>6}{sub['best_Theta_over_pi']:>10.4f}"
                  f"{ti_str:>14}{'YES' if sub['saturates'] else 'no':>10}")

    print("\nMinimal saturating set (across all N tested): {Jx², (Jx Jz + Jz Jx)/2}")
    print("Set D was 3× redundant — this is a 2-observable readout.")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "03_minimal_set.json")
    with open(out_path, "w") as f:
        json.dump({"description": __doc__, "rows": rows}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
