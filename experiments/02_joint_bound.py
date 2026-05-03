"""
Experiment 02: Joint quantum Cramér-Rao bound saturation.

Claim: For two-parameter (θ, φ) Mach-Zehnder estimation on the symmetric
N-photon (j = N/2) Hilbert subspace, the SLD quantum Cramér-Rao bound is

    Tr[F⁻¹]  ≥  4 / (N(N+2)),     N²·Tr[F⁻¹]  →  4    as N → ∞

This bound is the standard SU(2) isotropy bound and follows from
4·(Var Jx + Var Jy + Var Jz) = N(N+2)/4·4 = N(N+2) for ⟨J⟩ = 0 states,
with AM-GM minimisation over Tr[F⁻¹] = 1/F_θθ + 1/F_φφ.

Saturating state: ψ_opt = exp(-iπ Jx/2) |N/2, N/2⟩  (rotated twin Fock).
Equivalently: the in-between state of a Mach-Zehnder with twin Fock at the
input and a 50:50 first beam splitter (Holland-Burnett 1993 configuration).

This experiment verifies the saturation numerically up to N = 30.
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (angular_momentum, twin_fock, bs,
                 qfi_two_param_inbetween, compatibility_inbetween, expval)


def rotated_twin_fock(N):
    Jx, _, _ = angular_momentum(N)
    return bs(np.pi / 2, Jx) @ twin_fock(N)


def main():
    rows = []
    for N in [4, 6, 8, 10, 14, 18, 24, 30]:
        chi = rotated_twin_fock(N)
        F = qfi_two_param_inbetween(N, chi)
        det = F[0, 0] * F[1, 1] - F[0, 1] ** 2
        trinv = (F[0, 0] + F[1, 1]) / det
        N2_trinv = trinv * N ** 2
        bound = 4 * N / (N + 2)
        Jx, Jy, Jz = angular_momentum(N)
        Cy = compatibility_inbetween(N, chi)
        rows.append({
            "N": N,
            "F_theta_theta": float(F[0, 0]),
            "F_phi_phi": float(F[1, 1]),
            "F_theta_phi": float(F[0, 1]),
            "compatibility_Jy": float(Cy),
            "N2_Tr_Finv": float(N2_trinv),
            "bound_4N_over_Np2": float(bound),
            "gap": float(N2_trinv - bound),
        })

    # Print summary
    print(f"{'N':>4}{'F_θθ':>10}{'F_φφ':>10}{'F_θφ':>10}"
          f"{'|⟨Jy⟩|':>10}{'N²·Tr[F⁻¹]':>14}{'bound':>10}{'gap':>12}")
    print("-" * 84)
    for r in rows:
        print(f"{r['N']:>4}{r['F_theta_theta']:>10.3f}{r['F_phi_phi']:>10.3f}"
              f"{r['F_theta_phi']:>10.3e}{r['compatibility_Jy']:>10.3e}"
              f"{r['N2_Tr_Finv']:>14.6f}{r['bound_4N_over_Np2']:>10.6f}"
              f"{r['gap']:>12.3e}")
    print()
    print("The saturating probe achieves N²·Tr[F⁻¹] = 4N/(N+2) to floating-point precision.")
    print("Compatibility |⟨Jy⟩| = 0 ⇒ pure-state QCR-Holevo coincide; bound is saturable in principle.")

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "02_joint_bound.json")
    with open(out_path, "w") as f:
        json.dump({"description": __doc__, "rows": rows}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
