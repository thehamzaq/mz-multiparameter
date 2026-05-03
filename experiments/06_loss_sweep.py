"""
Experiment 06: Joint quantum Cramér-Rao bound under symmetric photon loss.

For η ∈ [0.4, 1.0] and N ∈ [4, 20], compute the joint SLD-Cramér-Rao bound
N²·Tr[F⁻¹] under symmetric photon loss for three probes:

  - rotated twin Fock     (joint-CR-saturating at η=1)
  - sine state (BW)       (Berry-Wiseman optimal for phase-only)
  - NOON                  (Heisenberg phase, SQL reflectivity)

Findings:
  - rot Twin Fock dominates throughout η ∈ [0.4, 1.0]
  - NOON catastrophically fails under loss for joint estimation
    (F_φφ → 0 quickly; Tr[F⁻¹] blows up by 4–6 orders of magnitude)
  - sine is more loss-robust than rot-TF in *relative* terms but starts so
    much worse it's dominated for all η ≥ 0.5.
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import (angular_momentum, twin_fock, sine_state, noon_state, bs,
                 lossy_qfi_matrix)


def rotated_twin_fock(N):
    Jx, _, _ = angular_momentum(N)
    return bs(np.pi / 2, Jx) @ twin_fock(N)


def joint_imprecision_x_N2(F, N):
    det = F[0, 0] * F[1, 1] - F[0, 1] ** 2
    if det <= 1e-10:
        return float("inf")
    return float((F[0, 0] + F[1, 1]) / det) * N ** 2


def main():
    theta_op = np.pi / 2 + 0.005
    phi_op = np.pi / 2 + 0.005
    Theta_op = np.pi / 2

    probes = [
        ("rot Twin Fock", rotated_twin_fock),
        ("sine (BW)",     sine_state),
        ("NOON",          noon_state),
    ]

    rows = []
    for N in [4, 8, 12, 16, 20]:
        if N % 2:
            continue
        Jx, _, _ = angular_momentum(N)
        for eta in [1.0, 0.95, 0.9, 0.8, 0.6, 0.4]:
            for name, builder in probes:
                chi = builder(N)
                psi_in = bs(-theta_op, Jx) @ chi  # FIXED input
                F = lossy_qfi_matrix(N, psi_in, eta, theta_op, Theta_op, phi_op)
                ti = joint_imprecision_x_N2(F, N)
                bound = 4 * N / (N + 2)
                rows.append({
                    "N": N, "eta": eta, "probe": name,
                    "F_theta_theta": float(F[0, 0]),
                    "F_phi_phi": float(F[1, 1]),
                    "F_theta_phi": float(F[0, 1]),
                    "N2_Tr_Finv": ti if ti < float("inf") else None,
                    "ratio_to_bound": ti / bound if ti < float("inf") else None,
                })

    print(f"{'N':>4}{'η':>6}{'probe':<18}{'F_θθ':>12}{'F_φφ':>12}{'N²·Tr[F⁻¹]':>14}{'×bound':>10}")
    print("-" * 76)
    for r in rows:
        ti_str = f"{r['N2_Tr_Finv']:.4f}" if r['N2_Tr_Finv'] is not None else "  inf  "
        rb_str = f"{r['ratio_to_bound']:.2f}" if r['ratio_to_bound'] is not None else "inf"
        print(f"{r['N']:>4}{r['eta']:>6.2f}{r['probe']:<18}{r['F_theta_theta']:>12.4f}"
              f"{r['F_phi_phi']:>12.4f}{ti_str:>14}{rb_str:>10}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "06_loss_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"description": __doc__, "rows": rows}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
