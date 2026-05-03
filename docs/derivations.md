# Derivations and structural arguments

This document collects the algebraic identities and structural arguments
behind the numerical observations in this repository. **Symbolic proofs are
provided where complete; numerical evidence is flagged when complete proofs
are not yet available.**

## 1. Setup

The two-mode Mach-Zehnder interferometer is parameterised by

| Symbol | Meaning |
|---|---|
| θ ∈ [0, π] | first beam-splitter reflectivity (the parameter we want to estimate) |
| φ ∈ [0, 2π) | inter-arm phase shift (the second parameter) |
| Θ ∈ [0, π] | controllable second beam-splitter reflectivity (a control, not estimated) |

In the SU(2) Schwinger picture on the symmetric *N*-photon subspace
(j = N/2), the unitary chain is

```
ψ_in  ──B(θ)──▶  χ  ──P(φ)──▶  ψ̃  ──B(Θ)──▶  ψ_out  ──[ photon counting | parity | …]──▶ outcomes
```

with `B(θ) = exp(−iθ Jx)` and `P(φ) = exp(−iφ Jz)`. The "in-between
state" χ is `B(θ) ψ_in`. Quantum Fisher information matrix (QFIM)
elements for the parameter pair (θ, φ), using the Symmetric Logarithmic
Derivative definition, can be written on χ as

| | |
|---|---|
| F_θθ | = 4 Var_χ(Jx) |
| F_φφ | = 4 Var_χ(Jz) |
| F_θφ | = 2 (⟨{Jx,Jz}⟩ − 2⟨Jx⟩⟨Jz⟩) |

These follow from QFI invariance under the post-imprint unitary B(Θ)
(parameter-independent) plus the chain rule (e.g., Liu et al., *J. Phys. A*
**53**, 023001, 2020).

## 2. The joint quantum Cramér-Rao bound

**Claim.** For any pure state on the j = N/2 symmetric subspace satisfying
⟨J⟩ = 0 (a necessary condition for compatibility) and Var(Jy) = 0,

```
Tr[F⁻¹] ≥ 4 / (N(N+2)),     N²·Tr[F⁻¹] → 4 as N → ∞.
```

**Proof.** The Casimir `J² = Jx² + Jy² + Jz²` has eigenvalue j(j+1) on the
symmetric subspace. For a state with ⟨Jα⟩ = 0 for all α,

```
Var(Jx) + Var(Jy) + Var(Jz) = ⟨J²⟩ = j(j+1) = N(N+2)/4.
```

Multiplying by 4: `4(Var Jx + Var Jy + Var Jz) = N(N+2)`. With Var(Jy) = 0
(Matsumoto compatibility for joint estimation; see §3), this becomes
`F_θθ + F_φφ = N(N+2)`. Minimising `Tr[F⁻¹] = 1/F_θθ + 1/F_φφ` subject to
the sum constraint gives, by AM-GM,

```
F_θθ = F_φφ = N(N+2)/2,    Tr[F⁻¹]_min = 4/(N(N+2)).   ∎
```

This is the standard SU(2) isotropy bound; it appears in Liu-Yuan-Lu-Wang
2020 review §4.3 and earlier references.

## 3. Saturating state: the rotated twin Fock

**Claim.** The bound is saturated by

```
ψ_opt = exp(−iπ/2 · Jx) |N/2, N/2⟩,
```

i.e. the twin-Fock |j=N/2, m=0⟩ rotated 90° about Jx. (In Mach-Zehnder
language: the in-between state of an MZI with twin Fock at the input and a
50:50 first beam splitter — the *Holland-Burnett 1993 configuration*.)

**Verification.** On twin Fock |j, 0⟩ for integer j:

| | | |
|---|---|---|
| Jx \|tF⟩ | = | (c/2)(\|j, 1⟩ + \|j, −1⟩) where c = √(j(j+1)) |
| Jy \|tF⟩ | = | (c/(2i))(\|j, 1⟩ − \|j, −1⟩) |
| Jz \|tF⟩ | = | 0 |

So Var(Jx)|tF⟩ = ⟨Jx²⟩ = c²/4 + c²/4 = j(j+1)/2 = N(N+2)/8 (similarly Jy).
After rotation by exp(−iπ/2 Jx), the operator Jz → −Jy, and so:

| | |
|---|---|
| ⟨Jα⟩_ψ_opt | = 0 (parity around Jx) |
| Var(Jx)_ψ_opt | = Var(Jx)\|tF⟩ = N(N+2)/8 (Jx commutes with the rotation) |
| Var(Jy)_ψ_opt | = Var(Jz)_\|tF⟩ = 0 (twin Fock is Jz=0 eigenstate) |
| Var(Jz)_ψ_opt | = Var(Jy)\|tF⟩ = N(N+2)/8 |

Hence F_θθ = F_φφ = N(N+2)/2, F_θφ = 0, ⟨Jy⟩ = 0 (compatibility), and
Var(Jy) = 0 (saturating AM-GM). ∎

The saturation also appears in arXiv:2412.19119 (Cassemiro et al.,
Jan 2025) for two-of-three SU(2) parameters via method-of-moments.

## 4. The algebraic identity Jy|j,0⟩ = −i·{Jx, Jz}|j,0⟩

**Claim.** On twin Fock |j, 0⟩ for integer j:

```
Jy |j, 0⟩  =  −i · (Jx Jz + Jz Jx) |j, 0⟩  =  −i · {Jx, Jz} |j, 0⟩.
```

**Proof.**

```
Jx |j, 0⟩ = (c/2)(|j, 1⟩ + |j, −1⟩)
Jz |j, 0⟩ = 0
Jz Jx |j, 0⟩ = (c/2)(1·|j, 1⟩ + (−1)·|j, −1⟩) = (c/2)(|j, 1⟩ − |j, −1⟩)
Jx Jz |j, 0⟩ = Jx · 0 = 0
{Jx, Jz} |j, 0⟩ = Jx Jz |j, 0⟩ + Jz Jx |j, 0⟩ = (c/2)(|j, 1⟩ − |j, −1⟩)
                                              = i · (c/(2i))(|j, 1⟩ − |j, −1⟩)
                                              = i · Jy |j, 0⟩   ⇒   Jy = −i {Jx, Jz}  on |tF⟩. ∎
```

This identity is verified numerically (to floating-point precision) in
`tests/test_smoke.py::test_jy_anticommutator_identity` for
N ∈ {4, 6, 8, 10, 12}. The above derivation lines are the symbolic argument.

## 5. Minimal saturating method-of-moments readout (numerical observation)

**Numerical observation (this repository).** For the rotated twin Fock probe
at the slightly-off-symmetric operating point (θ_op, φ_op, Θ_op) =
(π/2 + ε, π/2 + ε, Θ_opt), the minimal observable set whose method-of-moments
classical Fisher matrix saturates the joint quantum CR bound is

```
D_min = { Jx² ,  (Jx Jz + Jz Jx) / 2 }     (only 2 quadratic observables)
```

Verified numerically for N ∈ {4, 6, 8, 10, 14}.

**Structural reason** (numerical evidence; full symbolic proof open).
A pure-state method-of-moments readout with observables {O_k} saturates the
joint quantum Cramér-Rao bound iff for each parameter generator G_p:

```
G_p |ψ⟩  ∈  span{ (O_k − ⟨O_k⟩) |ψ⟩ : k ∈ {O_k} }.
```

For the rotated twin Fock, the parameter generators (in the appropriate
frame after the post-imprint unitary B(Θ) is absorbed) are linear in
{Jx, Jy, Jz}. The Jx and Jz directions are spanned trivially; the Jy
direction is the non-trivial one. By the identity in §4, after accounting
for the e^(−iπ/2 Jx) rotation, the Jy-direction action on |ψ⟩ can be
expressed via the {Jx, Jz} anticommutator (with imaginary coefficients).

Hence Jx² (carrying F_θθ-direction information) plus (Jx Jz + Jz Jx)/2
(synthesising the missing Jy direction) is sufficient. **Symbolic proof at
general N is open** but the numerical evidence to N = 14 is exact to
floating-point precision.

This parallels the result of Volkoff & Ryu (Frontiers Phys 2024) who
showed that for the *single-parameter* phase problem, the 2-observable set
{Jz², (J+² + J−²)/2} is *globally* optimal. Our 2-observable set for the
two-parameter problem differs in its second observable; the swap

```
single-parameter:  Jz², (J+² + J−²)/2  ⟼  joint two-parameter:  Jx², (Jx Jz + Jz Jx)/2
```

reflects the reorientation needed to capture the second parameter.

## 6. Holevo Cramér-Rao bound under loss (numerical observation)

For pure states with compatibility ⟨Jy⟩ = 0, the SLD Cramér-Rao bound
coincides with the Holevo Cramér-Rao bound (HCRB). Under photon loss the
state becomes mixed and the equivalence is not automatic.

**Numerical observation (this repository).** For the rotated twin Fock
probe under symmetric photon loss with η ∈ [0.5, 1.0] and the symmetric
operating point, HCRB and SLD bound coincide to ≤ 0.001% (the SCS solver
tolerance floor) for N ∈ {4, 6, 8}.

**Implication.** No multi-parameter incompatibility gap opens under loss
for this probe at this operating point. The SLD bound is the relevant
fundamental quantum limit.

**Symbolic proof for general N: open.** Computational scaling
(O((N+1)⁴(N+2)⁴) for dense SDP) prevents direct verification beyond N = 8.

## 7. Exact F_θθ invariances (numerical observation)

**Numerical observation (this repository).** For the rotated twin Fock
probe at the symmetric operating point (π/2, π/2, π/2):

(a) Under Jz-dephasing of arbitrary strength γ ∈ [0, 1]:

```
F_θθ(γ)  =  N(N+2)/2     EXACTLY (to floating-point precision)
F_φφ(γ)  →  0            as γ → ∞
```

Verified for N ∈ {4, 8, 14}.

(b) Under one-arm photon loss with η_a ∈ [0.1, 1.0] and η_b = 1:

```
F_θθ(η_a, 1)  =  F_θθ(1, 1)  =  N(N+2)/2     EXACTLY
F_φφ(η_a, 1)  ≈  η_a · F_φφ(1, 1)            (degrades normally)
```

Verified for N ∈ {4, 6, 8, 12, 16}.

**Conjectured structural reason.** Both noise channels preserve the
mode-b stabilizer `n_b · I` (b-mode photon number). The rotated twin
Fock + the Jx-generator + this stabilizer combine such that the noise
acts trivially on the F_θθ-relevant subspace — a partial decoherence-
free subspace structure for the reflectivity parameter only.

**Symbolic proof: open.** This conjecture was identified by AI numerical
exploration; we are not aware of a published derivation. **Independent
verification by a quantum-metrology theorist is required before any
analytical claim.** The numerical evidence is unambiguous; the structural
claim is plausible but unproven here.
