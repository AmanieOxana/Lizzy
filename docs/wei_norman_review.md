# Wei–Norman implementation review

Reviewed against the primary literature on 2026-09-25 and rechecked during the
code-organization audit on 2026-10-07. Lizzy implements the Wei–Norman
product-coordinate equations for finite-qubit, closed-unitary dynamics. It does
not implement every framework or example in the cited papers.

## Sources and scope

Sofia Qvarfort and Igor Pikovski, [*Solving Quantum Dynamics with a Lie-Algebra
Decoupling Method*, PRX Quantum **6**, 010201 (2025)](https://doi.org/10.1103/PRXQuantum.6.010201),
Sec. III A, especially Theorem 1 and Eqs. (14), (24), (28)–(33), gives the
product-of-exponentials construction. The theorem states local existence near
the initial time, not a globally nonsingular fixed-order chart.
The [published PDF](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.6.010201)
is available under CC BY 4.0.

Claudio Altafini, [*On the generation of sequential unitary gates from continuous
time Schrödinger equations driven by external fields*, arXiv:quant-ph/0203005](https://arxiv.org/abs/quant-ph/0203005),
Sec. II, Eqs. (4)–(8), explicitly treats the coordinate Jacobian and its local
invertibility; Sec. II A discusses its singular set. This supports treating
coordinate breakdown separately from the perfectly regular physical propagator.

Lizzy covers real coefficients multiplying fixed Hermitian Pauli words. It does
not implement the tutorial's general non-Hermitian operator bases, infinite
bosonic Hilbert spaces, Gaussian phase-space machinery, or Lindblad dynamics.

## Fidelity verdict

The production core is a numerical implementation of the paper's Lie-algebra
decoupling method, not an unrelated variational ansatz: it constructs the control
closure, derives the coordinate Jacobian from adjoint actions, and integrates
the resulting Wei–Norman equations. It neither fits endpoint unitaries nor uses
a product-formula approximation inside a chart. Numerical integration still
introduces error.

Lizzy-specific engineering surrounds that core: deterministic basis ordering,
commuting-component splitting, pulse boundaries, condition-triggered interval
bisection, work limits, and native gate emission. These are not new theorems
from the papers. In particular, restarts concatenate locally valid factors;
they do not provide a globally nonsingular minimal parameterization.

## Derivation in Lizzy's convention

Set \(\hbar=1\), \(H(t)=\sum_k h_k(t)P_k\), and \(T_k=-iP_k\). The real Lie algebra
is spanned by these anti-Hermitian \(T_k\); the stored basis contains their
Hermitian Pauli labels. The desired equation is

\[
\dot U=-iH(t)U,\qquad
U(\theta)=e^{-i\theta_1P_1}\cdots e^{-i\theta_dP_d}.
\]

Define the prefix \(V_{j-1}=e^{-i\theta_1P_1}\cdots
e^{-i\theta_{j-1}P_{j-1}}\) and the **real** matrix \(M\) by

\[
V_{j-1}P_jV_{j-1}^{\dagger}=\sum_k M_{kj}(\theta)P_k.
\]

Differentiating the ordered product gives

\[
\dot U U^{-1}=-i\sum_{kj}\dot\theta_jM_{kj}P_k=-iH,
\qquad M(\theta)\dot\theta=h(t).
\]

Thus the code solves `M @ theta_dot = coefficients`, with no extra factor of
\(i\). Initially \(M(0)=I\). A single-generator check reduces immediately to
\(\dot\theta=h\), hence \(U=\exp[-iP\int h(t)dt]\).

For anticommuting Pauli words with \(PQ=isR\), \(s\in\{-1,1\}\),

\[
e^{-i\theta P}Qe^{i\theta P}
=\cos(2\theta)Q+s\sin(2\theta)R.
\]

This checks both the sign and the factor of two. Commuting directions are
unchanged. No truncated BCH expansion is used.

### Cross-check against Altafini's convention

In Altafini's Eqs. (4)–(8), choose the skew-Hermitian basis \(A_j=-iP_j\),
\(\gamma_j=\theta_j\), and \(a_j+u_j=h_j\). Then \(\Xi=M\), giving the same
real coordinate equation directly. His later single-qubit example uses a
different generator normalization and a repeated-axis ZYZ product; Lizzy uses
each distinct basis element once per chart, not that specific Euler chart.
Unlike the phase-insensitive state analysis in his Sec. III A, Lizzy retains
an identity generator and the associated scalar phase when supplied.

For example, independently differentiating the XYZ product in Lizzy's units
gives

\[
M(x,y,z)=
\begin{pmatrix}
1 & 0 & \sin(2y)\\
0 & \cos(2x) & -\sin(2x)\cos(2y)\\
0 & \sin(2x) & \cos(2x)\cos(2y)
\end{pmatrix},\qquad \det M=\cos(2y).
\]

This chart is regular at zero and singular at \(y=\pi/4+k\pi/2\). The
closed-form regression in [the coordinate-solver tests](../tests/test_driven.py) checks both the
Jacobian and this singular set, separately from dense numerical references.

### Reading the tutorial's notation carefully

The published Eq. (24) defines conjugation coefficients using an explicit
\(-i\) prefactor. In that notation, \(\xi=iM\), so Eqs. (31) and (33) reduce to
the same physical equation above. It would be wrong to insert their printed
\(i\) into Lizzy's **real** \(M\) equation. With Eq. (24)'s definition,
\(\xi(0)=iI\), not the \(I\) stated below Eq. (32). Also, Eq. (29) retains a
\(-i\) that should cancel when equating the differentiated product with
\(-iHU\). These are consistency checks of the cited version, not an
author-issued erratum.

The argument around Eq. (26) additionally infers termination of nested
commutators from finite closure. That inference is not valid generally:
\([X,Y]=2iZ\), \([X,[X,Y]]=4Y\), and the sequence continues. Finite closure
instead gives a finite-dimensional adjoint **matrix exponential**, which is
analytic without nilpotence. Lizzy evaluates its Pauli rotation planes exactly.

## Correspondence to code

| Mathematical operation | Implementation |
| --- | --- |
| Closure under commutators of declared controls | `driven._closure` |
| Signed adjoint rotation planes | `driven._adjoint_pairs` |
| Prefix-conjugated basis columns of \(M\) | `driven._coordinate_matrix` |
| Numerical solution of \(M\dot\theta=h\) | `driven.synthesize_driven` |
| Commuting-component and pulse decomposition | `wei_norman.synthesize_wei_norman` |
| Phase-preserving concrete native gates | `native.ladder_circuit`, `native.native_frame_circuit` |
| Shared scalar/time/solver-option validation | `_numerical` (no synthesis or mathematical approximation) |

The logical circuit stores application order, opposite to the displayed matrix
product, so each chart emits its factors in reverse basis order. Restarting
from zero coordinates on the next time interval appends a **left increment**:
\(U(t_2,t_0)=U(t_2,t_1)U(t_1,t_0)\). This remains valid for driven Hamiltonians
and backwards integration.

The high-level API is `synthesize_wei_norman`, accepting a static Pauli sum or a
`DrivenHamiltonian`, with optional pulse `breakpoints` and concrete native gate
emission. `synthesize_driven` is the lower-level single-closure compiler for
smooth controls; its time argument is an endpoint pair, and it returns logical
Pauli rotations. The explicit static `synthesize(..., method="wei-norman")`
route calls the high-level API. The automatic static router does not select
Wei–Norman. Magnus/Fer in `expansions.py` are separate approximation methods,
not part of this implementation of the Wei–Norman equations.

## Integrated policy and remaining limitations

- `chart_radius=None` selects compact, condition-limited numerical charts.
  `chart_radius=0.5` retains the earlier conservative radius policy; positive
  smaller radii are also supported. Both policies retain bounded interval
  bisection/restarts and fail without returning partial output when a work
  budget is exhausted. The static automatic router is unchanged.
- Conditioning is checked at RHS evaluations and accepted ODE mesh states, not
  every continuous-time point. For example, an XYZ chart for \(H=100Y\) over
  \([0,0.01]\) can cross its singularity between samples while still returning
  a correct endpoint. A sampled guard is not a certified chart atlas.
- Local `rtol`/`atol` do not bound final operator error. Set `max_step` to resolve
  the fastest controls and supply known pulse breakpoints; adaptive integration
  may miss narrow pulses. `error_guaranteed` remains false.
- Closure includes all declared Pauli controls, including initially zero ones.
  Exact correlations between composite controls can make a smaller abstract
  algebra, but are not inferred from black-box coefficient samples. The closure
  cap is a resource limit, not a claim of generic polynomial scaling.
- Identity coefficients remain central and retain their global phase. Tests
  compare the actual native unitary with independent analytic or dense
  references in strict operator norm, not only after phase alignment.
- Factor order affects conditioning and gate cost. Compact charts remove an
  artificial radius overhead but do not optimize order, guarantee one chart,
  certify minimum CX counts, or establish superiority over Cartan synthesis.

The independent Jacobian, analytic rotating-frame, reverse-time, global-phase,
singularity, resource-limit, and dense-reference tests check different failure
modes. The bounded results in `experiments/wei_norman_compact_results.json`
are a retained historical record motivating the chart-policy change, not a
second maintained solver or a chemistry/scalability claim. The superseded
prototype and its monkeypatch comparison harness have been retired; production
stress cases now live in `experiments/wei_norman_validation.py`.

## Integration validation

The 2026-10-07 refactor centralizes input validation and removes the high-level
API's dummy zero-time compilation used solely to check options. It does not
change closure, adjoint/Jacobian construction, the ODE, factor order, chart
policy, numerical defaults, or phase handling. Production math tests remain
independent of the retired prototype.

The 2026-10-07 rerun of the consolidated production harness on all 27 targets and
both step policies gave **52 PASS and two expected CAP** results; the maximum
passing strict error was `4.583e-11`. The historical snapshots below were not
overwritten by that rerun. Routine tests now use representative cases and retain
the independent equation, phase and failure-mode checks; the full target sweep
remains in the production validation harness.

The [production validation snapshot](wei_norman_validation.json) exercises 27
targets with both `max_step=0.025` and the adaptive default, using the actual
integrated API. All 52 in-cap runs passed strict, phase-sensitive operator-norm
checks; two runs hit the expected default dimension cap for the same `su(8)`
target. The largest passing strict error was approximately `4.59e-11` against
the `1e-6` target. This is empirical validation, not a bound for unseen controls.

The refreshed [driven benchmark](driven_benchmark.txt) emits 32 CX for driven
TFIM3, versus 256 with the historical radius-limited policy. The
[static/driven comparison](wei_norman_benchmark.json) and
[raised-cap su(8) comparison](wei_norman_su8_benchmark.json) retain unsupported
routes and resource caps explicitly. Static TFIM3 still favors the existing
exact route on CX count (28 versus 32).
