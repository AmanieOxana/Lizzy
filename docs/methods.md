# Methods and limits

[Get started](getting_started.md) · [Results](compiler_comparison.md)

Lizzy compiles Hamiltonian evolution, not arbitrary input matrices. PauLie owns
classification, kak-tools orthogonal decomposition, OpenFermion fermionic algebra/
Gaussian circuits, and ffsim molecular factorization. Lizzy adds routing, reusable
plans and common emission—not new claims of global optimality.

## Circuits and accuracy

Static synthesis compares exact and approximate candidates; driven Wei–Norman
integrates algebraic coordinates. Fermionic inputs use specialized upstream
algorithms. Their different accuracy contracts must not be interchanged.

`Circuit.rotations` stores $\exp(-i\theta P)$ in application order.
`logical_two_qubit_gates` is an analytical pair-block/ladder charge.
`two_qubit_gates` prices the retained quote: `builtin` is analytical;
`native-ladder`, `native-frame`, optional pytket and `clifford-t` retain concrete
artifacts. Check `emission_is_concrete`. Native phase preservation cannot restore
a phase lost by factorization. Counts assume unrestricted connectivity.

The CX router compares exact, hybrid and formula candidates per commuting
component, then quotes the full sequence. Explicit BDI keeps its algorithm;
hybrid kernels use Givens. Fixed `steps`, nonunit `calibration` and opt-in qDRIFT
do not certify formula error. Bounded cost shortlisting sets `routing_estimated`,
not an error bound. `Compiler` holds reusable settings; `_routing.py` selects
candidates without benchmark oracles.

T mode compares complete BDI-reference, legal-nullspace BDI, Givens and eligible
Gaussian circuits at one rotation budget. Its original lowest-T ladder winner
fixes a reference; shared frames may replace it only if **neither T nor CX
increases**, minimizing `(T, CX)` with stable ties. Frames use lookahead 0/8,
at most eight qubits and 256 rotations. `t_selection` retains counts/rejections;
failed optional candidates cannot discard a valid reference. No formula or
Wei–Norman fallback is silently added.

The common [pygridsynth](https://github.com/quantum-programming/pygridsynth) backend
charges snapping and generic-rotation error; opposite angles reuse inverse words.
Counts include T†. `rotation_error_bound` is a numerical triangle bound, excluding
factorization/ODE error; T-mode `error_guaranteed=False`. Scalar phase is uncharged
metadata for uncontrolled circuits, but needs implementation in controlled evolution.

## Cartan BDI and Givens

Wierichs et al., [Recursive Cartan decompositions for unitary synthesis](https://arxiv.org/html/2503.19014v2),
Sections VI.1/VI.2 and Appendices F.4–F.6, provide the general endpoint and reusable
horizontal constructions. `exact.py` derives signed $\mathfrak{so}(m)$ mappings
from generators, not named-model templates. `_orthogonal_mapping.py` supplies a
bounded general embedding when horizontality fails; upstream verifies the Lie
map. This supports mapped orthogonal components, not all Hamiltonians/Cartan types.
BDI keeps its own partition/order; Givens separately uses cheap-adjacent ordering.

For $M=\rho(+iH)$ and horizontal partition $p+q=m$, the off-diagonal generator
block's SVD gives $M=KAK^{\mathsf T}$ with $K\in SO(p)\times SO(q)$.
`prepare_bdi(H)` recursively factors both K blocks once and uses unwrapped
Hamiltonian rates in the center; the inverse wing is the reversed, negated
sequence. Only center angles change with time. A general plan caches the map
but refactors each endpoint. Zero time returns an empty circuit.

Balanced recursion uses upstream cosine-sine decomposition, four smaller
orthogonal factors and a commuting center, down to $SO(2)$. For odd size $r$,
center planes pair $i$ with $\lceil r/2\rceil+i$, leaving the middle axis unpaired.
Reflection corrections remain upstream. The signed spinor convention is

$$
\rho(+iP)=2s_P(E_{ij}-E_{ji}),\qquad s_P\in\{-1,+1\}.
$$

The desired representative is $\exp(-tM)$: a plane angle $\theta$ maps to logical
$-\theta/(2s_P)$, a central rate $\lambda$ to $t\lambda/(2s_P)$.
Application order is $K^\dagger$, center, K, opposite matrix-product order.
Givens eliminates $\exp(+tM)$ to obtain its inverse; a zero lower entry permits
skipping only a nonnegative pivot, including half-turn cases.

Horizontal rates/inverse wings retain the physical spin lift. Givens/general
endpoint BDI promise only equivalence up to phase: $H=X,t=\pi$ yields $-I$ but
orthogonal representative $I$. `plan.phase_preserving` describes the logical
plan, not arbitrary SDK optimizations; controlled evolution needs strict checks.

Balanced BDI and Givens have at most $d=m(m-1)/2$ rotation entries; BDI obeys

$$
T(r)=2T(\lfloor r/2\rfloor)+2T(\lceil r/2\rceil)+\lfloor r/2\rfloor=r(r-1)/2.
$$

A horizontal plan instead has at most

$$
2\left[\frac{p(p-1)}2+\frac{q(q-1)}2\right]+\min(p,q).
$$

This equals $d$ for $|p-q|\le1$, but can exceed it otherwise. Mirrored wings
share parameters: entries are not independent degrees of freedom or gate counts.
An unconditional $4^n-1$ entry bound is wrong. Optional `optimize="t"` tries up to
eight legal QR nullspace completions, preserving active rates and reconstruction
before the same BDI recursion; its estimate is not an actual T count.
This is Lizzy policy, not a paper optimality theorem. The separate variational
[Cartan method](https://doi.org/10.1103/PhysRevLett.129.070501) is not implemented.
Generic emission also does not implement Section VI.3.1's specialized paired
XY/YX and layer-cancellation CX construction.

## Wei–Norman

[Qvarfort–Pikovski](https://doi.org/10.1103/PRXQuantum.6.010201), Section III A,
Theorem 1 and Eqs. (14), (24), (28)–(33), and
[Altafini](https://arxiv.org/abs/quant-ph/0203005), Section II Eqs. (4)–(8),
give local product coordinates—not a globally nonsingular chart. Lizzy covers
real controls on finite-qubit Hermitian Paulis, not non-Hermitian/infinite
bosonic bases or Lindblad dynamics. With $\hbar=1$, $H=\sum_kh_kP_k$, $T_k=-iP_k$,

$$
\dot U=-iHU,\qquad U(\theta)=e^{-i\theta_1P_1}\cdots e^{-i\theta_dP_d}.
$$

For $V_{j-1}=e^{-i\theta_1P_1}\cdots e^{-i\theta_{j-1}P_{j-1}}$, define real M:

$$
V_{j-1}P_jV_{j-1}^{\dagger}=\sum_kM_{kj}P_k,\qquad
\dot U U^{-1}=-i\sum_{kj}\dot\theta_jM_{kj}P_k,\qquad M\dot\theta=h.
$$

`driven.py` constructs the closure and solves `M @ theta_dot = coefficients`,
with $M(0)=I$ and no extra i. For $PQ=isR$, $s\in\{-1,1\}$,

$$
e^{-i\theta P}Qe^{i\theta P}=\cos(2\theta)Q+s\sin(2\theta)R.
$$

Commuting directions stay fixed. These are exact adjoint planes, not truncated
BCH or endpoint fitting; integration is numerical. Altafini's corresponding
notation is $A_j=-iP_j$, $\gamma_j=\theta_j$, $a_j+u_j=h_j$, $\Xi=M$.
Independent XYZ differentiation gives

$$
M(x,y,z)=\begin{pmatrix}
1&0&\sin(2y)\\
0&\cos(2x)&-\sin(2x)\cos(2y)\\
0&\sin(2x)&\cos(2x)\cos(2y)
\end{pmatrix},\qquad\det M=\cos(2y).
$$

The chart is singular at $y=\pi/4+k\pi/2$ despite regular physical evolution.
Factors emit in reverse basis order. Restarts append left increments
$U(t_2,t_0)=U(t_2,t_1)U(t_1,t_0)$, including reverse time; identity controls
retain scalar phase. A `WeiNormanBasis` reuses closure/adjoint data, not a solved
trajectory. The wrapper adds component/pulse splitting and shared work limits.

The tutorial's Eq. (24) implies $\xi=iM$, hence $\xi(0)=iI$, not the I below
Eq. (32); Eqs. (31)/(33) then agree. Eq. (29)'s $-i$ should cancel against
$-iHU$. These are consistency observations, not an author-issued erratum.
Finite closure does not terminate nested commutators as argued near Eq. (26):
$[X,Y]=2iZ$, $[X,[X,Y]]=4Y$, etc. It permits finite-dimensional adjoint matrix
exponentials without nilpotence.

Declare all potentially active controls and known pulse `breakpoints`
(descending for reverse time). Sampled conditioning, interval bisection and
closure/work caps fail closed, but do not certify continuous nonsingularity.
For example, $H=100Y$ on $[0,0.01]$ can cross a singularity between samples.
Set `max_step` to resolve controls; narrow unknown pulses can be missed.
Local `rtol`/`atol` do not bound final operator error; closure can be exponential.
One chart has at most d angles ($4^n$ including identity); restarts can exceed it.
Neither minimal parameters nor better gate counts follow. `error_guaranteed`
is false; static auto never silently chooses Wei–Norman.

## Fermionic inputs and expansions

`gaussian.py` uses OpenFermion [quadratic diagonalization/Bogoliubov circuits](https://quantumai.google/reference/python/openfermion/circuits/bogoliubov_transform)
with `initial_state=None`: full basis wings, mode phases and scalar phase, not
state preparation. Complex hopping/pairing and direct `from_quadratic` input are
supported without dense $2^n$ synthesis. Recognition uses the supplied JW order;
nonzero interactions are rejected. Auto compares whole circuits up to eight
modes; above that cap available Gaussian synthesis delegates before DLA work.
That is a disclosed search cap, not optimality. A polynomial-size reconstruction
guard checks time-amplified upstream truncations; one compensated Fourier
fallback retains the same threshold. Remaining failures reject the candidate,
not silently start unbounded DLA work. The guard is numerical, not a certificate.

For interacting molecules keep orbital tensors. `synthesize_molecular_ffsim`
preserves upstream factor order; real spin-restricted tensors, layout, finite-step
and factorization tolerances matter. DF defaults to alpha-then-beta; Pauli
conversion defaults to interleaved and drops scalar identity energy. Interleaved
DF conversion needs a charged fermionic parity network. Optional fragment
reordering changes the approximant; fewer CX need not mean equal-accuracy savings.
`accuracy_certified` is false. The completed selector study had common coverage
only 2/8 and post-hoc headroom 2.33%; no chemistry advantage or selector was found.

`expansions.py` implements published fourth-order
[Magnus/Fer schemes](https://personales.upv.es/serblaza/2011EncyclopediaFerMagnus.pdf)
in coefficient space. Each step's one/two Pauli-sum exponentials still need
synthesis: commuting factors directly, others through Wei–Norman. Truncation
and synthesis errors are separate; Fer4 is not exactly time-reversal symmetric.
Split discontinuous pulses. These are opt-in, not new decomposition theorems.

## API reference

```{eval-rst}
.. autoclass:: lizzy.synthesize.Compiler
   :members: compile

.. autoclass:: lizzy.driven.WeiNormanBasis
   :members: jacobian

.. autofunction:: lizzy.wei_norman.synthesize_wei_norman
```

Tests should protect independent mathematical invariants, public contracts and
concrete regressions—not model×size×time grids hidden in parametrization or loops.
Wider sweeps and retired research are indexed in the
[reproduction guide](reproduce.md).
