# Exact synthesis: paper correspondence and scope

Reviewed against the implementation on 2026-10-07. This note concerns
`lizzy.exact`, not the independent two-qubit KAK kernel or numerical Wei–Norman
solver. The relevant source is Wierichs et al.,
[Recursive Cartan decompositions for unitary synthesis](https://arxiv.org/html/2503.19014v2),
Sections VI.1–VI.3 and Appendices F.3–F.6.

## What changed after the first BDI comparison

The first BDI integration applied recursive cosine-sine decomposition to the
same reordered endpoint matrix used by Givens. That was a valid orthogonal
factorization, but it did not retain the paper's horizontal mapping or reuse
its time-independent outer transformation. Its results remain in the
[historical record](../experiments/bdi_results.json); they are not the results
of the current BDI implementation.

BDI now derives its representation from the actual Pauli generators, independently
of Givens' low-weight-plane heuristic. This is an algebraic dispatch, not a lookup
for named TFIM/TFXY/XY models. It selects the horizontal construction when the
generator structure admits one; otherwise it uses the general small-representation
pipeline. The support boundary is still a successfully mapped `so(m)` algebra
within resource limits, not all Hamiltonians or all classical Cartan types.

## Code path and paper correspondence

The implementation in [exact.py](../lizzy/exact.py) separates four responsibilities:

1. Eligibility uses PauLie classification and explicit resource limits. An
   eligible label is not a guarantee that every presentation can be embedded.
2. Representation construction maps Pauli generators to signed coordinate
   planes. BDI first seeks a horizontal mapping of the supplied terms, preserving
   its coordinate order and partition. If none exists, it derives a general
   embedding from the Pauli closure in
   [_orthogonal_mapping.py](../lizzy/_orthogonal_mapping.py), then asks upstream
   to verify the complete signed Lie map. This adapter is Lizzy's implementation
   choice within the general representation step, not a new claim in the paper.
   Givens retains its original horizontal mapping where available and now uses
   the same verified general-embedding fallback when horizontality is impossible.
   It separately applies its cheap-adjacent heuristic; its automatic eligibility
   filter remains narrower. Sharing the representation fallback does not replace
   either method's factorization or impose Givens' ordering on BDI.
3. A horizontal Hamiltonian is factored at the **generator level**, giving a
   reusable `K A(t) K†` plan. Other supported presentations use recursive BDI on
   the time-evolution matrix. Both use the upstream balanced BDI recursion for
   orthogonal blocks, rather than substituting Givens for the BDI wings.
4. Terminal planes become Pauli rotations with the signed spinor scale below.
   Circuit emission remains a separate layer.

The general matrix route corresponds to Section VI.1. The reusable horizontal
route corresponds to Section VI.2. The generic horizontal mapping follows
Appendix F.6; Appendix F.3's ordered Jordan–Wigner/Majorana construction is the
paper's manual example, **not a model-specific template selected by Lizzy**.
Balanced recursive BDI is the construction of Appendix F.4, with the plane-to-
Pauli conversion of Appendix F.5.

`synthesize(..., method="bdi")` applies the explicit BDI route to each commuting
summand. It does not silently switch to Givens or a product formula. The
automatic router compares BDI and Givens, with an additional upstream Gaussian
candidate for JW-quadratic inputs; hybrid formula kernels retain Givens.
This routing policy is separate from, and does not alter, the paper recursion. A standalone
`prepare_bdi(H)` prepares one supported component for repeated calls to
`plan.circuit(time)`; the high-level API handles commuting-component splitting.
The plan reports `mapping_kind` (`"horizontal-graph"` or `"general-graph"`),
`partition`, `irrep_size`, `parameter_bound` and `phase_preserving`.

Kökcü et al.'s [variational Cartan method](https://doi.org/10.1103/PhysRevLett.129.070501)
is a different algorithm. Lizzy does not implement its optimization over a
parameterized `K`; obtaining a `K A(t) K†` circuit here does not make those
numerical procedures identical.

## Horizontal generator factorization and reuse

Write the mapped skew generator as `M = rho(+iH)`. For a horizontal partition
`p + q = m`, its diagonal blocks vanish and its off-diagonal block determines
the Hamiltonian. Upstream `horizontal_generator_decomposition` uses an SVD of
that block to obtain `M = K A K.T`, correcting determinant signs so that the two
diagonal blocks of `K` belong to `SO(p)` and `SO(q)`.

The central coefficients are signed **Hamiltonian rates**, not angles recovered
from a finite-time matrix exponential. They are never divided by the requested
time or wrapped modulo `2*pi`. Each diagonal block of `K` is recursively BDI-
factorized once. The opposite circuit wing is made by reversing and negating
that same rotation sequence, not by independently factorizing `K.T`.

Consequently, repeated evolution times change only the central rates' multiplier.
The generator spectrum and both outer wings stay fixed. Preparation and
evaluation are therefore distinct costs in the [benchmark](bdi_benchmark.md).
A general, nonhorizontal plan instead caches its mapping and recompiles each
time-dependent endpoint. Zero time is exactly an empty circuit, rather than a
tolerance-based shortcut.

## Balanced recursion and angle counts

The opt-in `prepare_bdi(..., optimize="t")` changes only legal nullspace
completion choices in the initial horizontal SVD. These transformations commute
with the Cartan generator and are independently checked to preserve H. Each
candidate still uses the same recursive BDI and factor order. This bounded
QR-completion search is **Lizzy's optimization policy**, not an algorithm or
optimality theorem attributed to the paper. Sections V.3 and VI.1 motivate
rotation-aware costs and choice of recursion/bases, but do not specify this
search. The original decomposition remains the default and fallback.
See [T-aware synthesis](fault_tolerant.md) for actual compiled counts and limits.

Each general `SO(r)` block is split into `floor(r/2)` and `ceil(r/2)` parts.
Upstream cosine-sine decomposition supplies the Cartan factor and four smaller
orthogonal factors; the recursion stops at `SO(2)`. Reflection corrections are
handled upstream. Lizzy reads the factor tree in circuit-application order and
maps every terminal plane back to a Pauli rotation.

For odd blocks, commuting Cartan planes pair index `i` with
`ceil(r/2) + i`; the middle axis is unpaired. Balanced recursion has
`T(r) = 2 T(floor(r/2)) + 2 T(ceil(r/2)) + floor(r/2) = r(r-1)/2`
angular slots. Vanishing or foldable angles may reduce the emitted sequence.

A horizontal plan instead has at most

```text
2 * [p(p-1)/2 + q(q-1)/2] + min(p, q)
```

rotation entries: two identical-size outer wings and one commuting center.
This equals `dim so(m)` when `abs(p-q) <= 1`; an unbalanced initial partition
can exceed that dimension. The mirrored wings share parameters, so rotation
entries should not be confused with independent tunable degrees of freedom.
Neither count proves minimal CX cost or hardware depth.

## Signs, factor order, and global phase

For a Pauli word `P` mapped to plane `(i,j)`, the bridge convention is

```text
rho(+i P) = 2 s_P (E_ij - E_ji),     s_P in {-1, +1}.
```

Lizzy requests `exp(-itH)`, so its small representative is `exp(-t M)`.
A plane with matrix angle `theta` maps to circuit angle `-theta/(2*s_P)`.
A central generator rate `lambda` maps to circuit angle `t*lambda/(2*s_P)`.
The horizontal circuit applies the `K†` wing first, then the central gates,
then the `K` wing; matrix multiplication reads this in reverse.

The horizontal construction preserves the physical spin lift: the two wings
are exact inverses in the Pauli representation, and the middle uses unwrapped
generator rates. Its logical circuit therefore preserves global phase within
floating-point error, including long times and resonances.
`plan.phase_preserving` records this distinction explicitly.
This flag applies to the logical plan. Native emission preserves its phase;
an optional SDK artifact requires its own phase-sensitive verification.

In contrast, endpoint factorization alone does not recover that lift. For
`H=X, t=pi`, the physical evolution is `-I` while its orthogonal representative
is `I`. Givens and general nonhorizontal endpoint BDI promise only equivalence
up to global phase. Preserving an emitted circuit's phase cannot repair a phase
already lost by its logical decomposition. Use the horizontal or Wei–Norman
phase-preserving contract when controlled evolution requires it.

Givens has a separate sign convention because it eliminates `exp(+t M)` to
produce its inverse. Its 2026-10-07 negative-pivot fix remains important: a
zero lower entry permits skipping elimination only when the pivot is
nonnegative. This corrects a genuine half-turn error, not merely a global sign.

## Validation and remaining comparison boundaries

The focused [exact-synthesis tests](../tests/test_exact.py) use
independent dense Pauli commutators, physical evolution and matrix-product
reconstruction to check signs, ordering, half-turns, odd blocks and angle
bounds. The same suite checks explicit component dispatch, unsupported inputs
and preservation of the requested method; separate
[mapping tests](../tests/test_orthogonal_mapping.py) verify all Lie brackets.
Horizontal validation must also check generator reconstruction, unchanged wings
across times and strict physical evolution, not just an endpoint in `SO(m)`.
Generic encoded and nonhorizontal inputs guard against model-specific dispatch.

The [current benchmark](../experiments/bdi_benchmark.py) compares actual emitted
artifacts while retaining each algorithm's representation choice. Independent
native ladders and the shared native-frame emitter are useful controlled
baselines, but are **not the paper's optimized CX construction**. In particular,
the specialized paired `XY/YX` circuits and layer cancellations discussed in
Section VI.3.1 are not implied by using the correct mathematical decomposition.

No result here establishes a universal BDI-versus-Givens ranking, performance
parity with the paper's large-system timings, or arbitrary-unitary coverage.
See [the synthesis guide](synthesis.md) for API and emission contracts and the
[Wei–Norman review](wei_norman_review.md) for its separate paper correspondence.
