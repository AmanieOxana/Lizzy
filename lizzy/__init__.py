"""Structure-aware synthesis for Pauli Hamiltonians.

The stable top-level interfaces are lizzy.synthesize (Compiler, Result and
synthesize) and lizzy.hamiltonian (Hamiltonian helpers and the logical Circuit).

Specialist modules are grouped by responsibility:
- algebra: classification, symmetries and representation maps;
- synthesis: exact/product-formula/numerical algorithms and routing;
- emission: native and Clifford+T gates, local kernels and backend adapters;
- fermions: Gaussian and molecular synthesis through optional upstream libraries.

Import specialist modules explicitly. These namespaces do not eagerly load
optional SDKs. Dense verification and HamLib dataset access remain in dense
and hamlib; benchmark drivers live outside the installed package.
"""
