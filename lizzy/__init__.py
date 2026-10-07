"""Structure-aware synthesis for Pauli Hamiltonians.

Public entry points live in ``lizzy.synthesize`` (static routing),
``lizzy.wei_norman`` (numerical static/driven synthesis), and
``lizzy.gaussian`` (optional OpenFermion quadratic synthesis), and
``lizzy.chemistry`` (optional molecular adapters). ``lizzy.clifford_t`` provides
opt-in fault-tolerant emission. Import these modules explicitly
to keep optional SDK and chemistry dependencies out of the package namespace.
"""
