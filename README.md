# UMTC data and consistency checks

A Python package for reading numerical anyon data from JSON and checking their consistency. It also supports the information about the ``intrinsic symmetry'' of the UMTC in terms of a GAP symmetry 
group, with a
homomorphism `rho: G -> Z2` specifying antiunitarity. 

There are two ways to supply category data:

- **Tables**, for categories such as Fibonacci or Ising: list the anyons and
  every admissible symbol, dimension and spin. Loading turns those tables into
  lookup functions.
- **Functions**, for tuple-labeled categories such as the toric code: give
  formulas in JSON, or supply Python functions directly, for fusion, F, R,
  dimensions, spins, U and eta.

Both modes expose the same callable interface. Matrix access and consistency
checks use these functions in either mode.

The toric-code JSON defines its category data with:

```json
"symbol_format": "functions",
"fusion_rules": {"expression": "1 if c == ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2) else 0"},
"quantum_dimensions": {"expression": "1"},
"topological_spins": {"expression": "(-1) ** (a[0] * a[1])"},
"F_symbols": {"expression": "1"},
"R_symbols": {"expression": "(-1) ** (a[0] * b[1])"}
```

The symbol conventions follow Section II of
[Ye and Zou, arXiv:2210.02444](https://arxiv.org/abs/2210.02444).
See [the conventions and equations](docs/conventions.md) for matrix orientations
and [the JSON format](docs/json-format.md) for the complete input schema.

## Install and run

Python 3.10 or newer is required, with no additional Python dependencies.
GAP 4 must be installed with `gap` on `PATH` to load GAP symmetry groups,
including the toric-code example. On macOS, install it with `brew install gap`;
see [GAP downloads](https://www.gap-system.org/Download/) for other systems.
Fibonacci and legacy table-defined symmetries do not require GAP.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

umtc-check examples/toric_code.json
umtc-check examples/fibonacci.json
```

`python -m umtc` is equivalent to `umtc-check`. To use the checkout without
installing anything:

```sh
PYTHONPATH=src python -m umtc examples/fibonacci.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Python API

```python
from umtc import load_json, check_pentagon, check_hexagon, check_symmetry

tc = load_json("examples/toric_code.json")
one, e, m, psi = (0, 0), (1, 0), (0, 1), (1, 1)

assert tc.N(e, m, psi) == 1
assert tc.fusion(e, m) == {psi: 1}
assert tc.F(e, m, e, m, psi, psi) == 1
assert tc.R(e, m, psi) == -1
assert tc.spin(psi) == -1
assert tc.quantum_dimension(e) == 1

assert check_pentagon(tc) is True
assert check_hexagon(tc) is True
assert check_symmetry(tc) is True
assert tc.check_all() is True

sym = tc.symmetry
assert sym is not None
assert sym.action("X", e) == m
assert sym.action("T", e) == m
assert sym.G.mul("X", "T") == "X*T"
assert sym.rho("X") == 0
assert sym.rho("T") == 1
assert sym.is_antiunitary("T")
assert sym.U("T", e, m, psi) == -1
assert sym.eta(psi, "T", "T") == -1

fib = load_json("examples/fibonacci.json")
assert fib.symmetry is None
assert fib.check_all()
print(fib.F_matrix("tau", "tau", "tau", "tau"))
print(fib.R("tau", "tau", "1"))
```

To supply actual Python functions, use the same category data with
`UMTC.from_functions`:

```python
import json
from umtc import UMTC

with open("examples/toric_code.json") as handle:
    data = json.load(handle)

def F(a, b, c, d, e, f):
    return 1

def R(a, b, c):
    return (-1) ** (a[0] * b[1])

def N(a, b, c):
    return int(c == ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2))

def quantum_dimension(a):
    return 1

def spin(a):
    return (-1) ** (a[0] * a[1])

def U(g, a, b, c):
    return R(a, b, c) if g in ("X", "T") else 1

def eta(a, g, h):
    return spin(a) if g in ("X", "T") and h in ("X", "T") else 1

def action(g, a):
    return (a[1], a[0]) if g in ("X", "T") else a

tc = UMTC.from_functions(
    data, F=F, R=R, N=N, quantum_dimension=quantum_dimension,
    spin=spin, U=U, eta=eta, action=action,
)
assert tc.check_all()
```

Each provided function replaces its corresponding entry in `data`, which may
then be omitted. Omitted or `None` arguments preserve the existing definition.
F and R must both be functions in the resulting data; when starting from a
fully tabulated file, supply both F and R callbacks.
An action-only override preserves the existing F/R mode.
U, eta and action callbacks require the symmetry group and grading in
`data`. Function mode also accepts explicit tables for fusion, dimensions,
spins, U or eta, so these fields can be converted individually.

The toric-code symmetry specifies its group and grading compactly:

```json
"group": {"gap": "AbelianGroup([2,2])", "generator_names": ["X", "T"]},
"rho": {"generator_images": [0, 1]},
"action": {"expression": "(a[1], a[0]) if g == 'X' or g == 'T' else a"}
```

GAP constructs the finite group and checks that the generator images extend to
a homomorphism into Z₂. The identity has label `"1"`; other elements use
canonical generator words such as `"X*T"`, exposed in `sym.G.elements`.
The action is a function in either category mode. Its group law and compatibility
with the category are tested by `check_symmetry`.

You can also construct the group and homomorphism in Python:

```python
from umtc import GAPGroup

G = GAPGroup.from_gap("AbelianGroup([2,2])", ["X", "T"])
rho = G.homomorphism_to_z2([0, 1])
data["symmetry"]["group"] = G
data["symmetry"]["rho"] = rho
tc = UMTC.from_functions(data, action=action)
assert tc.symmetry.G is G
assert tc.symmetry.rho("X*T") == 1
```

F, R and U only need to define admissible coefficients: the package returns
zero for forbidden channels before calling them. For nontrivial fusion
multiplicities, F must also accept `alpha,beta,mu,nu`, and R and U must accept
`mu,nu`. These index arguments can be positional or keyword-only. Functions
must be deterministic. Fusion multiplicities must be nonnegative integers;
dimensions must be finite real numbers, and the other symbols finite complex
numbers.

The main accessors are:

| Accessor | Result |
| --- | --- |
| `cat.anyons`, `cat.vacuum` | Ordered anyon labels and tensor unit |
| `cat.N(a,b,c)`, `cat.fusion(a,b)` | Fusion multiplicity and nonzero outcomes |
| `cat.fusion_rules(a,b,c)` | Callable alias of `cat.N(a,b,c)` |
| `cat.F(a,b,c,d,e,f,alpha=0,beta=0,mu=0,nu=0)` | One F coefficient |
| `cat.F_matrix(a,b,c,d)` | Full immutable F matrix |
| `cat.F_left_basis(...)`, `cat.F_right_basis(...)` | The ordered row/column labels |
| `cat.R(a,b,c,mu=0,nu=0)`, `cat.R_matrix(a,b,c)` | One R coefficient or its matrix |
| `cat.F_symbols(...)`, `cat.R_symbols(...)` | Callable aliases of `cat.F(...)` and `cat.R(...)`, in both modes |
| `cat.symbol_format` | `"tables"` or `"functions"` |
| `cat.spin(a)`, `cat.quantum_dimension(a)` | Twist phase θ and quantum dimension |
| `cat.topological_spins(a)`, `cat.quantum_dimensions(a)` | Callable aliases of `cat.spin(a)` and `cat.quantum_dimension(a)` |
| `cat.total_quantum_dimension` | Square root of the sum of squared dimensions |
| `sym.G`, `sym.rho(g)` | GAP group and antiunitary parity (0 or 1) |
| `sym.action(g,a)` | Callable anyon action; alias of `sym.act(g,a)` |
| `sym.act(g,a)`, `sym.mul(g,h)`, `sym.inverse(g)` | Anyon permutation and group operations |
| `sym.U(g,a,b,c,mu=0,nu=0)`, `sym.U_matrix(g,a,b,c)` | U coefficient or matrix, using destination labels |
| `sym.eta(a,g,h)` | Symmetry fractionalization phase |
| `sym.U_symbols(...)`, `sym.eta_symbols(...)` | Callable aliases of `sym.U(...)` and `sym.eta(...)` |
| `s_matrix(cat)` | Normalized modular S matrix in `cat.anyons` order |

Indices are zero based. Known but inadmissible scalar F/R/U channels and
out-of-range nonnegative multiplicity indices return `0j`. Unknown labels and
negative or noninteger indices raise `DataError`. Missing admissible symbol
records in table mode raise `DataError` when loading; they never silently become
identities. Formula syntax and native function signatures are checked at loading.
Fusion functions are evaluated and memoized while loading to establish the
fusion channels and bases. Other functions are evaluated when accessed or
checked. Evaluation errors raise `DataError` identifying the symbol and its
arguments.

`UMTC.from_dict(data)` accepts a decoded JSON dictionary. `UMTC.from_json(path)`
and `load_json(path)` read files. Loaded objects and their table data are
read-only; edit the JSON/dictionary and load a new object to change data.
All symbol accessors and their aliases are functions in both modes.

## What is checked

Each Boolean function accepts `atol` and `rtol`, both defaulting to `1e-9`.

| Function | Conditions |
| --- | --- |
| `check_pentagon(cat)` | Every coefficient of the pentagon, summing all channels and multiplicity indices |
| `check_hexagon(cat)` | Both braiding hexagons, plus F/R unitarity used to replace inverses by adjoints |
| `check_symmetry(cat)` | Group laws, antiunitary grading, anyon action, fusion/spin/dimension preservation, U unitarity, U/eta normalization, F/R covariance, U composition and the twisted eta cocycle |
| `check_fusion(cat)` | Unit, commutativity, associativity, unique duals and positive dimension identities |
| `check_unitarity(cat)` | Every F and R matrix |
| `check_ribbon(cat)` | Twist phases, dual twists, the R trace formula and balancing |
| `check_modularity(cat)` | Unitarity of normalized S |
| `check_all(cat)` | All of the above |

The same functions are available as `cat.check_pentagon()`, etc. More focused
symmetry functions are exported as `check_symmetry_action`, `check_symmetry_f`,
`check_symmetry_r`, `check_u_consistency`, and `check_eta_consistency`; these also
verify the prerequisites needed for their equations. A missing symmetry section
means no specified symmetry, so symmetry checking returns `True` with no equations.

For diagnostics, use:

```python
from umtc import coherence_report

report = coherence_report(tc, checks=["pentagon", "hexagon", "symmetry"])
print(report.ok, report.checked, report.failure_count, report.max_error)
for failure in report.failures:
    print(failure.equation, failure.labels, failure.lhs, failure.rhs)
```

The report evaluates every selected equation, counts all failures, and retains
the first 20 by default. Set `max_failures=None` to retain every failure. Labels
include the anyons and matrix or fusion-space indices needed to locate the
problem. `report.to_dict()` can be serialized as JSON; a nonfinite arithmetic
result is represented by `null` in this diagnostic export.

```sh
umtc-check examples/fibonacci.json --checks pentagon hexagon --atol 1e-10
umtc-check examples/toric_code.json --json
```

Exit codes are **0** for passing checks, **1** for failed equations, and **2** for
malformed data, file errors or invalid command arguments.

## Examples

The [example catalog](examples/README.md) contains 26 JSON files, including the
category families in arXiv:2309.15118 and arXiv:2608.14180: U(1), all eight
Ising variants, toric code, Z₃ and Z₄ gauge theories, doubled U(1), and SU(2)
at levels 1, 2, 3, 4, 6 and −3, together with Fibonacci. The paper examples
include the full intrinsic symmetry for their selected parameters.
The catalog documents a normalization
correction needed for the first paper's doubled-U(1) formulas.

Toric code and Fibonacci are described below.

### Toric code with Z₂ × Z₂ᵀ

[toric_code.json](examples/toric_code.json) uses integer tuples
`a=(x,y)`, with fusion given by componentwise addition modulo two:

| Label | Anyon | Quantum dimension | Twist |
| --- | --- | --- | --- |
| `(0,0)` | 1 | 1 | 1 |
| `(1,0)` | e | 1 | 1 |
| `(0,1)` | m | 1 | 1 |
| `(1,1)` | ψ | 1 | −1 |

For admissible channels, F=1 and R(a,b)=(-1)^(xₐ yᵦ).
The order-two generators X and T both exchange e and m, commute, and square
to the identity; T and XT are antiunitary. Their product XT fixes all anyons.

Let p(X)=p(T)=1 and p(1)=p(XT)=0. The example specifies the consistent choice

```text
U_g(a,b;a+b) = R(a,b) ^ p(g)
eta_a(g,h)   = theta_a ^ (p(g)*p(h))
```

In particular, `eta_psi(T,T) = eta_psi(X,X) = -1`. These phases are part of
the supplied symmetry data. The file gives one consistent choice of symmetry
fractionalization; the permutation alone does not specify all of these phases.

### Fibonacci

[fibonacci.json](examples/fibonacci.json) uses the flat list `["1", "tau"]`,
with τ×τ=1+τ, dτ=φ=(1+√5)/2, and θτ=exp(4πi/5). Its only nontrivial F matrix is

```text
F^(tau,tau,tau)_tau = [[1/phi,        1/sqrt(phi)],
                      [1/sqrt(phi), -1/phi     ]]
```

in channel order `["1", "tau"]`. R(τ,τ;1)=exp(−4πi/5) and
R(τ,τ;τ)=exp(3πi/5). No intrinsic symmetry data are specified.

Fibonacci supplies its fusion rules, dimensions, spins and every admissible F/R
matrix explicitly; the loader wraps them as functions. The toric code supplies
compact expressions for fusion, dimensions, spins, F, R, U, eta and the anyon
action, together with its GAP group and parity homomorphism. Both files can be regenerated
deterministically with `python scripts/generate_examples.py`.

## Scope and verification

Checks use floating-point complex arithmetic, so passing means agreement within
the chosen tolerances, not an exact symbolic proof. Group tables and structural
conditions are checked exactly. Symmetry groups must be finite; GAP loading
defaults to an order limit of 256, configurable through the Python group API.
Exhaustive checks can be expensive well below that bound.
Continuous groups and automatic discovery of symmetries are outside
this package. These equations do not calculate a symmetry anomaly.

`check_all` covers the identities listed above. Evaluation/coevaluation maps and
unitors are not input fields; no particular vacuum-vertex gauge for F is enforced.

The test suite includes both examples, corrupted-symbol negatives, complex vertex
gauges, equivalence of tables/formulas/native functions, an order-three anyon
permutation, and independently constructed Rep(A₄)
data with fusion multiplicity two and complex offdiagonal braiding. Rep(A₄) is
intentionally nonmodular, providing a negative test for modularity.
