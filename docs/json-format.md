# JSON format, version 1

Category data have two input modes, selected by `"symbol_format": "tables"` or
`"symbol_format": "functions"`. If omitted, the mode is inferred from the F/R
definitions, preserving existing table files. Both symbols must use the same mode.
Table mode uses explicit lists for category symbols. Function mode requires functional F
and R and also accepts functions for fusion, dimensions, spins, U and eta. Each
of those additional fields can independently retain its explicit table format.
The symmetry group, grading and action are independent of this choice: either
mode accepts a GAP group, a homomorphism into Z₂ and a functional anyon action.

Table mode is suitable for listed anyons such as Fibonacci and Ising. Function
mode is suitable for formulas acting on tuple labels, such as the toric code.
The representation does not impose a fusion law or restrict the label type.
In Python, **both modes produce callable symbols**, with the same signatures.

## Anyon labels

Use either a flat list of string/integer labels:

```json
"anyons": ["1", "tau"]
```

or tuples of integers, represented as JSON arrays:

```json
"anyons": [[0, 0], [0, 1], [1, 0], [1, 1]]
```

The equivalent explicit tuple form is
`{"type":"tuples","labels":[[0,0],[0,1],[1,0],[1,1]]}`. For a Cartesian
product of integer ranges, use:

```json
"anyons": {"type": "tuples", "moduli": [2, 2]}
```

This enumerates tuples in `itertools.product(range(2),range(2))` order. It only
defines labels: **fusion is still read from `fusion_rules`**. No Abelian group
law is inferred from the labels. Tuples must have a common length. Lists must be
nonempty and labels unique. String `"1"` and integer `1` are distinct labels.

Tuple labels become Python tuples; accessors also accept integer lists for
convenience. Matrix channel order follows the anyon list or generated order.

## Numbers

Real numbers can be JSON numbers. Complex numbers use exactly two fields:

```json
{"re": -0.8090169943749473, "im": -0.5877852522924732}
```

Booleans, `NaN`, infinities and string-valued table entries are rejected. Quantum
dimensions must be finite real numbers; their positivity is checked by
`check_fusion`. Spins store the **twist phase θ=exp(2πih)**, not the real number h.

## Category fields

| Field | Contents |
| --- | --- |
| `schema_version` | Integer `1` |
| `name` | Optional display string |
| `symbol_format` | Optional `"tables"` or `"functions"`; inferred when omitted |
| `anyons` | One of the forms above |
| `vacuum` | A label in `anyons` |
| `fusion_rules` | Records `{"a":a,"b":b,"c":c,"multiplicity":N}` for positive integer N, or a function expression object |
| `quantum_dimensions` | One `{"anyon":a,"value":d}` for each anyon, or a function expression object |
| `topological_spins` | One `{"anyon":a,"value":theta}` for each anyon, or a function expression object |
| `F_symbols` | Table records `{"a":a,"b":b,"c":c,"d":d,"matrix":[...]}`, or a function expression object |
| `R_symbols` | Table records `{"a":a,"b":b,"c":c,"matrix":[...]}`, or a function expression object |
| `symmetry` | Optional symmetry object, or `null` |

In a fusion table, an unlisted triple has multiplicity zero. A fusion function
must return zero for such triples.

## Scenario 1: explicit tables become functions

Use `"symbol_format": "tables"`, list the nonzero fusion multiplicities and
every dimension and spin, and list all admissible F and R matrices, including
vacuum channels. A scalar channel is still a 1×1 matrix, for example
`"matrix": [[1]]`. The loader validates complete coverage and creates callable
lookup functions; consistency checks never need to inspect the original lists.

F rows are `(e,alpha,beta)` with
`0 <= alpha < N(a,b,e)` and `0 <= beta < N(e,c,d)`.
F columns are `(f,mu,nu)` with
`0 <= mu < N(b,c,f)` and `0 <= nu < N(a,f,d)`.
Order each basis by the anyon order, then the first multiplicity index, then
the second. An F record is required whenever either basis is nonempty.

R(a,b;c) has **N(b,a,c) rows and N(a,b,c) columns**. This orientation follows
the reference paper. It matters when fusion multiplicities exceed one.

In table mode, loading checks shape and completeness, not the equations. For example, a
rectangular F matrix can be structurally valid input but will fail unitarity.
Duplicate records, duplicate JSON object keys, missing data, unknown labels,
inadmissible symbol records, and invalid shapes raise `DataError`.

See [the complete Fibonacci file](../examples/fibonacci.json) for a small
ready-to-load example, including its 2×2 F block.

## Scenario 2: explicit functions

For the toric code, the category functions are:

```json
"symbol_format": "functions",
"fusion_rules": {"expression": "1 if c == ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2) else 0"},
"quantum_dimensions": {"expression": "1"},
"topological_spins": {"expression": "(-1) ** (a[0] * a[1])"},
"F_symbols": {"expression": "1"},
"R_symbols": {"expression": "(-1) ** (a[0] * b[1])"}
```

Symbol expressions define scalar functions; the action returns an anyon label:

| JSON field | Expression arguments | Python accessor |
| --- | --- | --- |
| `fusion_rules` | `a,b,c` | `cat.N(a,b,c)` |
| `quantum_dimensions` | `a` | `cat.quantum_dimension(a)` |
| `topological_spins` | `a` | `cat.spin(a)` |
| `F_symbols` | `a,b,c,d,e,f,alpha,beta,mu,nu` | `cat.F(a,b,c,d,e,f,alpha=0,beta=0,mu=0,nu=0)` |
| `R_symbols` | `a,b,c,mu,nu` | `cat.R(a,b,c,mu=0,nu=0)` |
| `symmetry.U_symbols` | `g,a,b,c,mu,nu` | `sym.U(g,a,b,c,mu=0,nu=0)` |
| `symmetry.eta_symbols` | `a,g,h` | `sym.eta(a,g,h)` |
| `symmetry.action` | `g,a` | `sym.action(g,a)` |

The anyon arguments are the labels, so tuple components are accessed with
`a[0]`, `a[1]`, etc. Multiplicity indices are integers. An expression need not
mention unused arguments. In particular, eta expressions use `a` for the anyon,
although eta table records use the key `anyon`.

Before invoking F, R or U, the accessors check the fusion channel and
multiplicity indices. Forbidden channels return zero. Thus a constant F=1
only applies to admissible coefficients. It does **not** mean an identity
matrix in a space of dimension greater than one; write an index-dependent
expression when needed.

JSON expressions are parsed once into functions. Fusion is evaluated and
memoized while loading to enumerate admissible channels and construct bases.
Other functions are evaluated only when accessed or checked and are not expanded
into symbol tables at loading. The grammar supports arithmetic,
literal integer tuple indexing, comparisons, `and`/`or`/`not`, conditional
expressions (`x if condition else y`), bitwise `^`, `&`, `|`, and the named
mathematical functions `sqrt`, `exp`, `sin`, `cos`, `conj`, `abs`, `complex`.
Use `pi` and complex literals such as `1j`; for example `exp(4j*pi/5)`.
Tuple/list and string literals can be used in comparisons.

This is a bounded mathematical expression grammar, not arbitrary Python:
imports, attribute access, lambdas, comprehensions and unknown calls are
rejected. Expressions are limited to 4096 characters, 256 AST nodes and depth
32, with numerical magnitude and exponent limits. Fusion functions must return
an exact nonnegative integral real scalar: `0`, `1` and `1.0` are accepted,
while booleans, fractional values and non-real complex values are rejected. Dimension
functions must return finite real scalars; `check_fusion` checks positivity.
F, R, spins, U and eta must return finite complex scalars (real numbers are also
accepted). Syntax errors surface at loading; evaluation errors raise `DataError`
with the symbol and arguments.

For unrestricted mathematical definitions in **your own Python code**, use
`UMTC.from_functions(data, F=F, R=R, N=N,
quantum_dimension=quantum_dimension, spin=spin, U=U, eta=eta, action=action)`. All these keyword
arguments are optional. A supplied function replaces its corresponding field,
which may therefore be omitted from `data`; omitted or `None` arguments preserve
the existing field. A required field without any definition still raises
`DataError`. F and R must both be functions in the resulting data: to convert
a fully tabulated file, supply both F and R callbacks. Supplying only `N`, for
example, does not convert F/R tables into functions. Callables can also be
used for the action alone: `UMTC.from_functions(data, action=action)` preserves
the F/R mode, including tables. Callables can otherwise be
provided directly in the corresponding fields
of a dictionary passed to `UMTC.from_dict`; JSON files use expression objects.

In multiplicity-free categories the functions may have signatures
`F(a,b,c,d,e,f)`, `R(a,b,c)` and `U(g,a,b,c)`. For multiplicities, use
`F(a,b,c,d,e,f,alpha=0,beta=0,mu=0,nu=0)` and
`R(a,b,c,mu=0,nu=0)` / `U(g,a,b,c,mu=0,nu=0)`; keyword-only index parameters are
also accepted. The remaining signatures are `N(a,b,c)`,
`quantum_dimension(a)`, `spin(a)`, `eta(a,g,h)` and `action(g,a)`.
Native functions are trusted application code, should be deterministic, and
follow the same evaluation timing and scalar validation as expressions. See
[the Python example](../README.md#python-api) for definitions of these functions.

In either mode, all accessors in the table above are callable. The original
field names are also callable aliases: `cat.fusion_rules`,
`cat.quantum_dimensions`, `cat.topological_spins`, `cat.F_symbols`,
`cat.R_symbols`, `sym.U_symbols` and `sym.eta_symbols`. `F_matrix`, `R_matrix`
and `U_matrix` assemble matrices using those same scalar functions and the
basis order above. The mathematical checking code has no separate
table/function branches.

## Symmetry object

Specify a finite GAP group `G`, a homomorphism `rho: G -> Z2`, and an anyon
action. The toric-code example uses:

```json
"symmetry": {
  "group": {"gap": "AbelianGroup([2,2])", "generator_names": ["X", "T"]},
  "rho": {"generator_images": [0, 1]},
  "action": {"expression": "(a[1], a[0]) if g == 'X' or g == 'T' else a"},
  "U_symbols": {"expression": "(-1) ** (a[0] * b[1]) if g == 'X' or g == 'T' else 1"},
  "eta_symbols": {"expression": "(-1) ** (a[0] * a[1]) if (g == 'X' or g == 'T') and (h == 'X' or h == 'T') else 1"}
}
```

| Field | Contents |
| --- | --- |
| `group` | `{"gap": constructor_expression, "generator_names": [...]}`; names are optional |
| `rho` | `{"generator_images": [0,1,...]}` in GAP's `GeneratorsOfGroup(G)` order |
| `action` | `{"expression": ...}` defining `action(g,a)`; explicit image records also accepted |
| `U_symbols` | `{"g":g,"a":a,"b":b,"c":c,"matrix":[...]}` for every group element and admissible fusion triple, or a function expression object |
| `eta_symbols` | `{"anyon":a,"g":g,"h":h,"value":...}` for every anyon and ordered group pair, or a function expression object |

GAP must be available as `gap` on `PATH`. It constructs the group and uses
`GroupHomomorphismByImages` to validate the grading. Images must be integers
0 or 1, with one image per GAP generator; 1 denotes antiunitary elements.
The homomorphism need not be surjective. For example, the generator of
`CyclicGroup(3)` cannot map to 1, but can map to 0.

The group expression supports constructor calls `AbelianGroup`, `CyclicGroup`,
`DihedralGroup`, `SymmetricGroup`, `AlternatingGroup`, `SmallGroup`, `Group`
and `DirectProduct`, integer/list arguments and permutation cycles. GAP's
`IsPermGroup`/`IsPcGroup` constructor options are also supported. Arbitrary GAP
statements, assignments, files and process calls are rejected. Groups must be
finite and have order at most 256 by default. The Python API exposes
`GAPGroup.from_gap(..., executable="gap", timeout=30, max_order=256)` to set
the executable, time limit and order bound. Loading errors become `DataError`;
direct group API errors are `GAPError` (or `GAPUnavailableError`).
Each GAP process has a 256 MiB GAP workspace limit; permutation points in
construction expressions are limited to 1,000,000.

Elements use canonical string labels generated by breadth-first traversal in
GAP generator order, multiplying on the right. Identity is `"1"`; the first
word reaching each element supplies its label. For the example, `sym.G.elements`
is `("1", "X", "T", "X*T")`. Names default to `g1`, `g2`, etc.
Redundant generators can refer to the same canonical element; inspect
`G.generators` for their labels. Use `G.mul(g,h)`, `G.inverse(g)` and
`G.pow(g,n)` for group operations rather than concatenating words yourself.

The Python API also accepts a `GAPGroup` object as `group` and the object
returned by `G.homomorphism_to_z2([...])` as `rho`. A native homomorphism must
have that same group object as its source. The loaded symmetry exposes them as
`sym.G` and `sym.rho`; `sym.is_antiunitary(g)` tests `sym.rho(g) == 1`.

The action expression receives the canonical group label `g` and anyon label
`a`, and returns a known anyon label: a string, integer, or integer tuple/list.
For instance `"a"` defines the trivial action for any label type. Boolean,
floating-point and complex results are invalid labels. Evaluation is lazy;
`check_symmetry_action` checks permutations and the action law for every
element and anyon. Native `action(g,a)` callbacks are accepted through
`from_dict` or `from_functions(..., action=action)`. Supplying U, eta or action
overrides without a symmetry object raises `DataError`.

For compatibility, the legacy group format remains readable: `elements` is a
list of unique strings, `identity` names the unit, `multiplication_table` has
entry gh at row g and column h, and `antiunitary` lists the odd elements.
These four fields replace `group` and `rho`; mixing the formats is an error.
Legacy symmetries expose `sym.G=None` and a callable grading `sym.rho(g)`.
Their group and grading laws are checked by the consistency checkers.
In either group format, action tables are wrapped as functions; each element
has an image record in the anyon ordering:

```json
{"g": "X", "images": [[0,0], [1,0], [0,1], [1,1]]}
```

U matrices have shape N(a,b,c)×N(a,b,c). Their anyon arguments are the
**destination labels** of the symmetry action. Thus Eq. (19) uses
`U(g, act(g,a), act(g,b), act(g,c))`, while Eqs. (23) and (24) pull labels back
through `inverse(g)` where specified.

U and eta definitions are required for all their admissible arguments,
including identity entries. The checker uses
the normalization in Eq. (27): U is identity on vacuum vertices, U for the
identity group element is identity everywhere, and eta is one when its anyon is
vacuum or either group element is the identity. U matrices must be unitary and
eta values must have unit magnitude.

The action need not be faithful: distinct elements can induce the same anyon
permutation. For example, X and T in
[the toric-code file](../examples/toric_code.json) both exchange e and m,
but only T is antiunitary. GAP validates the group and grading at loading;
the consistency checks verify the anyon action and symbols as well.
