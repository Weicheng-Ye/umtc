# Example category catalog

These JSON files contain the UMTC data and its intrinsic topological symmetry,
with a reference choice of U and eta when the intrinsic group is nontrivial.
They cover the category families studied in
[Ye and Zou, arXiv:2309.15118v3](https://arxiv.org/html/2309.15118v3) and
[Hao et al., arXiv:2608.14180v1](https://arxiv.org/html/2608.14180v1).
The infinite families are represented by the parameter choices below.

| Category family | JSON files | Intrinsic symmetry included |
| --- | --- | --- |
| U(1)₂N, N=1,…,5 | [u1_2](u1_2.json), [u1_4](u1_4.json), [u1_6](u1_6.json), [u1_8](u1_8.json), [u1_10](u1_10.json) | Trivial at N=1; unitary Z₂ charge conjugation otherwise |
| Ising⁽ν⁾, all odd ν modulo 16 | [ν=1](ising_nu1.json), [ν=3](ising_nu3.json), [ν=5](ising_nu5.json), [ν=7](ising_nu7.json), [ν=9](ising_nu9.json), [ν=11](ising_nu11.json), [ν=13](ising_nu13.json), [ν=15](ising_nu15.json) | Trivial |
| Toric code (Z₂ gauge theory) | [toric_code](toric_code.json), [z2_gauge](z2_gauge.json) | Z₂×Z₂ᵀ |
| Zₙ gauge theory, n=3,4 | [z3_gauge](z3_gauge.json), [z4_gauge](z4_gauge.json) | Dihedral group of order eight |
| U(1)₂N×U(1)₋₂N, N=1,2 | [double_semion](double_semion.json), [u1_4_x_u1_minus4](u1_4_x_u1_minus4.json) | Z₂ᵀ at N=1; dihedral group of order eight at N=2 |
| SU(2)ₖ, k=1,2,3,4,6,−3 | [k=1](su2_k1.json), [k=2](su2_k2.json), [k=3](su2_k3.json), [k=4](su2_k4.json), [k=6](su2_k6.json), [k=−3](su2_k_minus3.json) | Unitary Z₂ at k=6; trivial for the other shipped levels |

[Fibonacci](fibonacci.json) is also included. There are 26 files in total.

`symmetry: null` means no nontrivial intrinsic group is
attached. For a nontrivial group, `rho(g)` denotes antiunitary parity, while
`action(g,a)` gives the anyon permutation.

## Labels and conventions

The Abelian families use integer tuples and compact functions for fusion,
F, R, quantum dimensions, spins, action, U and eta. U(1) uses a one-component
tuple `(a,)`; gauge and doubled theories use two components.

Ising uses `["1", "sigma", "psi"]` and complete symbol tables. Its 2×2 F
block has intermediate channels `["1", "psi"]`. All eight values of ν are
included because the Frobenius–Schur indicator and braiding phases distinguish
them, even though their fusion rules coincide.

SU(2) uses listed integer labels `a=2j`, so the physical spin label is `j=a/2`.
F and R are complete tables in the paper's quantum 6j gauge. At k=6 the
intrinsic generator sends odd labels `a` to `6-a` and fixes even labels;
U and eta use the same destination-label convention as the package. The k=−3
file reverses chirality by conjugating R and the spins.

Some entries describe equivalent UMTCs in different labels or gauges:
SU(2)₁ is the semion category U(1)₂, SU(2)₂ is Ising⁽³⁾, and z2_gauge is the
toric code. In particular, the new gauge-theory files follow the first paper's
`R(a,b)=exp(2*pi*i*a_m*b_e/N)` convention; `toric_code.json` uses
`(-1)^(a_e*b_m)`. Both gauges satisfy the package's equations.

## Source normalization correction

The first paper's printed Eqs. (61)–(62) use `2*pi/N` for the doubled theory.
At N=1 this gives trivial braiding, inconsistent with the stated double-semion
category. The examples use the tensor product of the paper's U(1) data in
Eqs. (10)–(11) and their complex conjugates instead. Writing `L=2N`, this gives

```text
F(a,b,c) = (-1)^[a_s floor((b_s+c_s)/L) - a_bar floor((b_bar+c_bar)/L)]
R(a,b)   = exp(i*pi*(a_s*b_s-a_bar*b_bar)/L)
theta(a) = exp(i*pi*(a_s^2-a_bar^2)/L).
```

In particular, the double-semion spins are `i` and `-i`. For the Zₙ eta-symbols,
the examples use main-text Eq. (59), with the exponent `a_e*a_m`; Appendix
Eq. (166) prints an extraneous, undefined `b` label.

## Load, check and regenerate

From the project directory, with the package installed:

```python
from umtc import load_json

category = load_json("examples/su2_k6.json")
assert category.check_all()
assert category.symmetry.action("X", 1) == 5
assert category.symmetry.rho("X") == 0
```

The checks cover fusion, pentagon, both hexagons, unitarity, ribbon identities,
modularity, and the intrinsic action/U/eta consistency equations. Nontrivial
intrinsic groups require GAP on `PATH`.

```sh
umtc-check examples/z3_gauge.json
umtc-check examples/su2_k6.json --json
python scripts/generate_examples.py
python -m unittest discover -s tests -v
```

The generator modules expose `u1(level)`, `zn_gauge(n)`, `doubled_u1(level)`,
`ising(nu)` and `su2(k)`. The Abelian generators deliberately restrict parameters
to the ranges above, where the full intrinsic symmetry groups are implemented.
For example, U(1)₁₂ can have more symmetry than charge conjugation. The SU(2)
generator accepts other nonzero integer levels, though table sizes, numerical
conditioning and exhaustive checking costs grow with |k|.

The papers also classify microscopic lattice/spin symmetry enrichments. These
files supply the intrinsic categories and reference symmetry data; they do not
enumerate the many fractionalization classes for groups such as p4×SO(3), or
their anomaly classifications.
