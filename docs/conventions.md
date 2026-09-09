# Symbol conventions and implemented equations

This package uses the splitting-space conventions in Section II of
[arXiv:2210.02444](https://arxiv.org/pdf/2210.02444). Symbols are stored as
complex numbers in a chosen orthonormal basis. All indices below start at zero.
Whether fusion, F/R, dimensions, spins or U/eta originate in explicit tables,
JSON expressions or Python functions does not change any convention or equation:
every check calls the same scalar interfaces, with matrices assembled from them
when needed.

## F and R

Write `F[a,b,c,d][(e,α,β),(f,μ,ν)]` for the coefficient converting
`((a b)_e c)_d` to `(a (b c)_f)_d`. The multiplicity indices label the
two splitting vertices on each side. Rows describe the left tree and columns
the right tree.

The paper's `R[a,b,c]` has a row in the **ba** splitting space and a column
in the **ab** splitting space. Its vertex-basis transformation is
`R' = Γ_ba R Γ_ab†`. Consequently a tree map represented with a *source row*
that exchanges `a b -> b a` uses `B(a,b;c)=R[b,a,c]`.

F and R are unitary in a UMTC. F inverse is therefore F adjoint; the hexagon
checker includes unitarity checks before accepting this replacement.

## Pentagon

Fix external anyons a,b,c,d and total t. The initial basis is
`(((a b)_e c)_f d)_t` with multiplicities α,β,γ. The final basis is
`(a (b (c d)_h)_i)_t` with multiplicities δ,ε,ζ. For every pair of bases,
the code compares the two paths:

```text
sum_λ F[e,c,d,t][(f,β,γ),(h,δ,λ)]
      F[a,b,h,t][(e,α,λ),(i,ε,ζ)]

 = sum_(j,η,θ,κ) F[a,b,c,f][(e,α,β),(j,η,θ)]
                   F[a,j,d,t][(f,θ,γ),(i,κ,ζ)]
                   F[b,c,d,i][(j,η,κ),(h,δ,ε)] .
```

Sums run only over admissible channels and all their multiplicity indices.

## Both hexagons

Local braid maps preserve the spectator vertex and its index. With fixed total
charge, the two tree-map equalities are:

```text
B(a,bc) = F_abc† B_ab F_bac B_ac F_bca†
B(ab,c) = F_abc  B_bc F_acb† B_ac F_cab
```

Products are ordinary left-to-right row-source matrix products, with the ordered
leaves and fusion basis appropriate to each intermediate tree. Each equality is
checked entry by entry. This formulation includes the fusion-multiplicity sums.

## Symmetry

Let A_g(a) be the permutation, ρ(g) the homomorphism G → Z₂ recording
antiunitary parity, and σ_g(z) equal z
for unitary g or its complex conjugate for antiunitary g. The stored
`U(g,a,b,c)` uses destination labels, as in the paper. `eta(a,g,h)` is a phase.

GAP constructs the finite group and validates the supplied generator images
of ρ. The checker also verifies the exported group laws,
ρ(gh)=ρ(g)+ρ(h) modulo two, and A_g A_h=A_gh. The action must preserve the
vacuum, fusion multiplicities and dimensions, and satisfy
θ_(A_g a)=σ_g(θ_a). In the Python API these are `sym.rho(g)` and
`sym.action(g,a)`, respectively; the paper denotes the anyon action by ρ_g.

For Eq. (19), define L_g and Q_g as the two-vertex U actions on the left and
right fusion trees, respectively. Reorder the transformed F matrix by mapping
the original intermediate labels e,f through A_g. Then check

```text
L_g F_transformed = σ_g(F) Q_g

U(g,A_g b,A_g a,A_g c) R(A_g a,A_g b,A_g c)
    = σ_g(R(a,b,c)) U(g,A_g a,A_g b,A_g c).
```

Cross-multiplication avoids numerical inversion. U unitarity is also checked.

For Eq. (23), set
`κ = eta(a,g,h) eta(b,g,h) / eta(c,g,h)`.
The rearranged matrix equation is

```text
U(gh,a,b,c) = κ σ_g(U(h,A_(g^-1) a,A_(g^-1) b,A_(g^-1) c)) U(g,a,b,c).
```

The order of these matrices matters. The implementation also multiplies through
by `eta(c,g,h)` to avoid division.

For Eq. (24), check the twisted cocycle equation for every a,g,h,k:

```text
eta(a,g,h) eta(a,gh,k)
    = eta(a,g,hk) σ_g(eta(A_(g^-1) a,h,k)).
```

Normalization is Eq. (27), together with the identity symmetry functor:

```text
eta(vacuum,g,h) = eta(a,identity,g) = eta(a,g,identity) = 1
U(g,vacuum,a,a) = U(g,a,vacuum,a) = identity matrix
U(identity,a,b,c) = identity matrix.
```

## Dimensions, twists and modularity

The additional checks use positive d_a, d_1=1, and
`d_a d_b = sum_c N(a,b,c) d_c`. Twists must have unit magnitude, θ_1=1,
and θ_(dual a)=θ_a. The trace and balancing identities are

```text
d_a θ_a = sum_c d_c Tr(R(a,a,c))
θ_a θ_b R(b,a,c) R(a,b,c) = θ_c I.
```

With `D=sqrt(sum_a d_a²)`, the normalized matrix is

```text
S[a,b] = sum_c N(a,b,c) θ_c d_c / (D θ_a θ_b).
```

Modularity is tested through unitarity of S, in addition to the other conditions
when `check_all` is used. Calling a focused checker alone only establishes its
documented conditions.

## Numerical interpretation

Scalar comparisons use
`abs(lhs-rhs) <= max(atol,rtol*max(abs(lhs),abs(rhs)))`.
Nonfinite results fail. Discrete structural conditions are exact regardless of
the numerical tolerances. Matrix residuals are checked entrywise. A report's
`max_error` is the largest absolute scalar residual, including passing equations.
