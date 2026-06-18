# MFML Midsem Scoring Guide

**Course:** AIMLC ZC416 — Mathematical Foundations for Machine Learning
**Exam:** Mid-Semester (Closed Book, 90 min, 30% weight)
**Syllabus:** Sessions 1–8 (Weeks 1–8)
**Style:** Unicode-math, universally rendered Markdown.

---

## Syllabus map (sessions 1–8)

| Session | Topic | Companion |
|---|---|---|
| 1 | Solution of linear systems — equations, matrices, echelon form | Lecture 1 |
| 2 | Vector spaces, subspaces, linear independence, basis, rank | Lecture 2 |
| 3 | Analytic geometry — norms, inner products, orthogonality, orthonormal basis (Gram-Schmidt) | Lecture 3 |
| 4 | Matrix decomposition I — determinant, trace, eigenvalues, eigenvectors, Cholesky | Lecture 4 |
| 5 | Matrix decomposition II — eigen decomposition, diagonalization, SVD, matrix approximation | Lecture 5 |
| 6 | Vector calculus I — univariate differentiation, partial derivatives, gradients | Lecture 6 |
| 7 | Vector calculus II — gradients of matrices, useful identities, backprop, autodiff | Lecture 7 (Hessian) |
| 8 | Vector calculus III — higher-order derivatives, linearization, multivariate Taylor, unconstrained optima | Lecture 8 |

**Repeat structure of every recent midsem (2024 Reg/Makeup, Dec 2025, Jan 2026):**
- Q1: linear systems / echelon form / determinant / rank
- Q2: vector spaces / subspaces / basis / linear independence
- Q3 or Q4: SVD / diagonalization
- Q3 or Q4: inner product on Rⁿ via SPD matrix (distance, angle, orthogonality)
- Q5: gradient → critical points → Hessian classification (sometimes plus chain rule)
- Q6: Taylor polynomial of degree 2 or 3

---

## Hour-by-hour plan (10 hours)

| Hour | Block | Outcome |
|---|---|---|
| 1 | Echelon form + Gauss elimination + rank | Q1 fully scorable |
| 2 | Determinant tracking through row ops + inverse via [A|I] | Q1 (b)/(c) |
| 3 | Subspace test + null space + basis + dimension | Q2 |
| 4 | Linear independence + linear combination + span | Q2 |
| 5 | Inner product via SPD matrix + Sylvester + completing the square | Q3/Q4 (inner product part) |
| 6 | Distance, angle, orthogonality with custom inner product | Q3/Q4 (inner product part) |
| 7 | Eigenvalues, eigenvectors, diagonalisability (AM vs GM) | Q3/Q4 (eigen part) |
| 8 | SVD: AᵀA route, σ, V, U, padding Σ | Q3/Q4 (SVD) |
| 9 | Gradient → critical points → Hessian → classification + chain rule | Q5 |
| 10 | Taylor polynomial (deg 2/3), linearisation, smallest/largest values | Q6 |

---

## Hour 1 — Echelon form, Gauss elimination, rank

### 1.1 Three legal row operations (REF preserves the solution set)

1. Swap two rows: Rᵢ ↔ Rⱼ.
2. Scale a row by a non-zero constant: Rᵢ → λ·Rᵢ, λ ≠ 0.
3. Replace Rᵢ by Rᵢ + λ·Rⱼ (add a multiple of another row).

A matrix is in **row-echelon form (REF)** when (i) every all-zero row is at the bottom, (ii) the pivot of each non-zero row is strictly to the right of the pivot of the row above. **Reduced REF (RREF)** additionally has each pivot equal to 1 and is the only non-zero entry in its column.

### 1.2 The rank from REF (the most-asked single number)

> rank(A) = number of non-zero rows in any REF of A = number of pivot columns

It is also (i) the dimension of the column space, (ii) the dimension of the row space, (iii) the maximum number of linearly independent columns (or rows).

### 1.3 The fate of A·x = b — three cases via rank

Let A be m × n. Form the augmented matrix [A | b].

| Condition | Conclusion |
|---|---|
| rank(A) < rank([A | b]) | **No solution** (inconsistent) |
| rank(A) = rank([A | b]) = n | **Unique solution** |
| rank(A) = rank([A | b]) < n | **Infinitely many solutions** with (n − rank) free variables |

### 1.4 Worked numerical (Dec 2025 Q1a — re-graded)

Solve A·x = b with A = [[1, 2, 3], [2, 4, 6], [1, −1, 0]], b = [6, 12, 5]ᵀ.

Augment and reduce:

```
[1   2  3 | 6 ]   R2 → R2 − 2R1   [1   2  3 |  6]
[2   4  6 |12 ]  ──────────────►  [0   0  0 |  0]
[1  −1  0 | 5 ]   R3 → R3 −  R1   [0  −3 −3 | −1]

Swap R2 ↔ R3 to push the zero row to the bottom:

[1   2   3 |  6]
[0  −3  −3 | −1]
[0   0   0 |  0]
```

rank(A) = rank([A|b]) = 2 < 3, so infinitely many solutions, 1 free variable.

Take x₃ = k. Then −3x₂ − 3k = −1 ⇒ x₂ = (1 − 3k)/3 = 1/3 − k. And x₁ + 2(1/3 − k) + 3k = 6 ⇒ x₁ = 16/3 − k.

**General solution:** x = (16/3, 1/3, 0)ᵀ + k·(−1, −1, 1)ᵀ, k ∈ ℝ.

### 1.5 Determinant of a 4 × 4 via REF (Dec 2025 Q1b)

Track row swaps and scalings; the determinant is the product of pivots times (−1)^(# swaps), divided by every scaling factor you applied.

For an upper-triangular matrix U: det(U) = u₁₁ · u₂₂ · ⋯ · uₙₙ.

Worked: B = [[5, 25, −64, 32], [0, −1, −3, 4], [0, 1, −2, 5], [0, 0, 5, −3]].

R3 → R3 + R2 gives row [0, 0, −5, 9]. R4 → R4 + R3 gives [0, 0, 0, 6]. The matrix is now upper triangular with diagonal (5, −1, −5, 6). No swaps, no scalings.

det(B) = 5 · (−1) · (−5) · 6 = **150**. rank(B) = 4 (four non-zero rows).

### 1.6 Inverse via [A | I] → [I | A⁻¹] (Jan 2026 Q1b)

For A = [[1, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]]:

After forward elimination (R2 → R2 − R1, R3 → R3 − R2, R4 → R4 − R3) the left half becomes upper triangular with diagonal 1,1,1,1; det(A) = 1.

Continuing with backward elimination (R3 → R3 − R4, R2 → R2 − R3, R1 → R1 − R2) reduces the left half to I. The right half collects the inverse:

A⁻¹ = [[4, −3, 2, −1], [−3, 3, −2, 1], [2, −2, 2, −1], [−1, 1, −1, 1]].

Solve A·x = b for b = (−2, 2, 1, −1)ᵀ:
x = A⁻¹ b = (−11, 9, −5, 2)ᵀ.

### 1.7 Existence/uniqueness with a parameter (2020 Q2a, Practice Q6)

For a system with parameter α (or k), compute det(A) as a polynomial in α; the system fails to have full pivots exactly at the roots. For each root, plug back into the augmented matrix and use REF to decide between *no solution* (a row 0 = c with c ≠ 0) and *infinitely many*.

Pattern check (Practice Q6): for k = 1 the system became 0 = −3 (no solution); for k = −2 the augmented row reduced to all zeros (infinitely many).

### 1.8 The five-line cheat for Q1

1. Augment, choose first non-zero pivot in column 1 (swap if needed).
2. Eliminate below using `R → R − (pivot ratio) · R_pivot`.
3. Move to the next column, find the next pivot below the previous, repeat.
4. If a row reads 0 = c (c ≠ 0) → no solution.
5. If every row is 0 = 0 below the last pivot, count free variables (columns without pivots); back-substitute with each free variable parameterised.

---

## Hour 2 — Determinant (cofactors), trace, identities

### 2.1 Cofactor expansion

For an n × n matrix A with entry a_{ij}, define the **minor** M_{ij} as the determinant of the (n−1)×(n−1) submatrix obtained by deleting row i and column j, and the **cofactor** C_{ij} = (−1)^(i+j) M_{ij}.

> det(A) = Σⱼ aᵢⱼ · Cᵢⱼ = Σᵢ aᵢⱼ · Cᵢⱼ for any chosen row i (or column j).

**Tip:** expand along the row or column with the most zeros — every zero kills an entire cofactor.

### 2.2 2 × 2 and 3 × 3 fast formulas

- 2 × 2: det[[a, b], [c, d]] = ad − bc.
- 3 × 3 (Sarrus or first-row cofactor): det = a·(ei − fh) − b·(di − fg) + c·(dh − eg).

### 2.3 Properties to lean on

- det(Aᵀ) = det(A).
- det(A·B) = det(A)·det(B).
- det(A⁻¹) = 1/det(A).
- det(λA) = λⁿ · det(A) for n × n A.
- For a triangular matrix det(A) = product of diagonal entries.
- Swapping two rows multiplies the determinant by (−1).
- Scaling a row by λ multiplies the determinant by λ.
- Adding a multiple of one row to another **does not change** det(A).

### 2.4 Trace

tr(A) = Σᵢ aᵢᵢ. Properties: tr(A + B) = tr(A) + tr(B), tr(αA) = α·tr(A), tr(In) = n, **cyclic** identity tr(AB) = tr(BA), tr(ABC) = tr(BCA) = tr(CAB) (rotate, do not reverse).

### 2.5 Sum/product of eigenvalues — a free check

For any n × n matrix A with eigenvalues λ₁, …, λₙ (counted with multiplicity):

> tr(A) = Σ λᵢ and det(A) = Π λᵢ

Use this constantly to catch arithmetic errors after computing eigenvalues.

### 2.6 Worked: 3 × 3 cofactor expansion

A = [[3, 1, 0], [−2, −4, 3], [5, 4, −2]]. Expand along row 1.

- C₁₁ = +det[[−4, 3], [4, −2]] = (8 − 12) = −4
- C₁₂ = −det[[−2, 3], [5, −2]] = −(4 − 15) = 11
- C₁₃ = +det[[−2, −4], [5, 4]] = (−8 + 20) = 12

det(A) = 3·(−4) + 1·(11) + 0·(12) = −1.

### 2.7 Determinant via REF (the *fastest* method on the exam)

Do Gaussian elimination, count row swaps s, do not scale rows (or compensate). Then det(A) = (−1)^s · (product of diagonal entries of the upper-triangular form).

---

## Hour 3 — Subspaces, null space, basis, dimension

### 3.1 The three-axiom subspace test

A subset U of a vector space V is a **subspace** iff:
1. **0 ∈ U** (contains the zero vector);
2. **closed under addition:** u, v ∈ U ⇒ u + v ∈ U;
3. **closed under scalar multiplication:** u ∈ U, c ∈ ℝ ⇒ c·u ∈ U.

**Quick disqualifiers:** any set defined by an equation with non-zero constant on the right (e.g. x + y + z = 1) fails (1). Any bounded set (unit square, ball) fails (3).

### 3.2 Two free subspaces from any matrix A (m × n)

- **Null space** N(A) = { x ∈ ℝⁿ : A·x = 0 } — a subspace of ℝⁿ.
- **Column space** C(A) = span(columns of A) — a subspace of ℝᵐ.

### 3.3 Span is always a subspace

For any vectors v₁, …, vₖ in V, span{v₁, …, vₖ} is automatically a subspace of V. So if a question defines V = span{…}, you may quote “span of vectors is a subspace” — full marks (this is exactly how Dec 2025 Q2b is graded).

### 3.4 Basis = spanning + linearly independent

A finite set B ⊂ V is a **basis** of V iff B spans V and B is linearly independent.

**Equivalent characterisations** (use whichever is easiest):
- smallest spanning set;
- largest linearly independent set;
- every vector in V has a *unique* representation as a linear combination of B.

### 3.5 Dimension

dim(V) = number of vectors in any basis (well-defined). Standard cases:
- dim(ℝⁿ) = n;
- dim(ℝ^{m×n}) = m·n;
- dim({0}) = 0.

### 3.6 Algorithm for finding a basis from a spanning set

1. Stack the candidate vectors as columns of A.
2. Reduce A to REF.
3. The **original** columns corresponding to **pivot columns** form a basis. Non-pivot columns are linear combinations of the pivots to their left.

### 3.7 Worked: null space of a singular matrix (Jan 2026 Q2)

A = [[1, −1, 1], [−2, 2, −2], [3, −3, 3]] has all rows proportional to [1, −1, 1]. RREF is [1, −1, 1; 0, 0, 0; 0, 0, 0]. Single equation x₁ − x₂ + x₃ = 0 ⇒ x₁ = x₂ − x₃. Two free variables x₂ = s, x₃ = t.

**Basis** of N(A): {(1, 1, 0)ᵀ, (−1, 0, 1)ᵀ}. **Dimension** = 2.

### 3.8 Rank-nullity theorem

For any m × n matrix A:

> rank(A) + nullity(A) = n   (number of columns)

Use to cross-check Q2 numerics in 5 seconds.

### 3.9 Tactic for "find a 1-dimensional subspace U inside V"

Pick any single non-zero vector u from a basis of V; then U = span{u} is automatically a 1-dim subspace of V (Jan 2026 Q2c).

---

## Hour 4 — Linear independence, linear combination, span

### 4.1 Linear independence in one line

Vectors v₁, …, vₖ are **linearly independent** iff the only solution of c₁v₁ + … + cₖvₖ = 0 is c₁ = ⋯ = cₖ = 0. Otherwise they are **linearly dependent**.

### 4.2 Three equivalent tests

| Test | When fastest |
|---|---|
| (a) Stack as columns of A; reduce to REF; *all columns must contain pivots* | Most general |
| (b) For square matrix: det(A) ≠ 0 ⇔ independent | When n equals the number of vectors and is small |
| (c) Show one vector is (or is not) a linear combination of the others | When you suspect dependence and want to exhibit the relation |

### 4.3 Pigeonhole consequence

> Any set of more than n vectors in ℝⁿ is automatically linearly dependent.

This is exactly Lecture 2's "independence-of-coordinates theorem." Useful free-marks observation.

### 4.4 Linear combination — the routine

To write b = c₁v₁ + ⋯ + cₖvₖ, augment [v₁ v₂ ⋯ vₖ | b], row-reduce, read off c₁, …, cₖ. If the reduced last row is 0 = (non-zero), then b is **not** in span{v₁, …, vₖ}.

### 4.5 Dec 2025 Q2 worked

Vectors a = (5, −3, 1)ᵀ, b = (6, 2, −3)ᵀ, c = (4, −8, 5)ᵀ.

Augment [a, b | c], reduce: gives c₁ = 2, c₂ = −1, so c = 2a − b. Hence 2a − b − c = 0 with non-trivial coefficients ⇒ {a, b, c} is **linearly dependent** in ℝ³.

For span{a, b}: row-reduce [a, b]; both columns have pivots ⇒ {a, b} is independent ⇒ basis of V = span{a, b}, dim V = 2.

To check (4, −8, 2)ᵀ ∈ V: augment, reduce, last row gives 0 = −27 ⇒ **not in V**.

### 4.6 Determinant test for linear dependence (Practice Q2)

Vectors are linearly dependent ⇔ det(matrix with vectors as columns) = 0.

A = [[1, −1, 0], [0, 1, 1], [1, 0, 1]] has det(A) = 0 ⇒ dependent. Find the relation by solving the homogeneous system.

### 4.7 Subspace-membership shortcut (geometric, two-vector spans in ℝ³)

To check whether a target vector lies in span{v₁, v₂} (a plane through 0 in ℝ³):
1. Compute the normal n = v₁ × v₂.
2. The plane equation is n·x = 0.
3. Plug the target into n·x; non-zero ⇒ off the plane.

This is a one-line alternative to row reduction (e.g. Dec 2025 Q2c, Saurabh's Method 2).

---

## Hour 5 — Inner products via SPD matrices

### 5.1 Definition

An **inner product** ⟨·, ·⟩ on a real vector space V is a map V × V → ℝ with three properties:
1. **Symmetry:** ⟨x, y⟩ = ⟨y, x⟩.
2. **Bilinearity (linearity in each argument):** ⟨αx + βy, z⟩ = α⟨x, z⟩ + β⟨y, z⟩.
3. **Positive-definiteness:** ⟨x, x⟩ ≥ 0 with equality iff x = 0.

### 5.2 The matrix recipe

For any matrix A ∈ ℝⁿˣⁿ define op(x, y) = xᵀAy. Then:

> op is an inner product on ℝⁿ ⇔ A is **symmetric and positive definite (SPD)**.

The standard dot product corresponds to A = I.

### 5.3 Three ways to check **A is SPD**

A is **symmetric** if A = Aᵀ (visual check: mirror entries across the diagonal match).

For positive-definiteness pick the easiest of:
1. **Sylvester's criterion:** every leading principal minor D₁, D₂, …, Dₙ is strictly positive. (D_k = det of top-left k × k block.)
2. **Eigenvalue check:** every eigenvalue of A is strictly positive.
3. **Quadratic-form / completing the square:** show xᵀAx is a sum of non-negative squares with equality iff x = 0.

For block-diagonal A: SPD iff each diagonal block is SPD (use the smaller blocks individually; Jan 2026 Q4a does this).

### 5.4 Common 2 × 2 and 3 × 3 SPD checks (closed-form)

For 2 × 2 A = [[a, b], [b, c]]: SPD iff a > 0 and ac − b² > 0.

For 3 × 3 A = [[a, b, c], [b, d, e], [c, e, f]]: SPD iff a > 0, ad − b² > 0, det(A) > 0.

### 5.5 Quadratic-form trick (Dec 2025 Q4)

A = [[3, −1], [−1, 1]]. Q(x) = 3x₁² + x₂² − 2x₁x₂ = 2x₁² + (x₁ − x₂)². Sum of two squares, never simultaneously zero unless x = 0 ⇒ A is SPD.

### 5.6 Dec 2024 Q2 — A = [[m, 1, 1], [1, m, 1], [1, 1, m]] for m ≥ 2

xᵀAx = m(x₁² + x₂² + x₃²) + 2(x₁x₂ + x₁x₃ + x₂x₃) = (x₁ + x₂)² + (x₁ + x₃)² + (x₂ + x₃)² + (m − 2)(x₁² + x₂² + x₃²).

Each square is ≥ 0, and (m − 2) ≥ 0 with equality at m = 2 (still positive when at least one xᵢ ≠ 0 because then at least one square term is positive). Hence A is SPD.

### 5.7 Block-diagonal example (Jan 2026 Q4a)

A = block-diag([[2, 1], [1, 3]], [[4, 1], [1, 2]]). Both blocks are SPD by Sylvester (D₁ = 2 > 0, D₂ = 5 > 0; D₁ = 4, D₂ = 7). So A is SPD.

---

## Hour 6 — Distance, angle, orthogonality with custom inner products

### 6.1 Definitions induced by ⟨·, ·⟩_A

Let ⟨x, y⟩_A = xᵀAy with A SPD. Then:

| Quantity | Formula |
|---|---|
| Norm of x | ‖x‖_A = √(xᵀAx) |
| Distance between x and y | d_A(x, y) = ‖x − y‖_A = √((x − y)ᵀA(x − y)) |
| Angle ω between x, y | cos ω = ⟨x, y⟩_A / (‖x‖_A · ‖y‖_A) |
| Orthogonal | ⟨x, y⟩_A = 0 |

**Key takeaway:** orthogonality is *relative to the chosen inner product*. Two vectors orthogonal under the dot product can fail to be orthogonal under ⟨·, ·⟩_A and vice versa.

### 6.2 Worked: distance via SPD matrix (Dec 2025 Q4b)

A = [[3, −1], [−1, 1]], u = (1, −1)ᵀ, v = (2, 1)ᵀ. Then u − v = (−1, −2)ᵀ.

A(u − v) = [[3, −1], [−1, 1]]·(−1, −2)ᵀ = (−3 + 2, 1 − 2)ᵀ = (−1, −1)ᵀ.

(u − v)ᵀ A (u − v) = (−1)(−1) + (−2)(−1) = 1 + 2 = 3. So d_A(u, v) = √3.

### 6.3 Worked: orthogonality flip (Dec 2025 Q4c)

x = (−1, 2)ᵀ, y = (2, 1)ᵀ. Standard dot product: x·y = −2 + 2 = 0 (orthogonal).

⟨x, y⟩_A: A·y = (5, −1)ᵀ. xᵀ(Ay) = (−1)(5) + 2(−1) = −7 ≠ 0 ⇒ **not** orthogonal in this geometry.

### 6.4 Worked: distance in 4-D with block SPD (Jan 2026 Q4b)

A = block-diag([[2, 1], [1, 3]], [[4, 1], [1, 2]]), u = (−2, 0, 1, −1)ᵀ, v = (−2, 2, 1, 0)ᵀ. w = u − v = (0, −2, 0, −1)ᵀ.

A·w = (−2, −6, −1, −2)ᵀ. wᵀA w = 0·(−2) + (−2)(−6) + 0·(−1) + (−1)(−2) = 12 + 2 = 14.

So d_A(u, v) = √14.

### 6.5 Norm under custom inner product (2024 Q2c)

For A = [[m, 1, 1], [1, m, 1], [1, 1, m]] and x = (1, −1, −2)ᵀ: Ax = (m − 3, −m − 1, −2m)ᵀ. xᵀAx = (m − 3) + (m + 1) − 2(−2m) = (m − 3) + (m + 1) + 4m = 6m − 2.

So ‖x‖_A = √(6m − 2). (For m = 2: √10.)

### 6.6 Gram–Schmidt orthonormalisation (standard dot product)

Given linearly independent v₁, …, vₖ, build orthonormal {q₁, …, qₖ} by:

```
u₁ = v₁
u₂ = v₂ − proj_{u₁}(v₂)              proj_u(v) = (⟨v, u⟩ / ⟨u, u⟩) · u
u₃ = v₃ − proj_{u₁}(v₃) − proj_{u₂}(v₃)
…
qᵢ = uᵢ / ‖uᵢ‖
```

**Worked (Practice Q12):** v₁ = (1, 1, 0), v₂ = (1, 0, 1), v₃ = (0, 1, 1).

- q₁ = (1, 1, 0)/√2.
- u₂ = v₂ − ⟨v₂, q₁⟩ q₁ = (1, 0, 1) − (1/√2)(1/√2)(1, 1, 0) = (1/2, −1/2, 1). Scale to (1, −1, 2). Normalise: q₂ = (1, −1, 2)/√6.
- u₃ = v₃ − ⟨v₃, q₁⟩ q₁ − ⟨v₃, q₂⟩ q₂. After arithmetic, u₃ ∝ (−1, 1, 1); q₃ = (−1, 1, 1)/√3.

### 6.7 Cholesky factorisation (a quick SPD by-product)

Every SPD matrix A factors uniquely as A = L Lᵀ with L lower triangular and ℓᵢᵢ > 0. Compute entries top-left to bottom-right:

ℓ₁₁ = √a₁₁, ℓⱼ₁ = aⱼ₁ / ℓ₁₁ for j > 1, then ℓ₂₂ = √(a₂₂ − ℓ₂₁²), and so on.

**2 × 2 example:** A = [[4, 2], [2, 3]] gives ℓ₁₁ = 2, ℓ₂₁ = 1, ℓ₂₂ = √(3 − 1) = √2. So L = [[2, 0], [1, √2]], and indeed L Lᵀ = A.

If at any point you have to take √(negative), the matrix was not positive-definite (Cholesky doubles as a positive-definiteness test).

---

## Hour 7 — Eigenvalues, eigenvectors, diagonalisability

### 7.1 Definitions

For a square matrix A ∈ ℝⁿˣⁿ, a non-zero vector v with A·v = λ·v is an **eigenvector** with **eigenvalue** λ. Eigenvalues are the roots of the **characteristic polynomial**:

> p_A(λ) = det(A − λI) = 0

For each root λ_i, the **eigenspace** E(λ_i) = N(A − λ_i I), and any non-zero element of E(λ_i) is an eigenvector for λ_i.

### 7.2 Algebraic vs geometric multiplicity

- **Algebraic multiplicity** AM(λ) = multiplicity of λ as a root of p_A.
- **Geometric multiplicity** GM(λ) = dim(N(A − λ·I)) = number of independent eigenvectors for λ.
- Always 1 ≤ GM(λ) ≤ AM(λ).

> A is **diagonalisable** ⇔ for every eigenvalue, GM(λ) = AM(λ) ⇔ the eigenvectors form a basis of ℝⁿ.

If A is diagonalisable, then A = P·D·P⁻¹ with D = diag(λ₁, …, λₙ) and P = [v₁ … vₙ].

### 7.3 Spectral theorem

Every **symmetric** real matrix A is orthogonally diagonalisable:

> A = Q·D·Qᵀ with Qᵀ·Q = I and D = diag(λ₁, …, λₙ).

Eigenvectors of distinct eigenvalues of a symmetric matrix are automatically orthogonal.

### 7.4 Worked: 2 × 2 eigenvalues (Dec 2025 Q4 sub-question; reused everywhere)

A = [[3, −1], [−1, 1]]. Char poly: det([[3 − λ, −1], [−1, 1 − λ]]) = (3 − λ)(1 − λ) − 1 = λ² − 4λ + 2 = 0. Eigenvalues λ = 2 ± √2.

Cross-check: tr(A) = 4 = (2 + √2) + (2 − √2). det(A) = 2 = (2 + √2)(2 − √2) = 4 − 2. ✓

Eigenvector for λ₁ = 2 + √2: solve (A − λ₁ I) v = 0; one row gives (1 − √2)v₁ − v₂ = 0, so v₁ = (1, 1 − √2)ᵀ. Similarly λ₂ = 2 − √2 → v₂ = (1, 1 + √2)ᵀ.

### 7.5 Diagonalisability checklist on a 3 × 3

1. Compute p_A(λ) = det(A − λI) — expand cofactors, factor.
2. Find each λ; if all real and distinct ⇒ diagonalisable, AM = GM = 1 each.
3. If a λ has AM ≥ 2, compute rank(A − λI); GM = n − rank. If GM < AM ⇒ **not** diagonalisable.

### 7.6 2024 Q3 worked (typical 3 × 3 with one repeated root)

A = [[2, 0, 0], [1, 2, 1], [−1, 0, 1]]. Char poly = (2 − λ)·det([[2 − λ, 1], [0, 1 − λ]]) (expand first row) = (2 − λ)(2 − λ)(1 − λ) = 0. Eigenvalues: λ = 2 (AM = 2) and λ = 1 (AM = 1).

For λ = 2: A − 2I = [[0, 0, 0], [1, 0, 1], [−1, 0, −1]] reduces to one independent row ⇒ rank 1 ⇒ GM = 3 − 1 = 2 = AM. Eigenvectors: e₂ = (0, 1, 0)ᵀ and (1, 0, −1)ᵀ.

For λ = 1: A − I = [[1, 0, 0], [1, 1, 1], [−1, 0, 0]] ⇒ rank 2 ⇒ GM = 1 = AM. Eigenvector (0, 1, −1)ᵀ.

A is diagonalisable: P = [(0, 1, 0)ᵀ, (1, 0, −1)ᵀ, (0, 1, −1)ᵀ], D = diag(2, 2, 1).

### 7.7 Eigenvalue tricks worth memorising

- A and Aᵀ have the same eigenvalues.
- Eigenvalues of A² are λ² (eigenvectors unchanged); of A⁻¹ are 1/λ (when A invertible).
- Eigenvalues of A + cI are λ + c.
- Triangular matrix → eigenvalues are the diagonal entries.
- A is invertible ⇔ 0 is **not** an eigenvalue.

### 7.8 Powers of a matrix via diagonalisation

If A = P·D·P⁻¹ with D = diag(λ₁, …, λₙ), then for any positive integer k:

> Aᵏ = P·Dᵏ·P⁻¹, where Dᵏ = diag(λ₁ᵏ, …, λₙᵏ).

Same trick handles polynomial(A) and even matrix exponentials.

---

## Hour 8 — SVD: AᵀA route, Σ, V, U, padding

### 8.1 Statement

Every A ∈ ℝᵐˣⁿ has a **singular value decomposition**:

> A = U Σ Vᵀ

with U ∈ ℝᵐˣᵐ orthogonal, V ∈ ℝⁿˣⁿ orthogonal, and Σ ∈ ℝᵐˣⁿ "diagonal" with non-negative entries σ₁ ≥ σ₂ ≥ ⋯ ≥ σ_r > 0 (where r = rank A) and zeros elsewhere.

### 8.2 The standard recipe (works on any midsem question)

1. **Form AᵀA** (n × n, symmetric, positive semidefinite).
2. Find eigenvalues μ₁ ≥ μ₂ ≥ ⋯ ≥ μ_n ≥ 0 of AᵀA. The **singular values** are σᵢ = √μᵢ.
3. Find an **orthonormal eigenbasis** {v₁, …, v_n} of AᵀA — the columns of V.
4. For each non-zero σᵢ set uᵢ = (1/σᵢ)·A·vᵢ — these are the first r columns of U.
5. Pad U with any orthonormal basis of N(Aᵀ) (the left null space) to make U square.
6. Σ has σᵢ on the diagonal in the same order as the columns of V.

(Equivalent recipe: diagonalise AAᵀ for U, derive V from V = (1/σ)·Aᵀ·U.)

### 8.3 Pitfalls and shortcuts

- **Always order singular values decreasingly**; the order of v_i and u_i must match.
- **Σ is m × n** — pad with the right number of zero rows or columns to match A.
- For **square symmetric** A: if A has eigendecomposition Q·D·Qᵀ then SVD has σᵢ = |λᵢ| and the sign is absorbed into U (flip the corresponding column of U if λᵢ < 0).
- For a **rank-1** A = u·vᵀ: σ₁ = ‖u‖·‖v‖, all other singular values are 0.
- The **Frobenius norm** ‖A‖_F = √(Σ σᵢ²) = √(Σᵢⱼ aᵢⱼ²) — fastest cross-check.
- The **2-norm** (operator norm) ‖A‖₂ = σ₁.

### 8.4 Worked: SVD of a 2 × 2 matrix (Practice Q8/Q23 family)

A = [[3, 0], [0, −2]]. Already diagonal: singular values 3, 2 (note absolute value!). V = I, Σ = diag(3, 2), U = [[1, 0], [0, −1]] (because A·v₂ = (0, −2)ᵀ ⇒ u₂ = (0, −1)ᵀ to make σ₂ positive).

### 8.5 Worked: rank-1 example

A = [[1, 1], [2, 2]] = (1, 2)ᵀ · (1, 1). σ₁ = √(1² + 2²) · √(1² + 1²) = √5 · √2 = √10.

AᵀA = [[5, 5], [5, 5]] has eigenvalues 10, 0; eigenvectors v₁ = (1, 1)/√2, v₂ = (1, −1)/√2. u₁ = (1/√10)·A·v₁ = (1/√10)·(2/√2, 4/√2)ᵀ = (1/√5, 2/√5)ᵀ. Pad u₂ ⊥ u₁: u₂ = (2/√5, −1/√5)ᵀ. Σ = diag(√10, 0).

### 8.6 Worked: 2 × 3 case (Dec 2025 Q3, Jan 2026 Q3 family)

A = [[1, 2, 0], [0, 0, 1]]. AᵀA = [[1, 2, 0], [2, 4, 0], [0, 0, 1]]. Eigenvalues: 5, 1, 0. σ = (√5, 1) with V columns the unit eigenvectors v₁ = (1, 2, 0)ᵀ/√5, v₂ = (0, 0, 1)ᵀ, v₃ = (2, −1, 0)ᵀ/√5 (kernel direction).

u₁ = (1/√5)·A·v₁ = (1/√5)·(5/√5, 0)ᵀ = (1, 0)ᵀ. u₂ = (1/1)·A·v₂ = (0, 1)ᵀ. So U = I₂. Σ = [[√5, 0, 0], [0, 1, 0]].

Check: U Σ Vᵀ rebuilds A ✓.

### 8.7 Low-rank approximation (Eckart–Young)

Truncating SVD to the top k singular values gives the closest rank-k matrix in both 2-norm and Frobenius norm. Error ‖A − A_k‖_F = √(σ_{k+1}² + ⋯ + σ_r²).

### 8.8 Eigen vs SVD — when each applies

| Property | Eigendecomposition | SVD |
|---|---|---|
| Matrix shape | square only | any m × n |
| Diagonalisable | not always | **always** |
| Eigenvectors orthonormal | only if A symmetric | columns of U, V are always orthonormal |
| Useful for | A^k, dynamics, linear ODEs | rank, low-rank approx, pseudo-inverse, PCA |

---

## Hour 9 — Gradients, critical points, Hessian classification, chain rule

### 9.1 Gradient and Jacobian

For f: ℝⁿ → ℝ, the **gradient** is

> ∇f(x) = (∂f/∂x₁, …, ∂f/∂xₙ)ᵀ

For F: ℝⁿ → ℝᵐ, the **Jacobian** is the m × n matrix J_F(x) with (i, j)-entry ∂Fᵢ/∂xⱼ. (Numerator-layout convention used in Deisenroth/Bishop and the MFML companion.)

### 9.2 Hessian

For f: ℝⁿ → ℝ twice differentiable, the **Hessian** is

> H_f(x)_{ij} = ∂²f/(∂xᵢ ∂xⱼ)

By Schwarz/Clairaut H_f is **symmetric** when f is C².

### 9.3 The four useful identities (memorise)

| f(x) | ∇f(x) | H_f(x) |
|---|---|---|
| aᵀx | a | 0 |
| xᵀA x (general A) | (A + Aᵀ) x | A + Aᵀ |
| xᵀA x (symmetric A) | 2 A x | 2 A |
| ½ ‖x − a‖² | x − a | I |

For matrix-by-matrix derivatives (rare on midsem): ∂(tr(AX))/∂X = Aᵀ, ∂(tr(XᵀAX))/∂X = (A + Aᵀ)X.

### 9.4 The unconstrained-optimum recipe (Q5 always)

1. **Critical points:** solve ∇f(x) = 0.
2. **Classify each** with the Hessian H_f(x*):
   - H_f(x*) **positive definite** ⇒ strict local **minimum**;
   - H_f(x*) **negative definite** ⇒ strict local **maximum**;
   - H_f(x*) **indefinite** (mixed-sign eigenvalues) ⇒ **saddle point**;
   - H_f(x*) **semi-definite** ⇒ test inconclusive (use higher-order terms or direct comparison).

For 2 × 2 H = [[a, b], [b, c]] there is the convenient classifier:
- a > 0 and ac − b² > 0 ⇒ local min;
- a < 0 and ac − b² > 0 ⇒ local max;
- ac − b² < 0 ⇒ saddle;
- ac − b² = 0 ⇒ inconclusive.

### 9.5 Worked: Dec 2025 Q5 (typical)

f(x, y) = x³ − 3xy + y³. ∇f = (3x² − 3y, 3y² − 3x). Setting both to zero: y = x² and x = y² ⇒ x = x⁴ ⇒ x(x³ − 1) = 0. Critical points (0, 0) and (1, 1).

H_f = [[6x, −3], [−3, 6y]].

At (0, 0): H = [[0, −3], [−3, 0]], det = −9 < 0 ⇒ saddle.
At (1, 1): H = [[6, −3], [−3, 6]], a = 6 > 0, det = 27 > 0 ⇒ local min. f(1, 1) = 1 − 3 + 1 = −1.

### 9.6 Chain rule for f∘g where g: ℝᵐ → ℝⁿ, f: ℝⁿ → ℝ

> ∇(f ∘ g)(x) = J_g(x)ᵀ · ∇f(g(x))   (column gradient)

Or in Jacobian form: J_{f∘g}(x) = J_f(g(x)) · J_g(x).

For composed scalar functions y = f(u(x)): dy/dx = f'(u) · u'(x). Multivariate version: if z = f(x, y) and x = x(t), y = y(t), then dz/dt = (∂f/∂x)·(dx/dt) + (∂f/∂y)·(dy/dt).

### 9.7 Backpropagation in one slide

For a feed-forward computation y = f_L(f_{L−1}(…f_1(x)…)):

> ∂y/∂x = J_{f_L} · J_{f_{L−1}} · … · J_{f_1}

Compute right-to-left for forward-mode (Jacobian-vector products) or left-to-right for reverse-mode (vector-Jacobian products, which is "back-propagation"). Reverse mode is cheaper when the output dimension is smaller than the input dimension — which is the standard ML case.

### 9.8 Common bug-checks

- Verify ∇f(x*) = 0 by **substituting back** before classifying.
- For symmetric A, eigenvalues are real ⇒ definiteness is well-defined.
- A 3 × 3 Hessian is positive definite iff the three Sylvester minors are > 0; negative definite iff signs alternate −, +, − starting from D₁ < 0.

---

## Hour 10 — Taylor polynomial, linearisation, Q6

### 10.1 Univariate Taylor

f(x₀ + h) = f(x₀) + f'(x₀)·h + (f''(x₀)/2!)·h² + (f'''(x₀)/3!)·h³ + …

Truncated at degree n is the **n-th Taylor polynomial** T_n(x₀ + h).

### 10.2 Multivariate (vector x₀, displacement h)

T₁(f; x₀)(h) = f(x₀) + ∇f(x₀)ᵀ · h   (linearisation / tangent plane)

T₂(f; x₀)(h) = T₁ + ½ · hᵀ · H_f(x₀) · h

T₃(f; x₀)(h) = T₂ + (1/6) · Σ ∂³f/(∂xᵢ∂xⱼ∂xₖ) · hᵢ hⱼ hₖ summed over all triples.

In practice on the midsem the question lists ≤ 3 variables, so the cubic term is short.

### 10.3 Worked: Dec 2025 Q6a (linearisation of multi-variable f)

f(x, y, z) = x²·y + xz at (1, 2, 3). ∇f = (2xy + z, x², x). At (1, 2, 3): ∇f = (4 + 3, 1, 1) = (7, 1, 1). f(1, 2, 3) = 2 + 3 = 5.

T₁(h₁, h₂, h₃) = 5 + 7·(x − 1) + 1·(y − 2) + 1·(z − 3).

### 10.4 Worked: 2-D Taylor of degree 2 (recurring template)

f(x, y) = eˣ · cos y at (0, 0). f(0, 0) = 1. f_x = eˣ cos y → 1, f_y = −eˣ sin y → 0. f_{xx} = 1, f_{yy} = −1, f_{xy} = 0 (at origin).

T₂(x, y) = 1 + x + ½(x² − y²).

### 10.5 Smallest/largest values of T_n on a region

- Compute the gradient of T_n.
- Solve ∇T_n = 0 inside the region; classify with Hessian of T_n.
- On the **boundary** of a square or disc, parametrise and reduce to 1-D extremum problem.
- Compare values at all candidates.

### 10.6 Linearisation as best linear approximation

Given f and a point x₀, the linearisation L(x) = f(x₀) + ∇f(x₀)ᵀ(x − x₀) is the unique affine function tangent to f at x₀; for ‖x − x₀‖ small, |f(x) − L(x)| = O(‖x − x₀‖²). This is exactly Newton's method's linearisation and the basis of "delta-method" arguments in ML.

### 10.7 Final Q6 sanity checklist

- Did I evaluate every partial at the given x₀, not at a generic x?
- Did I divide by the right factorial? 1! for first-order, 2! for second, 3! for third.
- Are all symmetric mixed partials counted with the correct multiplicity (½ for the H term, 1/6 with all 3-tuples in the cubic)?
- Does T_n(x₀) = f(x₀)? Quick zero-displacement check.

---

## Final 30-minute "stand by the door" sweep

- Q1: write rank-augmented condition, then row-reduce. Note swaps for determinant. Free variables → infinite solutions.
- Q2: subspace = three axioms (or "span of vectors"). For independence stack as columns and reduce; for membership augment with target.
- Q3 (eigen/SVD): for symmetric A, just diagonalise (Q D Qᵀ). For SVD of m × n, always go through AᵀA.
- Q4 (inner product): write down xᵀAy explicitly. Distance = √((u−v)ᵀA(u−v)). Orthogonal in ⟨·, ·⟩_A means xᵀAy = 0.
- Q5 (gradient): set ∇f = 0; for each critical point write H_f(x*) and use Sylvester (or 2 × 2 a > 0 / det > 0 rule) to classify.
- Q6 (Taylor): expand around x₀, plug into T_n formula, simplify.
