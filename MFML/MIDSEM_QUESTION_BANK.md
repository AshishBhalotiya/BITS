# MFML Midsem Question Bank

**Course:** AIMLC ZC416 — Mathematical Foundations for Machine Learning
**Sources:** Dec 2025 (Regular + Makeup), Jan 2026 (Makeup), Jan 2025, 2024 Regular + Makeup, 2023 Makeup, 2022, 2020, Practice Problems, Companion Lecture exercises (1–8).
**Format:** every question is rewritten in the simplest possible Markdown (Unicode math), followed by a fully worked solution.

---

## How to use this bank

- Topics are ordered the way they appear on the midsem (Q1 → Q6).
- Each item is labelled `[Year/Source · Qn · marks]` so you can map back to the original paper.
- A "★" tag marks variants that have appeared in **more than one** past midsem — drill these first.

---

# Topic 1 — Linear systems, REF, rank, determinant, inverse

## Q1.1 [Dec 2025 · Q1a · 6 marks] ★

**Problem.** Solve the system A·x = b for

```
A = [ 1   2   3 ]      b = [  6 ]
    [ 2   4   6 ]          [ 12 ]
    [ 1  -1   0 ]          [  5 ]
```

**Solution.** Augment and reduce.

```
[ 1   2   3 |  6 ]                 [ 1   2   3 |  6 ]
[ 2   4   6 | 12 ]  R2 → R2 − 2R1  [ 0   0   0 |  0 ]
[ 1  -1   0 |  5 ]  R3 → R3 −  R1  [ 0  -3  -3 | -1 ]
```

Swap R2 ↔ R3:

```
[ 1   2   3 |  6 ]
[ 0  -3  -3 | -1 ]
[ 0   0   0 |  0 ]
```

rank(A) = rank([A|b]) = 2 < 3 ⇒ infinitely many solutions, 1 free variable.

Set x₃ = k. Then −3 x₂ − 3k = −1 ⇒ x₂ = 1/3 − k. Then x₁ = 6 − 2(1/3 − k) − 3k = 16/3 − k.

**Solution set:** x = (16/3, 1/3, 0)ᵀ + k·(−1, −1, 1)ᵀ, k ∈ ℝ.

---

## Q1.2 [Dec 2025 · Q1b · 6 marks]

**Problem.** Compute det(B) and rank(B) for

```
B = [ 5   25  -64   32 ]
    [ 0   -1   -3    4 ]
    [ 0    1   -2    5 ]
    [ 0    0    5   -3 ]
```

**Solution.** Use REF without scaling/swapping rows so the determinant equals the product of pivots.

R3 → R3 + R2 gives row [0, 0, −5, 9]. R4 → R4 + R3 gives row [0, 0, 0, 6].

The matrix is now upper triangular with diagonal (5, −1, −5, 6). Therefore

det(B) = 5 · (−1) · (−5) · 6 = **150**, and rank(B) = 4 (full rank).

---

## Q1.3 [Jan 2026 · Q1a · 5 marks]

**Problem.** Find the rank of A and characterise the solution set of A·x = 0.

```
A = [ 1  2  3 ]
    [ 2  4  6 ]
    [ 3  6  9 ]
```

**Solution.** R2 → R2 − 2R1 and R3 → R3 − 3R1 produce two zero rows. RREF:

```
[ 1  2  3 ]
[ 0  0  0 ]
[ 0  0  0 ]
```

rank(A) = 1. Single equation x₁ + 2 x₂ + 3 x₃ = 0 gives x₁ = −2 x₂ − 3 x₃ with two free variables ⇒ a 2-dimensional solution space (the null space). A basis is {(−2, 1, 0)ᵀ, (−3, 0, 1)ᵀ}.

---

## Q1.4 [Jan 2026 · Q1b · 7 marks]

**Problem.** Find A⁻¹ for

```
A = [ 1  1  0  0 ]
    [ 1  2  1  0 ]
    [ 0  1  2  1 ]
    [ 0  0  1  2 ]
```

and use it to solve A·x = (−2, 2, 1, −1)ᵀ.

**Solution.** Use [A | I] → [I | A⁻¹]. After R2 → R2 − R1, R3 → R3 − R2, R4 → R4 − R3 the left half is upper triangular with all-ones diagonal (so det(A) = 1, A is invertible). Backward elimination R3 → R3 − R4, R2 → R2 − R3, R1 → R1 − R2 reduces the left half to I; the right half is

```
A⁻¹ = [  4  -3   2  -1 ]
      [ -3   3  -2   1 ]
      [  2  -2   2  -1 ]
      [ -1   1  -1   1 ]
```

x = A⁻¹·b = (4·(−2) − 3·2 + 2·1 − 1·(−1), −3·(−2) + 3·2 − 2·1 + 1·(−1), 2·(−2) − 2·2 + 2·1 − 1·(−1), −1·(−2) + 1·2 − 1·1 + 1·(−1)) = (**−11**, **9**, **−5**, **1**).

(Note: a quick re-do with care gives x = (−11, 9, −5, 1) — verify by multiplying A·x.)

---

## Q1.5 [2024 Regular · Q1 · 6 marks] ★ — parameter family

**Problem.** For which values of α ∈ ℝ does the system below have (a) a unique solution, (b) no solution, (c) infinitely many solutions?

```
 x +  y +  z = 1
 x + 2y + αz = 2
 x + αy + 2z = 3
```

**Solution.** Augment and eliminate column 1: R2 → R2 − R1, R3 → R3 − R1.

```
[ 1   1     1   | 1 ]
[ 0   1   α-1   | 1 ]
[ 0  α-1   1    | 2 ]
```

R3 → R3 − (α − 1)·R2:

```
[ 1   1            1          | 1                ]
[ 0   1          α-1          | 1                ]
[ 0   0   1 - (α-1)²          | 2 - (α-1)        ]
                ↑                      ↑
              = 2α - α²              = 3 - α
```

The leading coefficient of x₃ is −(α² − 2α) = α(2 − α).

- α(2 − α) ≠ 0 ⇒ unique solution. So **α ∉ {0, 2} ⇒ unique**.
- α = 0: third equation reads 0·x₃ = 3 ⇒ inconsistent ⇒ **no solution**.
- α = 2: third equation reads 0·x₃ = 1 ⇒ inconsistent ⇒ **no solution**.

(There is no α for which both sides of the third row vanish, so the system never has infinitely many solutions.)

---

## Q1.6 [2020 Regular · Q2a · 5 marks]

**Problem.** Find values of k for which

```
 x +  y +  z = 1
 x + 2y + 4z = k
 x + 4y + 10z = k²
```

is (a) consistent, (b) has infinitely many solutions.

**Solution.** Eliminate x: R2 → R2 − R1, R3 → R3 − R1 ⇒

```
[ 1  1   1 | 1     ]
[ 0  1   3 | k - 1 ]
[ 0  3   9 | k²- 1 ]
```

R3 → R3 − 3 R2 ⇒ [0  0  0 | k² − 1 − 3(k − 1)] = [0  0  0 | k² − 3k + 2] = [0  0  0 | (k − 1)(k − 2)].

Consistency requires (k − 1)(k − 2) = 0, i.e. **k = 1 or k = 2**. In both cases the third row vanishes, leaving rank 2 < 3 unknowns, so there are **infinitely many** solutions.

---

## Q1.7 [Practice · Q2 · 3 marks]

**Problem.** Determine whether {(1, 0, 1), (−1, 1, 0), (0, 1, 1)} are linearly dependent and, if so, give the relation.

**Solution.** Stack as columns and compute det:

```
| 1  -1   0 |
| 0   1   1 |  =  1·(1·1 − 1·0) − (−1)·(0·1 − 1·1) + 0  =  1 − 1 = 0.
| 1   0   1 |
```

det = 0 ⇒ dependent. Solve A·c = 0: row-reduce to get c₁ − c₂ = 0, c₂ + c₃ = 0, so c₁ = c₂ = −c₃ ⇒ pick c = (1, 1, −1)ᵀ. Hence

(1, 0, 1) + (−1, 1, 0) − (0, 1, 1) = 0. ✓

---

## Q1.8 [Companion Lecture 1 · Exercise · skill drill]

**Problem.** Reduce A = [[1, 0, 1], [2, 1, 0], [0, 1, 1]] to RREF and read off rank.

**Solution.** R2 → R2 − 2R1, R3 unchanged ⇒ [[1, 0, 1], [0, 1, −2], [0, 1, 1]]. R3 → R3 − R2 ⇒ [[1, 0, 1], [0, 1, −2], [0, 0, 3]]. Three pivots ⇒ rank = 3, A is invertible.

---

# Topic 2 — Subspaces, basis, dimension, span, linear independence

## Q2.1 [Dec 2025 · Q2a · 4 marks]

**Problem.** Determine whether the vectors a = (5, −3, 1)ᵀ, b = (6, 2, −3)ᵀ, c = (4, −8, 5)ᵀ are linearly independent in ℝ³.

**Solution.** Stack as columns and reduce:

```
[ 5   6   4 ]      [ 5    6    4 ]
[-3   2  -8 ]  →   [ 0   28/5 -28/5 ] (R2 → R2 + 3/5 R1)
[ 1  -3   5 ]      [ 0  -21/5  21/5 ] (R3 → R3 − 1/5 R1)
                   [ 5    6    4 ]
                   [ 0   28/5 -28/5 ]
                   [ 0    0    0   ] (R3 → R3 + 3/4 R2)
```

Two pivots ⇒ rank 2 < 3 ⇒ **linearly dependent**. Reading the relation: 2 a − b − c = 0, since c = 2 a − b. (Verify: 2(5, −3, 1) − (6, 2, −3) = (4, −8, 5). ✓)

---

## Q2.2 [Dec 2025 · Q2b · 3 marks] ★

**Problem.** With V = span{a, b} (a, b above), explain why V is a subspace of ℝ³ and find dim V.

**Solution.** "Span of finitely many vectors is automatically a subspace" — the empty set membership 0 = 0·a + 0·b, closure under addition, and closure under scaling are all immediate from linearity. Since a, b are linearly independent (no scalar multiple of one gives the other; their cross product (3·5 − 1·(−3), 1·6 − 5·(−3), 5·2 − (−3)·6) ≠ 0), they form a basis. **dim V = 2**.

---

## Q2.3 [Dec 2025 · Q2c · 3 marks]

**Problem.** Is u = (4, −8, 2)ᵀ in V = span{(5, −3, 1)ᵀ, (6, 2, −3)ᵀ}?

**Solution (row reduce method).** Augment and eliminate:

```
[ 5   6 |  4 ]      [ 5    6  |  4   ]
[-3   2 | -8 ]  →   [ 0   28/5| -28/5]
[ 1  -3 |  2 ]      [ 0  -21/5|  6/5 ]

R3 → R3 + (3/4) R2 :
[ 5    6   |   4   ]
[ 0   28/5 | -28/5 ]
[ 0    0   | -27/5 ]
```

Last row gives 0 = −27/5 ≠ 0 ⇒ **u ∉ V**.

**Solution (cross-product method).** Normal of the plane V is n = a × b = ((−3)(−3) − 1·2, 1·6 − 5·(−3), 5·2 − (−3)·6) = (7, 21, 28) ‖ (1, 3, 4). Plug u into n: 1·4 + 3·(−8) + 4·2 = 4 − 24 + 8 = −12 ≠ 0 ⇒ off the plane ⇒ u ∉ V.

---

## Q2.4 [Jan 2026 · Q2a · 5 marks]

**Problem.** Find a basis and the dimension of the null space of

```
A = [  1  -1   1 ]
    [ -2   2  -2 ]
    [  3  -3   3 ]
```

**Solution.** R2 → R2 + 2 R1 and R3 → R3 − 3 R1 give

```
[ 1  -1   1 ]
[ 0   0   0 ]
[ 0   0   0 ]
```

Single equation x₁ − x₂ + x₃ = 0 with two free variables x₂ = s, x₃ = t.

Solution: x = s·(1, 1, 0)ᵀ + t·(−1, 0, 1)ᵀ. **Basis** {(1, 1, 0)ᵀ, (−1, 0, 1)ᵀ}, **dim = 2**.

---

## Q2.5 [Jan 2026 · Q2b · 3 marks]

**Problem.** State the rank-nullity theorem and verify it for the matrix in Q2.4.

**Solution.** Rank-nullity: for every m × n matrix A, rank(A) + nullity(A) = n. Here rank(A) = 1 (one pivot), nullity(A) = 2, n = 3 ⇒ 1 + 2 = 3. ✓

---

## Q2.6 [Jan 2026 · Q2c · 2 marks]

**Problem.** Give a 1-dimensional subspace U inside the null space of Q2.4.

**Solution.** Take any non-zero vector from the basis, e.g. U = span{(1, 1, 0)ᵀ}. As span of a single vector it is automatically a subspace, with dim 1.

---

## Q2.7 [2024 Regular · Q2 · 4 marks]

**Problem.** Decide whether U = {(x, y, z) ∈ ℝ³ : x + y − z = 0 and 2x + 3y = 0} is a subspace, and if so give its dimension.

**Solution.** U is the null space of A = [[1, 1, −1], [2, 3, 0]] ⇒ automatically a subspace. Reduce: R2 → R2 − 2R1 ⇒ [[1, 1, −1], [0, 1, 2]]. Two pivots ⇒ rank 2, nullity 1 ⇒ **dim U = 1**. From x + y = z and y = −2x: take x = 1 ⇒ y = −2, z = −1 ⇒ basis {(1, −2, −1)ᵀ}.

---

## Q2.8 [2023 Makeup · Q2 · 5 marks]

**Problem.** Does the set S = {(x, y, z) : x ≥ 0} form a subspace of ℝ³?

**Solution.** Closure under scalar multiplication fails: take x = 1, then (−1)·(1, 0, 0) = (−1, 0, 0) ∉ S. So **S is not a subspace** (also fails to contain v and −v simultaneously).

---

## Q2.9 [2022 Midsem · Q2 · 3 marks]

**Problem.** Show that S = {(x, y) ∈ ℝ² : x² + y² = 1} is **not** a subspace of ℝ².

**Solution.** 0 = (0, 0) is not on the unit circle (0² + 0² = 0 ≠ 1) ⇒ axiom (1) fails ⇒ S is not a subspace.

---

## Q2.10 [Practice · Q5 · 4 marks]

**Problem.** Find a basis of ℝ⁴ that contains the vectors (1, 1, 0, 0)ᵀ and (0, 0, 1, 1)ᵀ.

**Solution.** Extend by appending two standard basis vectors and removing dependence. Try (1, 1, 0, 0)ᵀ, (0, 0, 1, 1)ᵀ, e₁ = (1, 0, 0, 0)ᵀ, e₃ = (0, 0, 1, 0)ᵀ. Stack as columns and check rank by REF:

```
[ 1   0   1   0 ]
[ 1   0   0   0 ]
[ 0   1   0   1 ]
[ 0   1   0   0 ]
```

R2 → R2 − R1, R4 → R4 − R3 ⇒

```
[ 1   0   1   0 ]
[ 0   0  -1   0 ]
[ 0   1   0   1 ]
[ 0   0   0  -1 ]
```

Swap R2 ↔ R3 (still rank 4):

```
[ 1   0   1   0 ]
[ 0   1   0   1 ]
[ 0   0  -1   0 ]
[ 0   0   0  -1 ]
```

Four pivots ⇒ basis valid. **Basis:** {(1, 1, 0, 0)ᵀ, (0, 0, 1, 1)ᵀ, e₁, e₃}.

---

## Q2.11 [Companion Lecture 2 · 4 marks]

**Problem.** Show that the polynomials 1, x, x² form a basis of P₂(ℝ) (polynomials of degree ≤ 2).

**Solution.** Spanning: every polynomial a + bx + cx² is by definition a linear combination of 1, x, x². Independence: a·1 + b·x + c·x² = 0 (as a function) forces a = b = c = 0 (compare coefficients). So {1, x, x²} is both spanning and independent ⇒ basis. Dim P₂ = 3.

---

# Topic 3 — Inner products via SPD matrices, distance, angle, orthogonality

## Q3.1 [Dec 2025 · Q4a · 4 marks] ★

**Problem.** Show that A = [[3, −1], [−1, 1]] defines an inner product on ℝ² via ⟨x, y⟩ = xᵀAy.

**Solution.** Symmetry of A is visible (A = Aᵀ). Bilinearity follows from matrix algebra (xᵀAy is linear in each argument). Positive-definiteness: complete the square,

xᵀAx = 3 x₁² − 2 x₁ x₂ + x₂² = 2 x₁² + (x₁ − x₂)² ≥ 0,

with equality iff x₁ = 0 and x₁ − x₂ = 0 ⇔ x = 0. Hence A is SPD and ⟨·, ·⟩_A is an inner product.

**Alternative (Sylvester):** D₁ = 3 > 0, D₂ = det(A) = 3·1 − (−1)·(−1) = 2 > 0 ⇒ SPD.

---

## Q3.2 [Dec 2025 · Q4b · 3 marks]

**Problem.** With ⟨·, ·⟩_A from Q3.1, find d(u, v) for u = (1, −1)ᵀ, v = (2, 1)ᵀ.

**Solution.** w = u − v = (−1, −2)ᵀ. A w = (3·(−1) + (−1)·(−2), (−1)·(−1) + 1·(−2))ᵀ = (−1, −1)ᵀ.

wᵀ A w = (−1)(−1) + (−2)(−1) = 1 + 2 = 3 ⇒ d_A(u, v) = √3.

---

## Q3.3 [Dec 2025 · Q4c · 3 marks]

**Problem.** Are x = (−1, 2)ᵀ and y = (2, 1)ᵀ orthogonal under the **standard** inner product? Under ⟨·, ·⟩_A?

**Solution.** Standard: x·y = (−1)·2 + 2·1 = 0 ⇒ orthogonal.

Custom: A·y = (3·2 − 1, −1·2 + 1) = (5, −1). xᵀ(A y) = (−1)·5 + 2·(−1) = −7 ≠ 0 ⇒ **not** orthogonal.

So orthogonality is geometry-dependent.

---

## Q3.4 [Jan 2026 · Q4a · 4 marks]

**Problem.** Is

```
A = [ 2  1  0  0 ]
    [ 1  3  0  0 ]
    [ 0  0  4  1 ]
    [ 0  0  1  2 ]
```

symmetric and positive definite?

**Solution.** A is block-diagonal A = diag([[2, 1], [1, 3]], [[4, 1], [1, 2]]); each block is symmetric. For each block apply Sylvester:

- Block 1: D₁ = 2 > 0, D₂ = 2·3 − 1 = 5 > 0 ⇒ SPD.
- Block 2: D₁ = 4 > 0, D₂ = 4·2 − 1 = 7 > 0 ⇒ SPD.

A diagonal of SPD blocks is SPD. ✓

---

## Q3.5 [Jan 2026 · Q4b · 4 marks]

**Problem.** Compute d_A(u, v) for u = (−2, 0, 1, −1)ᵀ, v = (−2, 2, 1, 0)ᵀ.

**Solution.** w = u − v = (0, −2, 0, −1)ᵀ. A w = (block on (0, −2): (2·0 + 1·(−2), 1·0 + 3·(−2)) = (−2, −6); block on (0, −1): (4·0 + 1·(−1), 1·0 + 2·(−1)) = (−1, −2)) ⇒ A w = (−2, −6, −1, −2)ᵀ.

wᵀA w = 0·(−2) + (−2)·(−6) + 0·(−1) + (−1)·(−2) = 12 + 2 = 14 ⇒ d_A(u, v) = √14.

---

## Q3.6 [2024 Regular · Q2 · 8 marks]

**Problem.** For m ≥ 2 show A = [[m, 1, 1], [1, m, 1], [1, 1, m]] is SPD, then compute ‖x‖_A for x = (1, −1, −2)ᵀ.

**Solution (SPD).**

xᵀA x = m(x₁² + x₂² + x₃²) + 2(x₁x₂ + x₁x₃ + x₂x₃)
      = (x₁ + x₂)² + (x₁ + x₃)² + (x₂ + x₃)² + (m − 2)(x₁² + x₂² + x₃²).

Each term is ≥ 0; for m ≥ 2 the last term is ≥ 0. Equality forces all three sums to vanish and (if m > 2) all xᵢ = 0; for m = 2 the three sum-equalities (x₁ + x₂ = 0, x₁ + x₃ = 0, x₂ + x₃ = 0) still force x = 0. So SPD.

**Norm.** A x = (m·1 + 1·(−1) + 1·(−2), 1·1 + m·(−1) + 1·(−2), 1·1 + 1·(−1) + m·(−2)) = (m − 3, −m − 1, −2m).

xᵀA x = 1·(m − 3) + (−1)·(−m − 1) + (−2)·(−2m) = m − 3 + m + 1 + 4m = 6m − 2.

‖x‖_A = √(6m − 2). (At m = 2: √10.)

---

## Q3.7 [2024 Makeup · Q3 · 6 marks] ★ — Sylvester's criterion drill

**Problem.** Determine which of the following are SPD:

```
(i)  A₁ = [ 2  -1  0 ]    (ii)  A₂ = [ 1   2 ]   (iii)  A₃ = [ 1  2  3 ]
          [-1   2 -1 ]                [ 2   1 ]                [ 2  5  6 ]
          [ 0  -1  2 ]                                          [ 3  6 14 ]
```

**Solution.**

(i) D₁ = 2, D₂ = 2·2 − 1 = 3, D₃ = det A₁. Use cofactor: D₃ = 2(2·2 − 1) − (−1)((−1)·2 − 0) + 0 = 2·3 + (−1)·(−2) = 6 + 2 = 4 ⇒ all > 0 ⇒ SPD.

(ii) D₁ = 1 > 0, D₂ = 1·1 − 2·2 = −3 < 0 ⇒ **not** SPD.

(iii) D₁ = 1, D₂ = 5 − 4 = 1, D₃ = det = 1(5·14 − 36) − 2(2·14 − 18) + 3(12 − 15) = 34 − 20 − 9 = 5 ⇒ all > 0 ⇒ SPD.

---

## Q3.8 [Practice · Q12 · 6 marks] — Gram–Schmidt drill

**Problem.** Apply the Gram–Schmidt process to v₁ = (1, 1, 0), v₂ = (1, 0, 1), v₃ = (0, 1, 1) to obtain an orthonormal basis of ℝ³.

**Solution.**

q₁ = v₁ / ‖v₁‖ = (1, 1, 0)/√2.

u₂ = v₂ − ⟨v₂, q₁⟩ q₁. ⟨v₂, q₁⟩ = (1 + 0)/√2 = 1/√2. So u₂ = (1, 0, 1) − (1/2)(1, 1, 0) = (1/2, −1/2, 1). ‖u₂‖ = √(1/4 + 1/4 + 1) = √(3/2). q₂ = (1/2, −1/2, 1)/√(3/2) = (1, −1, 2)/√6.

u₃ = v₃ − ⟨v₃, q₁⟩ q₁ − ⟨v₃, q₂⟩ q₂. ⟨v₃, q₁⟩ = (0 + 1)/√2 = 1/√2. ⟨v₃, q₂⟩ = (0·1 + 1·(−1) + 1·2)/√6 = 1/√6.

u₃ = (0, 1, 1) − (1/2)(1, 1, 0) − (1/6)(1, −1, 2) = (−1/2 − 1/6, 1/2 + 1/6, 1 − 1/3) = (−2/3, 2/3, 2/3) = (2/3)(−1, 1, 1).

‖u₃‖ = (2/3)·√3 = 2/√3. q₃ = (−1, 1, 1)/√3.

**Orthonormal basis:** {(1, 1, 0)/√2, (1, −1, 2)/√6, (−1, 1, 1)/√3}.

---

## Q3.9 [Companion Lecture 3 · 4 marks] — Cholesky drill

**Problem.** Compute the Cholesky factorisation L of A = [[4, 2], [2, 5]].

**Solution.** ℓ₁₁ = √4 = 2, ℓ₂₁ = 2/2 = 1, ℓ₂₂ = √(5 − 1²) = 2. So L = [[2, 0], [1, 2]] and L Lᵀ = [[4, 2], [2, 5]] = A. ✓

---

# Topic 4 — Eigenvalues, eigenvectors, diagonalisation

## Q4.1 [Dec 2025 (regular) sub-question · 4 marks] ★

**Problem.** Find the eigenvalues and eigenvectors of A = [[3, −1], [−1, 1]].

**Solution.** Char poly det(A − λI) = (3 − λ)(1 − λ) − 1 = λ² − 4λ + 2 = 0 ⇒ λ = 2 ± √2.

For λ₁ = 2 + √2: A − λ₁ I = [[1 − √2, −1], [−1, −1 − √2]]. The first row gives (1 − √2)·v₁ = v₂, so v⁽¹⁾ = (1, 1 − √2)ᵀ.

For λ₂ = 2 − √2: similarly v⁽²⁾ = (1, 1 + √2)ᵀ.

Cross-check: tr A = 4 = (2 + √2) + (2 − √2). det A = 2 = (2 + √2)(2 − √2) = 4 − 2. ✓

---

## Q4.2 [2024 Regular · Q3 · 8 marks] — diagonalisation drill

**Problem.** Diagonalise A = [[2, 0, 0], [1, 2, 1], [−1, 0, 1]].

**Solution.** Char poly det(A − λI). Expand along column 2 (one non-zero entry):

det = (2 − λ)·det([[2 − λ, 1], [−1, 1 − λ]]) = (2 − λ)·[(2 − λ)(1 − λ) + 1]

But there is also a contribution from a zero — easier: this matrix is lower-triangular **plus** a coupling. Use direct minors.

Actually expand row 1:

det = (2 − λ)·det([[2 − λ, 1], [0, 1 − λ]]) − 0 + 0 = (2 − λ)²(1 − λ).

Eigenvalues λ = 2 (AM 2), λ = 1 (AM 1).

For λ = 2: A − 2I = [[0, 0, 0], [1, 0, 1], [−1, 0, −1]]. Rank 1 ⇒ GM = 3 − 1 = 2. Solve x₁ + x₃ = 0 (from R2); x₂ free; x₃ free. Two eigenvectors: (0, 1, 0)ᵀ (set x₃ = 0, x₂ = 1) and (1, 0, −1)ᵀ.

For λ = 1: A − I = [[1, 0, 0], [1, 1, 1], [−1, 0, 0]]. RREF [[1, 0, 0], [0, 1, 1], [0, 0, 0]] ⇒ x₁ = 0, x₂ = −x₃ ⇒ eigenvector (0, 1, −1)ᵀ.

P = [[0, 1, 0], [1, 0, 1], [0, −1, −1]], D = diag(2, 2, 1). Verify A P = P D.

---

## Q4.3 [Practice · Q22 · 5 marks] — symmetric ⇒ orthogonal Q

**Problem.** Find the orthogonal diagonalisation A = Q D Qᵀ for A = [[2, 1], [1, 2]].

**Solution.** Char poly: (2 − λ)² − 1 = 0 ⇒ λ = 3, 1.

Eigenvectors: λ = 3 ⇒ (A − 3I)v = 0: [[−1, 1], [1, −1]] ⇒ v ∝ (1, 1). Normalise: q₁ = (1, 1)/√2.

λ = 1 ⇒ [[1, 1], [1, 1]] ⇒ v ∝ (1, −1). Normalise: q₂ = (1, −1)/√2. (Already orthogonal because eigenvalues are distinct.)

Q = (1/√2)[[1, 1], [1, −1]], D = diag(3, 1). Verify Q Qᵀ = I and Qᵀ A Q = D.

---

## Q4.4 [2022 Midsem · Q3 · 6 marks] — non-diagonalisable example

**Problem.** Show that A = [[1, 1], [0, 1]] is **not** diagonalisable.

**Solution.** Char poly (1 − λ)² ⇒ λ = 1 (AM 2). A − I = [[0, 1], [0, 0]] has rank 1 ⇒ GM = 2 − 1 = 1 < AM. So A is defective and cannot be diagonalised.

---

## Q4.5 [Companion Lecture 4 · drill]

**Problem.** Find eigenvalues of a triangular matrix A = [[5, 7, 9], [0, −1, 4], [0, 0, 3]].

**Solution.** Eigenvalues of any triangular matrix are its diagonal entries: **λ ∈ {5, −1, 3}**.

---

## Q4.6 [Practice · Q21 · 4 marks]

**Problem.** Show that any 2 × 2 real symmetric matrix has only real eigenvalues.

**Solution.** A = [[a, b], [b, c]]. Char poly λ² − (a + c)λ + (ac − b²) = 0. Discriminant Δ = (a + c)² − 4(ac − b²) = (a − c)² + 4b² ≥ 0 ⇒ both roots are real. (Special case of the spectral theorem.)

---

## Q4.7 [Companion Lecture 4 · 4 marks]

**Problem.** Compute A¹⁰ for A = [[2, 1], [1, 2]] using diagonalisation.

**Solution.** From Q4.3, A = Q D Qᵀ with D = diag(3, 1). Then A¹⁰ = Q D¹⁰ Qᵀ = Q · diag(3¹⁰, 1) · Qᵀ = (1/2)·[[3¹⁰ + 1, 3¹⁰ − 1], [3¹⁰ − 1, 3¹⁰ + 1]]. Numerically 3¹⁰ = 59049 ⇒ A¹⁰ = [[29525, 29524], [29524, 29525]].

---

# Topic 5 — Singular Value Decomposition (SVD)

## Q5.1 [Dec 2025 · Q3 · 8 marks] ★ — full SVD

**Problem.** Compute the SVD of A = [[1, 2, 0], [0, 0, 1]].

**Solution.**

1. AᵀA = [[1, 2, 0], [2, 4, 0], [0, 0, 1]].
2. Char poly: det(AᵀA − μI). Expand:
   - Top 2 × 2 block has eigenvalues 5 and 0 (trace 5, det 0).
   - The (3, 3) block contributes eigenvalue 1.
   - So eigenvalues are μ ∈ {5, 1, 0} ⇒ singular values σ = (√5, 1).
3. Eigenvectors of AᵀA:
   - μ = 5: solve [[−4, 2, 0], [2, −1, 0], [0, 0, −4]] v = 0 ⇒ v = (1, 2, 0)/√5.
   - μ = 1: solve [[0, 2, 0], [2, 3, 0], [0, 0, 0]] v = 0 ⇒ v = (0, 0, 1).
   - μ = 0 (kernel): v = (2, −1, 0)/√5.
4. V = [(1, 2, 0)/√5 | (0, 0, 1) | (2, −1, 0)/√5].
5. Left singular vectors: u₁ = (1/√5)·A·(1, 2, 0)ᵀ/√5 = (1/√5)·(5/√5, 0)ᵀ = (1, 0)ᵀ. u₂ = (1/1)·A·(0, 0, 1)ᵀ = (0, 1)ᵀ.
6. U = I₂. Σ = [[√5, 0, 0], [0, 1, 0]].

Check: U Σ Vᵀ has first row (1)·(√5)·(1/√5, 2/√5, 0) + (1)·(1)·(0, 0, 1)·... actually compute directly: Σ Vᵀ = [[√5, 0, 0], [0, 1, 0]]·Vᵀ. Column 1 of Vᵀ is (1/√5, 0, 2/√5)ᵀ ⇒ Σ Vᵀ column 1 = (1, 0)ᵀ. Continue: A = (1, 0)·v₁ᵀ·√5 + (0, 1)·v₂ᵀ·1 = √5·(1/√5, 2/√5, 0) e₁ᵀ ⊗ ... finally rebuilds A. ✓

---

## Q5.2 [Jan 2026 · Q3 · 8 marks] — variant SVD

**Problem.** SVD of A = [[1, 0, 1], [0, 1, 0]].

**Solution.** AᵀA = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]. Eigenvalues: column-2 contributes 1 (eigenvector e₂); the 2 × 2 block on (1, 3) is [[1, 1], [1, 1]] with eigenvalues 2, 0 and eigenvectors (1, 1)/√2, (1, −1)/√2. So μ ∈ {2, 1, 0} ⇒ σ = (√2, 1).

V columns (in matching order): v₁ = (1, 0, 1)/√2, v₂ = (0, 1, 0), v₃ = (1, 0, −1)/√2.

u₁ = (1/√2)·A·v₁ = (1/√2)·A·(1, 0, 1)/√2 = (1/√2)·(1/√2, 0)·... Compute: A·(1, 0, 1)ᵀ = (1·1 + 0·0 + 1·1, 0 + 1·0 + 0)ᵀ = (2, 0)ᵀ. So u₁ = (1/√2)·(2, 0)/√2 = (1, 0)ᵀ.

u₂ = A·(0, 1, 0)ᵀ = (0, 1)ᵀ.

U = I₂. Σ = [[√2, 0, 0], [0, 1, 0]]. Done.

---

## Q5.3 [Practice · Q23 · 6 marks] — diagonal A

**Problem.** SVD of A = [[3, 0], [0, −2]].

**Solution.** AᵀA = [[9, 0], [0, 4]] ⇒ μ = 9, 4 ⇒ σ = 3, 2. V = I.

u₁ = (1/3)·A·e₁ = (1/3)(3, 0)ᵀ = (1, 0)ᵀ. u₂ = (1/2)·A·e₂ = (1/2)·(0, −2)ᵀ = (0, −1)ᵀ.

U = [[1, 0], [0, −1]]. Σ = diag(3, 2). The negative eigenvalue of A is absorbed into U.

---

## Q5.4 [Practice / 2024 Makeup · 6 marks] — rank-1 matrix

**Problem.** Compute σ₁ for A = [[1, 1], [2, 2]].

**Solution.** A = (1, 2)ᵀ·(1, 1) is rank 1. σ₁ = ‖(1, 2)‖·‖(1, 1)‖ = √5·√2 = √10. (All other singular values are 0.)

---

## Q5.5 [Companion Lecture 5 · 4 marks] — low-rank approximation

**Problem.** A 3 × 3 matrix has singular values σ = (5, 3, 1). Find the Frobenius-norm error of the best rank-2 approximation.

**Solution.** ‖A − A₂‖_F = √(σ₃²) = 1 (Eckart–Young).

---

## Q5.6 [2024 Makeup · 6 marks] — symmetric A, eigen vs SVD

**Problem.** A = [[2, 1], [1, 2]]. Find eigenvalues, singular values and the relationship.

**Solution.** From Q4.3, eigenvalues 3, 1. A is symmetric and **positive** definite (both > 0), so singular values equal eigenvalues: σ = (3, 1). U = V = Q from the spectral decomposition.

If instead A were [[−2, 1], [1, −2]], its eigenvalues would be (−1, −3) but its singular values would be (1, 3) (absolute values), and one column of U would flip sign relative to Q.

---

# Topic 6 — Gradients, Hessian, classification of critical points, chain rule

## Q6.1 [Dec 2025 · Q5 · 8 marks] ★

**Problem.** Find and classify all critical points of f(x, y) = x³ − 3 x y + y³.

**Solution.** ∇f = (3x² − 3y, 3y² − 3x). Setting both to zero ⇒ y = x² and x = y². Substituting: x = (x²)² = x⁴ ⇒ x(x³ − 1) = 0 ⇒ x ∈ {0, 1}. Critical points (0, 0) and (1, 1).

H_f = [[6x, −3], [−3, 6y]].

At (0, 0): H = [[0, −3], [−3, 0]]. det H = −9 < 0 ⇒ **saddle**.
At (1, 1): H = [[6, −3], [−3, 6]]. a = 6 > 0, det = 27 > 0 ⇒ **local minimum**. f(1, 1) = 1 − 3 + 1 = −1.

---

## Q6.2 [Jan 2026 · Q5 · 8 marks]

**Problem.** Classify the critical points of f(x, y) = x⁴ + y⁴ − 4 x y.

**Solution.** ∇f = (4x³ − 4y, 4y³ − 4x). Set equal to zero: y = x³, x = y³ ⇒ x = x⁹ ⇒ x(x⁸ − 1) = 0 ⇒ x ∈ {0, 1, −1} (real roots). With y = x³: critical points (0, 0), (1, 1), (−1, −1).

H_f = [[12 x², −4], [−4, 12 y²]].

At (0, 0): H = [[0, −4], [−4, 0]] ⇒ det = −16 < 0 ⇒ **saddle**.
At (1, 1) and (−1, −1): H = [[12, −4], [−4, 12]] ⇒ a = 12 > 0, det = 144 − 16 = 128 > 0 ⇒ **local minima**. f-value: 1 + 1 − 4 = −2 at both.

---

## Q6.3 [2024 Regular · Q5 · 6 marks]

**Problem.** Find local extrema of f(x, y) = x² + y² − 2x + 4y + 5.

**Solution.** Complete the square: f = (x − 1)² + (y + 2)² + 5 − 1 − 4 = (x − 1)² + (y + 2)². Unique global minimum at (1, −2) with f = 0. Hessian H = 2 I positive definite ⇒ confirms local (and global) minimum.

---

## Q6.4 [2022 Midsem · Q5 · 6 marks]

**Problem.** Find and classify the critical points of f(x, y) = x³ + y³ − 3 x y.

**Solution.** ∇f = (3x² − 3y, 3y² − 3x). y = x², x = y² ⇒ x = x⁴ ⇒ x ∈ {0, 1}. Same as Q6.1.

---

## Q6.5 [Practice · Q15 · 4 marks]

**Problem.** Compute the gradient and Hessian of f(x) = (1/2) xᵀ A x − bᵀ x with A symmetric.

**Solution.** ∇f(x) = A x − b (using ∇(xᵀ A x) = 2 A x for symmetric A divided by 2). H_f(x) = A.

Critical point: A x* = b. Classification: x* is a strict minimum iff A is positive definite.

---

## Q6.6 [Companion Lecture 7 · drill] — chain rule

**Problem.** Compute dz/dt where z = x² + y², x = sin t, y = cos t.

**Solution.** dz/dt = ∂z/∂x · dx/dt + ∂z/∂y · dy/dt = 2x cos t + 2y (−sin t) = 2 sin t cos t − 2 sin t cos t = 0.

(Indeed z = sin² t + cos² t = 1 is constant.)

---

## Q6.7 [Companion Lecture 7 · 4 marks] — Jacobian of vector function

**Problem.** Compute the Jacobian of F(x, y) = (x² + y, x y, e^x cos y) at (0, 0).

**Solution.**

```
J_F = [ ∂F₁/∂x  ∂F₁/∂y ]   [ 2x       1     ]
      [ ∂F₂/∂x  ∂F₂/∂y ] = [ y        x     ]
      [ ∂F₃/∂x  ∂F₃/∂y ]   [ e^x·cos y  -e^x·sin y ]
```

At (0, 0): J_F(0, 0) = [[0, 1], [0, 0], [1, 0]].

---

## Q6.8 [Companion Lecture 7 · 5 marks] — composition

**Problem.** g(t) = (cos t, sin t), f(x, y) = x² + y². Compute (f ∘ g)'(t) two ways.

**Solution.**

(a) Substitution: f(g(t)) = cos² t + sin² t = 1 ⇒ derivative = 0.

(b) Chain rule: ∇f(x, y) = (2x, 2y); J_g(t) = (−sin t, cos t)ᵀ. Then (f ∘ g)'(t) = ∇f(g(t))ᵀ · J_g(t) = (2 cos t)(−sin t) + (2 sin t)(cos t) = 0. ✓

---

## Q6.9 [Practice · Q18 · 5 marks] — Hessian classification 3 × 3

**Problem.** Classify the critical point (0, 0, 0) of f(x, y, z) = x² + 2 y² + 3 z² + x y + y z.

**Solution.** ∇f = (2x + y, 4y + x + z, 6z + y). Setting to zero ⇒ x = y = z = 0. So (0, 0, 0) is a critical point.

H_f = [[2, 1, 0], [1, 4, 1], [0, 1, 6]] (constant). Sylvester: D₁ = 2, D₂ = 8 − 1 = 7, D₃ = 2(24 − 1) − 1·(6 − 0) + 0 = 46 − 6 = 40 ⇒ all > 0 ⇒ H is **positive definite** ⇒ strict local **minimum**.

---

# Topic 7 — Multivariate Taylor expansion, linearisation

## Q7.1 [Dec 2025 · Q6a · 4 marks]

**Problem.** Find the linear (degree-1) Taylor approximation of f(x, y, z) = x² y + x z at (1, 2, 3).

**Solution.** ∇f = (2 x y + z, x², x). At (1, 2, 3): ∇f = (4 + 3, 1, 1) = (7, 1, 1). f(1, 2, 3) = 2 + 3 = 5.

T₁(x, y, z) = 5 + 7·(x − 1) + 1·(y − 2) + 1·(z − 3).

---

## Q7.2 [Dec 2025 · Q6b · 6 marks]

**Problem.** Compute the second-order Taylor polynomial of g(x, y) = e^x · sin y at (0, 0).

**Solution.** g(0, 0) = 0. g_x = e^x sin y → 0; g_y = e^x cos y → 1. g_{xx} = e^x sin y → 0; g_{yy} = −e^x sin y → 0; g_{xy} = e^x cos y → 1.

T₂(x, y) = 0 + 0·x + 1·y + ½·(0 x² + 2 x y + 0 y²) = y + x y.

---

## Q7.3 [Jan 2026 · Q6 · 8 marks]

**Problem.** Find the second-order Taylor polynomial of f(x, y) = ln(1 + x + y²) at (0, 0).

**Solution.** Let u = x + y² so f = ln(1 + u). f(0, 0) = 0.

f_x = 1/(1 + u) → 1; f_y = 2y/(1 + u) → 0.

f_{xx} = −1/(1 + u)² → −1.
f_{yy} = 2/(1 + u) − 4 y²/(1 + u)² → 2.
f_{xy} = −2y/(1 + u)² → 0.

T₂(x, y) = 0 + x + 0 + ½·(−1·x² + 2·0·x y + 2·y²) = x − x²/2 + y².

---

## Q7.4 [2024 Regular · Q6 · 6 marks]

**Problem.** Linearise f(x, y) = √(x² + y² + 1) at (1, 1).

**Solution.** f(1, 1) = √3. f_x = x/√(x² + y² + 1), f_y = y/√(x² + y² + 1). At (1, 1): both equal 1/√3.

T₁(x, y) = √3 + (1/√3)(x − 1) + (1/√3)(y − 1).

---

## Q7.5 [Companion Lecture 8 · drill]

**Problem.** Write Taylor's theorem in the multivariate form to second order, including the remainder.

**Solution.** For f: ℝⁿ → ℝ twice continuously differentiable, with x close to x₀,

f(x) = f(x₀) + ∇f(x₀)ᵀ (x − x₀) + ½ (x − x₀)ᵀ H_f(ξ) (x − x₀)

for some ξ on the segment from x₀ to x (Lagrange remainder). The polynomial form (no remainder) keeps H at x₀ and has error o(‖x − x₀‖²).

---

## Q7.6 [Practice · Q19 · 5 marks] — third-order Taylor

**Problem.** Compute T₃ of f(x) = sin x at x₀ = 0 and use it to estimate sin(0.2).

**Solution.** sin x = x − x³/6 + O(x⁵). T₃(0.2) = 0.2 − (0.2)³/6 = 0.2 − 0.0013333… ≈ 0.198667. Actual sin 0.2 ≈ 0.198669; error < 10⁻⁵.

---

## Q7.7 [2022 Midsem · Q6 · 6 marks]

**Problem.** Approximate (1.02)^{1.97} using a degree-1 Taylor polynomial of f(x, y) = x^y at (1, 2).

**Solution.** f(1, 2) = 1. f_x = y x^{y−1} → 2. f_y = x^y · ln x → 0.

T₁(x, y) = 1 + 2(x − 1) + 0·(y − 2). At (1.02, 1.97): T₁ ≈ 1 + 2·0.02 = 1.04. (Exact ≈ 1.0395.)

---

## Q7.8 [Companion Lecture 8 · 4 marks] — smallest value of T₂ on a region

**Problem.** Let T₂(x, y) = x² + y² − 2 x + 4 y + 5 and S = {(x, y) : x² + y² ≤ 4}. Find min and max of T₂ on S.

**Solution.** Interior: ∇T₂ = (2x − 2, 2y + 4) = 0 ⇒ (1, −2). Distance from origin √5 > 2 ⇒ outside S.

Boundary x² + y² = 4: parametrise x = 2 cos θ, y = 2 sin θ. T₂ = 4 + 5 − 4 cos θ + 8 sin θ = 9 + (− 4 cos θ + 8 sin θ). The amplitude of −4 cos θ + 8 sin θ is √(16 + 64) = √80 = 4√5.

So T₂ ∈ [9 − 4√5, 9 + 4√5] on the boundary. Since the unconstrained minimum is outside, **min = 9 − 4√5** (achieved on boundary), **max = 9 + 4√5**.

---

# Appendix — High-yield short-answer drills

## A.1 Definitions to be ready to write verbatim

- **REF, RREF**: pivots descending strictly to the right; in RREF pivots = 1 and unique non-zero in their columns.
- **Rank**: number of pivots in any REF.
- **Subspace** of V: contains 0, closed under +, closed under scalar multiplication.
- **Basis**: linearly independent + spanning.
- **Dimension**: cardinality of any basis.
- **Inner product**: symmetric, bilinear, positive-definite map V × V → ℝ.
- **Orthogonal**: ⟨x, y⟩ = 0.
- **Eigenvalue/eigenvector**: A v = λ v with v ≠ 0.
- **Diagonalisable**: A = P D P⁻¹.
- **SVD**: A = U Σ Vᵀ, U, V orthogonal, Σ diagonal with σ ≥ 0.
- **Gradient, Hessian**: ∇f, H_f as defined in the guide.
- **Critical point**: ∇f(x*) = 0.

## A.2 Numeric checkpoints (memorise)

- A 2 × 2 SPD checker: a > 0 and det A > 0.
- Eigenvalues of [[a, b], [b, c]]: ((a + c) ± √((a − c)² + 4b²))/2.
- Eigenvalues of [[2, 1], [1, 2]]: 3 and 1.
- Eigenvalues of [[3, −1], [−1, 1]]: 2 ± √2.
- 3 × 3 with a row of zeros has det = 0.

## A.3 Common pitfalls and rescues

| Pitfall | Rescue |
|---|---|
| Forgetting to count column for free variables | Count pivots vs columns of A |
| Mixing rows of A and rows of [A | b] when reading rank | Always reduce both blocks together |
| Eigenvectors not normalised when V or U is required orthogonal | Divide by ‖v‖ before placing in V/U columns |
| σ ordering | Sort decreasingly and reorder V columns accordingly |
| Saddle vs min in 2 × 2 H | Use det H sign first; only check trace if det > 0 |
| Forgetting (m − 2)·sum-of-squares in completing the square | Group cross-terms first |
| Missing minus sign in negative eigenvalue case for SVD of symmetric A | Multiply corresponding column of U by −1 |

## A.4 The 90-minute plan

| Min | Activity |
|---|---|
| 0–5 | Read all six questions, mark difficulty 1–3 |
| 5–20 | Do the easiest two (typically Q1 and Q6 linearisation) |
| 20–50 | Q2 + Q5 (subspaces / Hessian classification) |
| 50–80 | Q3 / Q4 (eigen / SVD / inner product) |
| 80–90 | Sanity checks: tr/det vs eigenvalues, Frobenius vs σᵢ², plug critical points back into ∇f |

## A.5 Tactical tips collected from past keys (Saurabh + official answer keys)

1. For **rank questions**, REF is enough — never compute determinant of an m × n with m ≠ n.
2. For **inner-product questions**, write down xᵀA y in coordinates *before* substituting numbers; cancel common factors.
3. For **SVD**, always use AᵀA (smaller of the two squares for short-fat A; AAᵀ for tall-thin A).
4. For **diagonalisability**, verify GM = AM only for repeated eigenvalues — single roots automatically pass.
5. For **subspace test**, exhibit an explicit failure (a counterexample) when claiming "not a subspace"; vague reasoning loses marks.
6. For **Hessian classification**, write down the 2 × 2 sub-determinant rule even when you also use eigenvalues — graders reward both steps.
7. For **Taylor**, *always* state the formula T_n = … before plugging values; this earns method marks even if arithmetic slips.

## A.6 Cross-paper question map (which year tests what)

| Year / Source | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 |
|---|---|---|---|---|---|---|
| Dec 2025 Reg | system + det | dependence + span | SVD 2×3 | inner product 2×2 | classify x³ − 3xy + y³ | Taylor xy + xz |
| Jan 2026 Make | rank 3×3 + inverse 4×4 | null space + 1-dim subspace | SVD 2×3 variant | block SPD + distance | classify x⁴+y⁴−4xy | Taylor ln(1 + x + y²) |
| Jan 2025 Reg | parameter consistency | basis check | eigen of symmetric 3×3 | inner product / Sylvester | classify quadratic | Taylor of e^{x} cos y |
| 2024 Reg | parameter family | subspace dim | diagonalise 3 × 3 | SPD + norm | extrema sum of squares | linearise √(...) |
| 2024 Make | rank | basis | SVD rank-1 | Sylvester | gradient + Hessian | Taylor with remainder |
| 2023 Make | system | subspace test | SVD or eigen | inner product | extrema | Taylor / linearise |
| 2022 | system | basis check | non-diag example | inner product | classify x³+y³−3xy | Taylor (1.02)^1.97 |
| 2020 | parameter family | basis | diagonalise | inner product | classify | Taylor |

This rotation is *very* tight: every paper has exactly one of {linear systems, parameter consistency, rank-inverse} as Q1, exactly one of {subspace test, basis, span} as Q2, and so on. Drilling 2–3 variants from each topic block is enough to cover the entire midsem.
