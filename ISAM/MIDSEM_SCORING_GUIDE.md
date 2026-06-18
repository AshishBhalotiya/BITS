# ISM Mid-Semester Exam — 10-Hour Scoring Guide

**Course**: AIMLCZC418 — Introduction to Statistical Methods
**Scope (per latest notification)**: Up to Contact Session 7. Primary focus: **Sessions 1–6** (significant portion of the paper). CS-7 included for safety.

This guide is built from: course handout, the seven session companion notes (Prof. Saurabh), all past mid-semester papers from 2023–2026 (regular + makeup), the practice problem set, and the webinar materials. Every formula uses **plain Unicode math** so it renders in any Markdown viewer.

---

## How the paper actually scores (look at every recent paper)

Six questions, 30 marks, 2 hours, closed book. The recurring pattern across **2023, 2024, 2024 makeup, 2025 regular, Dec 2025, Jan 2026 makeup** is:

| # | Worth | Almost-always asks |
| - | ----- | ------------------ |
| **Q1** | 5 | Descriptive stats: 5-number summary, IQR, outlier check, mean/SD, skewness comment |
| **Q2** | 5 | Probability rules **or** Naïve Bayes text classifier (one of the two) |
| **Q3** | 5 | Bayes' theorem (rare-disease, three-source, spam) **or** total-probability tree |
| **Q4** | 5 | Random variable: discrete PMF (find k, E[X], Var(X)) **or** joint PMF/PDF |
| **Q5** | 5 | Continuous RV (PDF/CDF) **or** Normal distribution (z-score, percentile) |
| **Q6** | 5 | Binomial / Poisson / Normal approximation **or** mixed inclusion-exclusion |

**Strategy.** Q1 and Q4/Q5 are mechanical — bank them in the first 40 minutes. Bayes (Q3) and Naïve Bayes (Q2) carry 10 marks and are formula-driven; do them next. Save approximation problems (Q6) for last because they reuse setup from earlier questions.

---

## 0. Notation & symbols you'll need

- Sample mean **x̄**, population mean **μ**.
- Sample variance **s²**, population variance **σ²**, standard deviation **σ** or **s**.
- Sum **Σ**, integral **∫**, factorial **n!**, "n choose k" written **C(n,k)** or **(n choose k)**.
- Probability of event A: **P(A)**. Conditional: **P(A | B)**. Complement: **Aᶜ**.
- Union **A ∪ B**, intersection **A ∩ B**, empty set **∅**.
- Greek you must recognise: **λ** (Poisson rate), **μ, σ** (Normal), **α** (significance), **π** (mixing weight or 3.14159…).
- ≈ "approximately", ∝ "proportional to".

---

## HOUR 1 — DESCRIPTIVE STATISTICS (Session 1)

This is **always** Q1. Master the mechanical recipe and you bank 4–5 marks in 8 minutes.

### 1.1 The four questions every dataset asks

1. **What kind of data?** Categorical (nominal, ordinal) vs Numerical (discrete, continuous). Levels of measurement: **Nominal → Ordinal → Interval → Ratio**. Higher level allows more arithmetic.
2. **Where is the centre?** mean, median, mode.
3. **How spread out?** range, variance, standard deviation, IQR.
4. **What shape?** symmetric vs skewed (left/right). Outliers via the 1.5×IQR rule.

### 1.2 Centre — mean, median, mode

For data y₁, y₂, …, yₙ:
- **Mean**: ȳ = (Σ yᵢ) / n. Pulled by outliers.
- **Median**: middle value of sorted data; for even n, average of two middle values.
- **Mode**: most-frequent value. Only centre that works for categorical data.

**Empirical relation (mildly skewed):** Mode ≈ 3·Median − 2·Mean.

### 1.3 Spread — variance, SD, range, IQR

- **Range** = max − min. Quick but distorted by outliers.
- **Sample variance**: s² = Σ(yᵢ − ȳ)² / (n − 1).
- **Sample SD**: s = √s². Reads as "typical distance from the mean."
- **Population**: divide by N (use μ instead of ȳ).
- **Coefficient of Variation**: CV = (s / ȳ) × 100%. Use to compare consistency across datasets.

> **Why divide by n−1?** Bessel's correction — the sample mean hugs its own data, so dividing by n underestimates spread. n−1 corrects that bias.

### 1.4 Shape — skew via mean vs median

- **Symmetric**: mean = median = mode.
- **Right (positive) skew**: mean > median > mode. Long right tail (incomes, waiting times).
- **Left (negative) skew**: mean < median < mode. Long left tail (easy-test scores).

**Test rule used in 2024 Q1:** if mean > median → positive skew; if mean < median → negative skew; if equal → symmetric. The mean leans toward the tail.

### 1.5 Five-number summary, IQR, outliers

The five-number summary is **(Min, Q1, Median, Q3, Max)**.

- **Q1** = 25th percentile (median of lower half).
- **Q3** = 75th percentile (median of upper half).
- **IQR** = Q3 − Q1.
- **Lower fence** = Q1 − 1.5·IQR.
- **Upper fence** = Q3 + 1.5·IQR.
- Any point **outside** [lower fence, upper fence] is a potential outlier.

#### How to find Q1, Q3 by hand (the method past papers accept)

Method 1 — **median-of-halves** (simplest, scored OK):
1. Sort the data.
2. Split into two halves at the median (exclude the median itself if n is odd).
3. Q1 = median of lower half; Q3 = median of upper half.

Method 2 — **(n+1) position** (used in Jan 2026 makeup answer key):
- Q1 position = 0.25·(n + 1); Q3 position = 0.75·(n + 1). Interpolate if not whole.

> **Either method earns full marks** if you state your method. The answer keys explicitly say "may be different when another method is used."

### 1.6 Worked example — full Q1 solution

**Data (Dec 2025 Q1, fast-food fat g/serving):** 7, 4, 20, 23, 25, 19, 30, 30, 40, 56.

1. **Sort:** 4, 7, 19, 20, 23, 25, 30, 30, 40, 56. (n = 10)
2. **Mean:** ȳ = (4+7+19+20+23+25+30+30+40+56) / 10 = 254/10 = **25.4**.
3. **Median:** average of 5th and 6th = (23 + 25)/2 = **24**.
4. **Q1 (median of lower 5: {4,7,19,20,23}):** = **19**.
5. **Q3 (median of upper 5: {25,30,30,40,56}):** = **30**.
6. **IQR:** 30 − 19 = **11**.
7. **Fences:** lower = 19 − 1.5·11 = 19 − 16.5 = 2.5; upper = 30 + 16.5 = 46.5.
8. **Outliers:** 56 > 46.5 → **56 is an outlier**.
9. **Sample SD:** Σ(yᵢ − 25.4)² = 458.16 + 338.56 + 40.96 + 29.16 + 5.76 + 0.16 + 21.16 + 21.16 + 213.16 + 936.36 = 2064.6. s² = 2064.6 / 9 = 229.4. s ≈ **15.15**.
10. **Skewness:** mean (25.4) > median (24) → **positively skewed**, consistent with the high outlier 56.

### 1.7 Five-number summary — the **interpretation paragraph** (always 1–2 marks)

When asked to "comment" or "interpret":
- *Centre*: mean ≈ X gives the typical value; the median is the robust middle.
- *Spread*: SD ≈ Y measures average distance from the mean; IQR captures the middle 50%.
- *Shape*: mean vs median tells skew direction; comment on the tail.
- *Outliers*: list them and say what they likely mean (process shift, data error, special-cause event, or genuine extremes).

This template alone earns the 2-mark "interpretation" sub-question on every paper.

### 1.8 Multiple-dataset comparison via CV (2023 Regular Q1)

When asked which dataset is "most consistent":
- Compute CV = (SD / mean)·100% for each.
- Smaller CV = more consistent.

When asked about **skew direction** from a summary table: compare mean to median (50% column).

When asked about **outliers from a summary table**: check if min < Q1 − 1.5·IQR or max > Q3 + 1.5·IQR.

---

## HOUR 2 — PROBABILITY BASICS (Session 2)

This drives Q2 (rules) and supplies the foundation for Q3 (Bayes), Q4 (RV), Q6 (distributions).

### 2.1 Vocabulary

- **Random experiment**: outcome unknown in advance under identical conditions.
- **Sample space S**: set of all possible outcomes.
- **Event A**: any subset of S.
- **|S|** = size of sample space (e.g., one die: 6, two dice: 36, three coins: 8).

### 2.2 The three axioms of probability

1. **Non-negativity:** P(A) ≥ 0.
2. **Certainty:** P(S) = 1.
3. **Additivity:** if A ∩ B = ∅, then P(A ∪ B) = P(A) + P(B).

**Free consequences:** P(Aᶜ) = 1 − P(A); P(∅) = 0; 0 ≤ P(A) ≤ 1.

### 2.3 Set operations and Venn picture

- **Union** A ∪ B: A or B (or both).
- **Intersection** A ∩ B: A and B.
- **Complement** Aᶜ: not A.
- **Mutually exclusive** (disjoint): A ∩ B = ∅.

### 2.4 General addition rule

P(A ∪ B) = P(A) + P(B) − P(A ∩ B).

For three events (inclusion–exclusion):

P(A ∪ B ∪ C) = P(A) + P(B) + P(C)
              − P(A ∩ B) − P(A ∩ C) − P(B ∩ C)
              + P(A ∩ B ∩ C).

> **Used directly in Dec 2025 Q6 (three weather alerts).** Memorise this formula.

### 2.5 Mutually exclusive vs Independent — the classic confusion

| Property | Mutually exclusive | Independent |
| -------- | ------------------ | ----------- |
| Definition | A ∩ B = ∅, so P(A ∩ B) = 0 | P(A ∩ B) = P(A)·P(B) |
| Plain meaning | can't both happen | don't influence each other |
| Knowing A occurred | tells you B did NOT | tells you nothing about B |
| Example | Heads vs Tails on one flip | Heads on flip 1 vs Heads on flip 2 |

**Important:** if both events have positive probability, mutually exclusive ⇒ NOT independent. They are almost opposites.

**2023 Regular Q2 trap:** "P(A∪B) = 0 because they are independent" — FALSE. Independent events still overlap; the formula is P(A∪B) = P(A) + P(B) − P(A)·P(B).

### 2.6 Independent complement rule

If A and B are independent, then so are Aᶜ and Bᶜ:
- P(Aᶜ) = 1 − P(A)
- P(Bᶜ) = 1 − P(B)
- P(Aᶜ ∩ Bᶜ) = P(Aᶜ)·P(Bᶜ) = (1 − P(A))·(1 − P(B))

> **2023 Regular Q2(c):** P(A) = 0.35, P(B) = 0.30 → P(Aᶜ ∩ Bᶜ) = 0.65 × 0.70 = 0.455.

### 2.7 The "at least one" complement trick

Whenever a question contains "at least one":

P(at least one) = 1 − P(none).

For n independent trials each with success probability p:

P(at least one success) = 1 − (1 − p)ⁿ.

> **2024 Regular Q3:** "at least 2 visits in 3 hours" used 1 − P(0) − P(1) for Poisson.

### 2.8 Worked example — set operations from a table (2024 Makeup Q1)

Given: P(M ∪ Q) = 0.40, P(M − Q) = 0.15, P(Q − M) = 0.10. Find P(M ∩ Q), P(M), P(Q).

- P(M ∪ Q) = P(M − Q) + P(Q − M) + P(M ∩ Q) ⇒ 0.40 = 0.15 + 0.10 + P(M ∩ Q) ⇒ **P(M ∩ Q) = 0.15**.
- P(M) = P(M − Q) + P(M ∩ Q) = 0.15 + 0.15 = **0.30**.
- P(Q) = P(Q − M) + P(M ∩ Q) = 0.10 + 0.15 = **0.25**.

### 2.9 Counting → probability (when outcomes equally likely)

P(A) = (favourable outcomes) / (total outcomes).

- Two dice: 36 outcomes; P(sum = 7) = 6/36 = 1/6.
- Committee selection from a group: use combinations C(n, k) = n! / [k!·(n−k)!].

> **Sample question (recurring):** "Choose 5 from 8 men + 4 women, women majority?" Answer: [C(4,3)·C(8,2) + C(4,4)·C(8,1)] / C(12,5) = (4·28 + 1·8) / 792 = 120/792 = **5/33**.

---

## HOUR 3 — CONDITIONAL & TOTAL PROBABILITY (Session 3)

The bridge from raw rules to Bayes' theorem.

### 3.1 Definition of conditional probability

P(B | A) = P(A ∩ B) / P(A),  provided P(A) > 0.

Read as "B given A". Note **P(B | A) ≠ P(A | B) in general** — different denominators, different "worlds".

### 3.2 Multiplication rule (chaining)

- Two events: P(A ∩ B) = P(A)·P(B | A) = P(B)·P(A | B).
- Three events: P(A ∩ B ∩ C) = P(A)·P(B | A)·P(C | A ∩ B).

> Use this whenever you compute "and" along a path (e.g., draws without replacement, multi-stage selection).

### 3.3 Independence — three equivalent definitions

A and B are independent if any one (and hence all) of these hold:
- P(A ∩ B) = P(A)·P(B)
- P(B | A) = P(B)
- P(A | B) = P(A)

**Independence test (recurring exam pattern):**
- Compute P(A) and P(A | B) (or P(A ∩ B) vs P(A)·P(B)).
- If equal → independent. If different → dependent.

### 3.4 With vs without replacement (drives independence)

- **With replacement** (deck/bag reset between draws): trials are independent, p constant.
- **Without replacement** (deck shrinks): trials dependent. Use multiplication rule with conditional probability.

> Two reds from {5R, 6B, 2W} (13 marbles):
> - With replacement: (5/13)·(5/13) = 25/169 ≈ 0.148.
> - Without replacement: (5/13)·(4/12) = 20/156 = 5/39 ≈ 0.128.

### 3.5 Law of total probability

If A₁, A₂, …, Aₖ partition the sample space (mutually exclusive and cover everything), then for any event B:

P(B) = Σᵢ P(B | Aᵢ)·P(Aᵢ) = P(B|A₁)P(A₁) + P(B|A₂)P(A₂) + … + P(B|Aₖ)P(Aₖ).

**The tree-diagram recipe.** Each branch = (P(case)·P(B | case)). Multiply along the branch, sum across all branches.

#### Worked example (2024 Regular Q3b — spam through 3 accounts)

P(A₁) = 0.70, P(A₂) = 0.20, P(A₃) = 0.10.
P(S | A₁) = 0.01, P(S | A₂) = 0.02, P(S | A₃) = 0.05.

P(S) = 0.01·0.70 + 0.02·0.20 + 0.05·0.10 = 0.007 + 0.004 + 0.005 = **0.016**.

#### Worked example (Jan 2026 makeup Q2 — Vice-Chancellor)

P(A) = 0.5, P(B) = 0.3, P(P) = 0.2; P(R|A) = 0.8, P(R|B) = 0.6, P(R|P) = 0.4.

P(R) = 0.8·0.5 + 0.6·0.3 + 0.4·0.2 = 0.40 + 0.18 + 0.08 = **0.66**.

### 3.6 Reading a contingency table both ways

Given a 2×k table, conditioning means **picking the right denominator (one row or column total)**.

For loan-default data with 32,219 middle-aged of which 27,368 didn't default; 38,130 total non-defaulters:
- P(no default | middle-aged) = 27368 / 32219 ≈ 0.85.
- P(middle-aged | no default) = 27368 / 38130 ≈ 0.72.

Same numerator, different denominator = different question. **Always identify the denominator first.**

---

## HOUR 4 — BAYES' THEOREM (Session 4, part 1)

A reliable 5-mark Q3. Master the recipe and you cannot lose marks here.

### 4.1 The formula

P(H | E) = P(E | H)·P(H) / P(E)

where the denominator (the "evidence") is computed by the law of total probability:

P(E) = Σᵢ P(E | Hᵢ)·P(Hᵢ).

Names of the four pieces:
- **Prior** P(H): belief before evidence.
- **Likelihood** P(E | H): how well H predicts E.
- **Evidence** P(E): normaliser.
- **Posterior** P(H | E): updated belief.

### 4.2 The 5-step recipe (works on every Bayes problem)

1. **Define events** (write each one in plain English).
2. **Write down what's given** (priors and likelihoods).
3. **Compute P(E)** via total probability.
4. **Apply Bayes:** P(H | E) = P(E | H)·P(H) / P(E).
5. **Interpret** the answer (and remark on base rate if rare event).

### 4.3 Worked example — rare disease (2024 Regular Q6, 2025 Regular Q4)

P(D) = 0.001, P(+|D) = 0.99 (sensitivity), P(+|Dᶜ) = 0.02 (false-positive).

- P(+) = 0.99·0.001 + 0.02·0.999 = 0.00099 + 0.01998 = **0.02097**.
- P(D | +) = 0.99·0.001 / 0.02097 ≈ **0.0472 (4.72%)**.

> **Key takeaway** ("base-rate fallacy"): a "99% accurate" test on a rare condition still gives a positive that is mostly *false* — because the small false-positive rate × huge healthy population swamps the few true positives.

### 4.4 Three-source attribution (the most repeated Bayes pattern)

Pattern (matches **Jan 2026 Q2, 2023 Regular Q4, 2024 Practice Q1.4**):

Given prior probabilities for k causes A₁, …, Aₖ and conditional probability of an effect E under each cause, find P(Aᵢ | E).

Algorithm:
- numerator: P(E | Aᵢ)·P(Aᵢ).
- denominator: Σⱼ P(E | Aⱼ)·P(Aⱼ).

> **Worked (Bolt machines, Practice 1.4):** P(A) = 0.40, P(B) = 0.35, P(C) = 0.25. Defective rates 0.05, 0.03, 0.02. Defective bolt found, P(came from A)?
> P(D) = 0.05·0.40 + 0.03·0.35 + 0.02·0.25 = 0.0355.
> P(A | D) = 0.05·0.40 / 0.0355 = 0.0200 / 0.0355 ≈ **0.5634 (56.34%)**.

### 4.5 Diagnostic Bayes via natural frequencies (a sanity-check trick)

Imagine a population of 1000 people with the same priors and rates:
- Disease 1 in 1000 → 1 sick, 999 healthy.
- Sensitivity 99% → 0.99 true positive.
- False-positive 5% → 49.95 false positives.
- Total positives ≈ 51, of which 1 is real → P(D | +) ≈ 1/51 ≈ 0.020 (matches the formula).

Use this trick to **sanity-check** any Bayes answer in the exam.

### 4.6 Common Bayes patterns and their phrases

| Phrase | Pattern |
| ------ | ------- |
| "test-positive given disease/no-disease" | sensitivity / (1 − specificity) |
| "rate of defects per machine" | three-source Bayes |
| "spam given the word 'offer'" | one-feature Bayes |
| "guilty given DNA match" | base-rate Bayes (Jan 2026 makeup style) |
| "research promoted given VC type" | three-source Bayes |

---

## HOUR 5 — NAÏVE BAYES CLASSIFIER (Session 4, part 2)

This is **Q2 in every recent paper** (Dec 2025, 2025 Regular, 2024 Makeup). Worth 5 marks. Mechanical once you learn the recipe.

### 5.1 The "naïve" assumption

For features X₁, …, Xₙ given class C:

P(X₁, X₂, …, Xₙ | C) = P(X₁ | C)·P(X₂ | C)·…·P(Xₙ | C).

That is, features are **conditionally independent given the class**.

### 5.2 The MAP decision rule

Choose the class y that maximises:

P(C = y) · Πᵢ P(Xᵢ | C = y).

The denominator P(X) is the same for every class so we ignore it; we just compare scores.

### 5.3 The 6-step Naïve Bayes recipe (text classification)

1. **Compute class priors:** P(C) = (#docs in class) / (#total docs).
2. **Build vocabulary V:** all unique words across the corpus. |V| = size.
3. **Count words per class:** number of occurrences of each word in each class. Total words per class = Nᶜ.
4. **Apply Laplace smoothing:** P(w | C) = (count(w, C) + α) / (Nᶜ + α·|V|), with α = 1 (default).
5. **Score each class** for the test document: P(C)·Π P(wᵢ | C).
6. **Argmax.** Higher score wins.

### 5.4 Why Laplace smoothing?

If a word never appeared with a class in training, its count is 0 and the entire product collapses to 0 (zero-frequency trap). Adding α = 1 to every count gives unseen words a small non-zero probability — distinguishing "never seen" from "impossible".

With α = 1 and |V| vocabulary words: denominator becomes (Nᶜ + |V|).

### 5.5 Worked example — Dec 2025 Q2 (Urgent vs Not Urgent)

**Training corpus:**
| Msg | Words | Label |
| --- | ----- | ----- |
| 1 | assignment due tomorrow please help | Urgent |
| 2 | lunch plans today | Not Urgent |
| 3 | exam rescheduled urgent attention | Urgent |
| 4 | thanks for the notes | Not Urgent |

**Test message:** "assignment urgent". α = 1.

Step 1 — priors: P(U) = P(N) = 2/4 = **0.5**.

Step 2 — vocabulary V = {assignment, due, tomorrow, please, help, lunch, plans, today, exam, rescheduled, urgent, attention, thanks, for, the, notes}. |V| = **16**.

Step 3 — totals: N_U = 9 words; N_N = 7 words.
- count(assignment, U) = 1; count(urgent, U) = 1.
- count(assignment, N) = 0; count(urgent, N) = 0.

Step 4 — smoothed likelihoods:
- P(assignment | U) = (1+1)/(9+16) = 2/25.
- P(urgent | U) = 2/25.
- P(assignment | N) = (0+1)/(7+16) = 1/23.
- P(urgent | N) = 1/23.

Step 5 — scores:
- Score_U = 0.5 · (2/25) · (2/25) = 0.5 · 4/625 = **0.0032**.
- Score_N = 0.5 · (1/23) · (1/23) = 0.5 / 529 ≈ **0.000945**.

Step 6 — Score_U > Score_N → **Classify as Urgent**.

### 5.6 Tabular Naïve Bayes (Weather → Play)

Same recipe, but features are categorical attributes (e.g., Outlook, Temp, Humidity, Windy). The 14-row "play tennis" dataset is a recurring exam pattern.

**Recipe:** for each class C ∈ {Yes, No}:
- Prior P(C) = (#rows of class C) / (total rows).
- For each feature value xᵢ in the test instance, compute P(xᵢ | C) = (#rows with xᵢ in class C) / (#rows of class C). Apply smoothing if any count is zero.
- Score = P(C)·Π P(xᵢ | C). Argmax.

**Tennis example** (P(Y) = 9/14, P(N) = 5/14) for test (Sunny, Hot, Normal, ¬Windy):

- Score_Yes = (9/14)·(3/9)·(2/9)·(6/9)·(6/9) ≈ 0.021.
- Score_No  = (5/14)·(3/5)·(2/5)·(1/5)·(2/5) ≈ 0.005.

→ Predict **Yes**.

### 5.7 Common Naïve Bayes pitfalls (each costs marks)

1. Forgetting to **smooth** when a count is zero.
2. Using wrong denominator (forgetting to add |V| in Laplace).
3. Not stating the **prior** explicitly.
4. Not multiplying by the **class prior** in the final score.
5. Tokenisation errors — count exactly the words in the test message. If a word in the test isn't in the vocabulary, just ignore it (or include it with smoothing if you formed V from training+test).

---

## HOUR 6 — RANDOM VARIABLES (Session 5)

This is the foundation for Q4 and Q5 (continuous RV / joint distribution / expectation).

### 6.1 Random variable

A **random variable X** is a function from the sample space to the real numbers. Two flavours:
- **Discrete**: countable values (number of heads, errors per page).
- **Continuous**: any value in an interval (height, time, voltage).

### 6.2 Discrete: PMF rules

A function f(x) = P(X = x) is a valid PMF iff:
- f(x) ≥ 0 for all x.
- Σₓ f(x) = 1.

#### Worked — Dec 2025 Q3a (find k from a PMF table)

| x | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| - | - | - | - | - | - | - | - | - |
| p(x) | 0 | k | 2k | 2k | 3k | k² | 2k² | 7k² + k |

Sum = 0 + k + 2k + 2k + 3k + k² + 2k² + (7k² + k) = 10k² + 9k = 1.

10k² + 9k − 1 = 0  ⇒  k = (−9 + √(81 + 40)) / 20 = (−9 + 11) / 20 = **1/10 = 0.1**.

(Reject negative root.)

Then:
- P(X < 6) = P(0)+P(1)+P(2)+P(3)+P(4)+P(5) = 0 + 0.1 + 0.2 + 0.2 + 0.3 + 0.01 = **0.81**.
- P(X ≥ 6) = 1 − 0.81 = **0.19**.
- P(0 < X < 5) = P(1)+P(2)+P(3)+P(4) = 0.1+0.2+0.2+0.3 = **0.8**.

### 6.3 CDF — definition and use

F(x) = P(X ≤ x) = Σ_{xᵢ ≤ x} f(xᵢ) (discrete) or ∫_{−∞}^{x} f(t) dt (continuous).

Properties: starts at 0, monotone non-decreasing, ends at 1.

**P(a < X ≤ b) = F(b) − F(a).** Useful for both discrete and continuous.

### 6.4 Expectation, variance, SD (discrete)

- **Expectation:** E[X] = μ = Σₓ x·f(x). Long-run average ("balance point").
- **E[g(X)]:** Σₓ g(x)·f(x). E.g., E[X²] = Σ x²·f(x).
- **Variance:** Var(X) = σ² = E[(X − μ)²] = E[X²] − μ². The second form (E[X²] − (E[X])²) is the **shortcut almost always used in exams**.
- **SD:** σ = √Var(X).

#### Linear transform (constantly tested)

For constants a, b:
- E[aX + b] = a·E[X] + b.
- Var(aX + b) = a²·Var(X). (constants disappear from variance)

> **2025 Regular Q5(d):** Var(2X + 3) = 4·Var(X). For Var(X) = 1 → Var(2X+3) = **4**.

#### Worked — discrete RV with PMF (2024 Makeup Q3b)

P(X = x) = k·x² for x ∈ {−1, 0, 1}.

- Sum: k(1) + k(0) + k(1) = 2k = 1 ⇒ **k = 0.5**.
- P(−1) = 0.5, P(0) = 0, P(1) = 0.5.
- E[X] = (−1)(0.5) + 0·0 + 1·0.5 = **0**.
- E[X²] = 1·0.5 + 0 + 1·0.5 = **1**.
- Var(X) = E[X²] − (E[X])² = 1 − 0 = **1**.

#### "Game-fair" check

A game is **fair** when E[gain] = 0. If E[gain] > 0, the player is favoured; if < 0, the house is favoured.

> **(Slide example):** Two dice, gain +3 if sum 7, −2 if sum 8, −4 if sum 3, else 0.
> P(7) = 6/36, P(8) = 5/36, P(3) = 2/36.
> E[gain] = (3·6 + (−2)·5 + (−4)·2)/36 = (18 − 10 − 8)/36 = 0/36 = **0** → fair.

### 6.5 Continuous RV: PDF

A continuous X is described by a density f(x) ≥ 0 with **∫_{−∞}^{∞} f(x) dx = 1**.

Probability is **area**: P(a < X < b) = ∫ₐᵇ f(x) dx.

Note P(X = a) = 0 for any single point. Hence P(X < a) = P(X ≤ a).

**E[X]** = ∫ x·f(x) dx;  **Var(X)** = ∫ (x − μ)² f(x) dx = E[X²] − (E[X])²;  **F(x)** = ∫_{−∞}^{x} f(t) dt;  **f(x) = F'(x)**.

#### Worked — piecewise PDF (2024 Makeup Q6)

f(x) = c·x² for 0 ≤ x ≤ 1, f(x) = c·(2 − x) for 1 < x ≤ 2.

(a) Normalisation:
∫₀¹ c·x² dx + ∫₁² c·(2 − x) dx = 1.
∫₀¹ x² dx = [x³/3]₀¹ = 1/3.
∫₁² (2 − x) dx = [2x − x²/2]₁² = (4 − 2) − (2 − 0.5) = 2 − 1.5 = 0.5.
So c·(1/3 + 1/2) = c·(5/6) = 1 ⇒ **c = 6/5 = 1.2**.

(b) P(0.5 < X < 1.5) = ∫_{0.5}^{1} (6/5)·x² dx + ∫_1^{1.5} (6/5)·(2−x) dx.
First piece: (6/5)·[x³/3] from 0.5 to 1 = (6/5)·(1/3 − 1/24) = (6/5)·(7/24) = 42/120 = 7/20.
Second piece: (6/5)·[2x − x²/2] from 1 to 1.5 = (6/5)·[(3 − 1.125) − (2 − 0.5)] = (6/5)·(0.375) = 9/20.
Sum = 7/20 + 9/20 = 16/20 = **4/5 = 0.8**.

(c) E[X] = ∫₀¹ x·(6/5)x² dx + ∫₁² x·(6/5)(2−x) dx
       = (6/5)·[x⁴/4]₀¹ + (6/5)·[x² − x³/3]₁²
       = (6/5)·(1/4) + (6/5)·[(4 − 8/3) − (1 − 1/3)]
       = (6/5)·(1/4) + (6/5)·(2/3)
       = 6/20 + 12/15 = 3/10 + 4/5 = 3/10 + 8/10 = 11/10
       = **1.1**.

#### Worked — uniform on a symmetric interval (2025 Regular Q4b)

f(x) = k/6 for −3 ≤ x ≤ 3, else 0.

(a) ∫_{-3}^{3} (k/6) dx = (k/6)·6 = k = 1 ⇒ **k = 1**.
(b) E[X] = ∫_{-3}^{3} x·(1/6) dx = 0 (symmetry around 0) → **E[X] = 0**.
(c) E[X²] = ∫_{-3}^{3} x²·(1/6) dx = (1/6)·[x³/3]_{-3}^{3} = (1/6)·(9 − (−9)) = 18/6 = 3.
   Var(X) = E[X²] − (E[X])² = **3**.

### 6.6 Joint, marginal, conditional distributions

For two random variables X and Y with joint PMF f(x, y) (or PDF):

- **Marginal of X:** fₓ(x) = Σᵧ f(x, y) (PMF) or ∫ f(x, y) dy (PDF).
- **Marginal of Y:** f_Y(y) = Σₓ f(x, y) (or ∫ f(x, y) dx).
- **Conditional:** f_{Y | X}(y | x) = f(x, y) / fₓ(x).
- **Independence:** X and Y independent ⇔ f(x, y) = fₓ(x)·f_Y(y) for all (x, y).
- **E[XY]:** Σ Σ xy·f(x, y) (or double integral). If independent, E[XY] = E[X]·E[Y].

#### Worked — joint PDF (Dec 2025 Q4)

f(x, y) = 6x²y on 0 ≤ x ≤ 1, 0 ≤ y ≤ 1.

(i) ∫₀¹ ∫₀¹ 6x²y dx dy = 6·(1/3)·(1/2) = **1** ✓ valid.

(ii) Marginals:
fₓ(x) = ∫₀¹ 6x²y dy = 6x²·(1/2) = **3x²**, 0 ≤ x ≤ 1.
f_Y(y) = ∫₀¹ 6x²y dx = 6y·(1/3) = **2y**, 0 ≤ y ≤ 1.

(iii) P(X < 0.5, Y > 0.5) = ∫₀^{0.5} ∫_{0.5}^{1} 6x²y dy dx
   = ∫₀^{0.5} 6x²·[y²/2]_{0.5}^{1} dx
   = ∫₀^{0.5} 6x²·(0.5 − 0.125) dx
   = ∫₀^{0.5} 6x²·(0.375) dx
   = 2.25·[x³/3]₀^{0.5} = 2.25·(0.125/3) = 2.25·0.04167 = **0.09375**.

(iv) Independence: f(x,y) = 6x²y; fₓ(x)·f_Y(y) = 3x²·2y = 6x²y ✓ → **independent**.

(v) Expectations (independence makes them easy):
E[X] = ∫₀¹ x·3x² dx = 3·[x⁴/4]₀¹ = **3/4**.
E[Y] = ∫₀¹ y·2y dy = 2·[y³/3]₀¹ = **2/3**.
E[XY] = E[X]·E[Y] = (3/4)·(2/3) = **1/2**.

---

## HOUR 7 — DISCRETE DISTRIBUTIONS: Bernoulli, Binomial, Poisson (Session 6)

The "moulds" question. Recognising which mould fits the wording is half the battle.

### 7.1 Bernoulli(p) — one trial, two outcomes

PMF: P(X = x) = pˣ·(1 − p)^(1−x), x ∈ {0, 1}.
- E[X] = p.
- Var(X) = p·q where q = 1 − p.

Variance is **maximum at p = 0.5** (= 0.25). A near-fair coin is the most unpredictable.

### 7.2 Binomial(n, p) — count successes in n independent trials

**The BINS check** — must hold for binomial:
- **B**inary outcome each trial.
- **I**ndependent trials.
- **N**umber of trials n is fixed.
- **S**ame success probability p across trials.

PMF: P(X = k) = C(n, k)·pᵏ·q^(n−k), k = 0, 1, …, n,  where C(n, k) = n! / [k!·(n − k)!].

- E[X] = np.
- Var(X) = npq.
- SD(X) = √(npq).

#### English-to-event translator (memorise)

| Phrase | Compute |
| ------ | ------- |
| "exactly k" | P(X = k) |
| "at most k" | P(X ≤ k) = Σ from 0 to k |
| "fewer than k" | P(X < k) = P(X ≤ k − 1) |
| "at least k" | 1 − P(X ≤ k − 1) |
| "no successes" | P(X = 0) = qⁿ |
| "at least one" | 1 − qⁿ |

#### Worked — restaurant water (slide Ex 1)

n = 10, p = 0.6, q = 0.4.

(i) P(X = 6) = C(10, 6)·(0.6)⁶·(0.4)⁴ = 210·0.046656·0.0256 ≈ **0.251**.
(ii) P(X < 9) = 1 − P(9) − P(10) = 1 − 0.0403 − 0.0060 = **0.954**.
(iii) P(X = 0) = (0.4)¹⁰ ≈ **0.0001**.
(iv) P(X ≤ 2) = 0.0001 + 0.0016 + 0.0106 ≈ **0.0123**.
(v) P(X ≥ 3) = 1 − 0.0123 = **0.9877**.

#### Reverse engineering (find n, p from mean and variance)

Given mean = np and variance = npq:
- q = variance / mean.
- p = 1 − q.
- n = mean / p.

> **Slide P10:** mean 6, variance 4 ⇒ q = 4/6 = 2/3, p = 1/3, n = 6/(1/3) = 18 ⇒ **Bin(18, 1/3)**.

> **2023 Makeup Q2:** μ = np = 4, σ = 2/√3, σ² = 4/3 = npq ⇒ q = 1/3, p = 2/3, n = 6 ⇒ **Bin(6, 2/3)**. P(6 < X < 9) = 0 (since X ≤ 6).

### 7.3 Poisson(λ) — rare events in a window

PMF: P(X = k) = e^(−λ)·λᵏ / k!, k = 0, 1, 2, …
- E[X] = λ; Var(X) = λ (mean = variance).

**The window rule (the most common Poisson trap):**
λ = rate × window size.

> "2 calls per 3 minutes, 9-minute window" ⇒ λ = (2/3)·9 = 6.
> "5 defects per 10 sq ft, 15-sq-ft sheet" ⇒ λ = (5/10)·15 = 7.5.

#### Worked — switchboard (slide Ex 5)

Calls at 2 per 3 minutes; P(≥ 5 in 9 minutes)?

λ = 6.
P(X ≥ 5) = 1 − P(0) − P(1) − P(2) − P(3) − P(4)
        = 1 − e⁻⁶·(1 + 6 + 18 + 36 + 54)
        = 1 − e⁻⁶·115
        = 1 − 0.2851 = **0.7149**.

#### Worked — defective surgical blades (Dec 2025 Q3b)

0.2% defective; 10 blades per packet; 10,000 packets.

For one packet: λ = n·p = 10·0.002 = **0.02**.

(i) P(X = 0) = e^(−0.02) ≈ 0.9802 → in 10000 packets: ≈ **9802 packets**.
(ii) P(X = 1) = 0.02·e^(−0.02) ≈ 0.0196 → ≈ **196 packets**.

### 7.4 Poisson approximation to Binomial

When n ≥ 100 and p ≤ 0.01 (rule of thumb: n large, p small, np moderate):
**Bin(n, p) ≈ Poisson(λ = np).**

This collapses ten/hundred-term sums into a one-line formula.

> **2024 Regular Q3a:** λ = 0.2 per foot. P(≥ 2 in 1 ft) = 1 − P(0) − P(1) = 1 − e^(−0.2)·(1 + 0.2) = 1 − 0.9824 = **0.0176**.

> **2025 Regular Q2b:** Bin(40, 0.15) → Poisson(λ = 6) is a poor fit because p = 0.15 is too large. Exact Bin: P(X = 5) ≈ 0.1692; Poisson approx: 0.1606. **Approximation breaks down** when p is not small.

> **Jan 2026 Q6a (cheque errors):** rate 1.2 per 100 cheques; 250 cheques → λ = 250·0.012 = **3**. P(X ≤ 2) = e⁻³·(1 + 3 + 4.5) = 8.5·e⁻³ ≈ **0.4232**. P(X > 4) = 1 − P(X ≤ 4) = 1 − e⁻³·(1 + 3 + 4.5 + 4.5 + 3.375) ≈ 1 − 0.8153 = **0.1847**.

### 7.5 Worked — machines breaking down (2024 Makeup Q5)

8% of machines break down per year, 25 chosen ⇒ Poisson approx with λ = 25·0.08 = 2.

(i) P(X = 5) = e⁻²·2⁵ / 5! = e⁻²·(32/120) ≈ **0.036**.
(ii) P(X ≥ 4) = 1 − e⁻²·(1 + 2 + 2 + 4/3) = 1 − 0.857 = **0.143**.
(iii) P(3 ≤ X ≤ 8) = e⁻² · Σ_{k=3}^{8} 2ᵏ/k! ≈ **0.323**.

---

## HOUR 8 — NORMAL DISTRIBUTION & APPROXIMATIONS (Session 6)

### 8.1 The Normal — N(μ, σ²)

PDF: f(x) = (1 / (σ·√(2π)))·exp(−(x − μ)² / (2σ²)), for −∞ < x < ∞.

Properties:
- Symmetric about μ. Mean = median = mode = μ.
- Inflection points at μ ± σ.
- Total area = 1.

### 8.2 The 68-95-99.7 rule (free probability calculator)

- ~68% of values within μ ± 1σ.
- ~95% within μ ± 2σ.
- ~99.7% within μ ± 3σ.

Use this for a **sanity check** of any z-score answer.

### 8.3 Standardisation

Z = (X − μ) / σ ~ N(0, 1).

After standardising, look up F(z) = P(Z ≤ z) in the standard-normal table.

**Two table tricks you must know:**
- P(Z > z) = 1 − F(z).
- F(−z) = 1 − F(z).

### 8.4 The 5-step Normal recipe

1. Sketch the bell, mark μ, mark x.
2. Identify the **shaded** region the question asks for.
3. Convert each x to z = (x − μ) / σ.
4. Look up F(z) from the table.
5. Assemble using the two table tricks.

### 8.5 Worked — washers (Dec 2025 Q5)

X ~ N(0.500, 0.005²). Acceptable: 0.492 ≤ X ≤ 0.506.

z₁ = (0.492 − 0.500) / 0.005 = **−1.6**.
z₂ = (0.506 − 0.500) / 0.005 = **1.2**.

P(0.492 ≤ X ≤ 0.506) = F(1.2) − F(−1.6) = 0.8849 − 0.0548 = **0.8301**.
P(defective) = 1 − 0.8301 = **0.1699 ≈ 17%**.

→ Process not capable; defect rate too high; reduce variation, recalibrate.

### 8.6 Worked — exam scores (Jan 2026 Q3, recurring)

X ~ N(73, 8²) (variance = 64, σ = 8).

(i) P(X < 91): z = (91 − 73)/8 = 2.25 ⇒ F(2.25) ≈ **0.9878**. → 98.78%, "most" is justified.
(ii) P(65 < X < 89): z₁ = −1, z₂ = 2 ⇒ F(2) − F(−1) = 0.9773 − 0.1587 = **0.8186** (~82%).
(iii) Dean's List = top 5%: find z s.t. F(z) = 0.95 → z ≈ 1.645. Cut-off = 73 + 1.645·8 ≈ **86.16**.

### 8.7 Percentile inversion (X from probability)

Given P(X < x) = p, find x:
- Look up z such that F(z) = p in the table.
- x = μ + z·σ.

Standard z values to memorise:
- z₀.₉₅ = 1.645 (right-tail 5%) ⇒ 90th percentile = z = 1.28.
- z₀.₉₇₅ = 1.96 (95% CI two-sided).
- z₀.₉₉ = 2.33.
- z₀.₉₉₅ = 2.575.

### 8.8 Worked — light-bulb / instrument life (2025 Regular Q2a, 2023 Makeup Q6)

X ~ N(12, 2²).
- P(X < 7): z = (7 − 12)/2 = −2.5 → F(−2.5) = 1 − F(2.5) = 1 − 0.9938 = **0.0062**.
- P(7 < X < 12): z₁ = −2.5, z₂ = 0 → F(0) − F(−2.5) = 0.5 − 0.0062 = **0.4938**.

### 8.9 Normal approximation to Binomial

Conditions: **np ≥ 15 AND nq ≥ 15** (some texts say ≥ 10; use the stricter rule).

X ~ Bin(n, p) ≈ N(μ = np, σ² = npq) **with continuity correction**.

#### Continuity correction — half-step rule

Discrete bars are 1 unit wide; the bar for k spans k − 0.5 to k + 0.5. So:

| Discrete | Continuous |
| -------- | ---------- |
| P(X = k) | P(k − 0.5 < X < k + 0.5) |
| P(X ≤ k) | P(X < k + 0.5) |
| P(X < k) | P(X < k − 0.5) |
| P(X ≥ k) | P(X > k − 0.5) |
| P(X > k) | P(X > k + 0.5) |

Forgetting the half-step is the most common 1-mark slip on this question type.

#### Worked — Bin(150, 0.2) (2023 Regular Q7)

n = 150, p = 0.2 → μ = 30, σ = √(150·0.2·0.8) = √24 ≈ 4.899.

P(50 < X < 80) → P(50.5 < X < 79.5) using continuity.

z₁ = (50.5 − 30)/4.899 ≈ 4.19; z₂ = (79.5 − 30)/4.899 ≈ 10.10.
F(10.10) ≈ 1, F(4.19) ≈ 1 ⇒ probability ≈ 1 − 1 = **0**.

#### Worked — college plans (slide Ex 15)

n = 50, p = 0.31. np = 15.5 ≥ 15 ✓, nq = 34.5 ≥ 15 ✓.
μ = 15.5, σ = √(50·0.31·0.69) ≈ 3.27.

P(X < 14) → P(X < 13.5).
z = (13.5 − 15.5)/3.27 = −0.61 ⇒ **F(−0.61) ≈ 0.2709**.

---

## HOUR 9 — SAMPLING DISTRIBUTIONS, CLT, CIs (Session 7, supplementary)

CS-7 is "secondary focus" but Sampling/CLT shows up in 2023 Q7–Q8 and a couple of past papers, so cover the essentials.

### 9.1 Population vs sample, parameter vs statistic

| Quantity | Population (Greek) | Sample (Roman) |
| -------- | ------------------ | -------------- |
| Mean | μ | x̄ |
| Variance | σ² | s² |
| Proportion | p | p̂ |

Population parameter is fixed (often unknown); sample statistic is computed from data and varies sample-to-sample.

### 9.2 Sampling methods

- **Simple random sample (SRS):** every unit equal chance.
- **Systematic:** pick every k-th unit after a random start (k = N/n).
- **Stratified:** sample within each stratum, then combine. Allocate proportionally:
  nᵢ = n · (Nᵢ / N).
- **Cluster:** sample whole groups (e.g., schools, then all students).
- **Multistage:** combinations of the above.

> **Slide example — stratified allocation:** strata sizes 2000, 6000, 1000 (N = 9000), n = 1500. Allocations: 333, 1000, 167.

### 9.3 Central Limit Theorem (CLT)

For a random sample of size n from a population with mean μ and finite variance σ²:

x̄ ~ N(μ, σ²/n)  approximately,  for n ≥ 30 (any n if the population itself is normal).

The quantity σ/√n is the **standard error** (SE) of the mean.

### 9.4 Z-score for a sample mean

z = (x̄ − μ) / (σ/√n).

**The most common 1-mark slip:** using σ instead of σ/√n. If the question is about an **average of n** observations, you must divide σ by √n.

### 9.5 Worked — vehicle ages (slide Ex 1, recurring)

μ = 96, σ = 16, n = 36. P(90 < x̄ < 100)?

SE = 16/√36 = 16/6 ≈ 2.667.
z₁ = (90 − 96)/2.667 = −2.25; z₂ = (100 − 96)/2.667 = 1.50.
P = F(1.50) − F(−2.25) = 0.9332 − 0.0122 = **0.9210**.

### 9.6 Worked — adult heights (2024 Makeup Q1b)

μ = 70, σ = 3, n = 49. P(69 < x̄ < 71)?

SE = 3/√49 = 3/7 ≈ 0.4286.
z₁ = (69 − 70)/0.4286 ≈ −2.33; z₂ = (71 − 70)/0.4286 ≈ 2.33.
P = F(2.33) − F(−2.33) = 0.9901 − 0.0099 = **0.9802**.

### 9.7 Finite-population correction (FPC)

When n / N > 5%:

SE(x̄) = (σ/√n) · √((N − n) / (N − 1)).

The factor is ≤ 1 (smaller pond ⇒ better estimate). Skip the correction if n/N ≤ 5%.

### 9.8 Confidence intervals (intuition only — sometimes asked)

For μ with σ known:
- 90% CI: x̄ ± 1.645·(σ/√n).
- 95% CI: x̄ ± 1.96·(σ/√n).
- 98% CI: x̄ ± 2.33·(σ/√n).
- 99% CI: x̄ ± 2.575·(σ/√n).

For σ unknown (use sample s and t-table with df = n − 1):
- x̄ ± t_{n−1, α/2}·(s/√n).

**Interpretation:** "We are C% confident the true μ lies in the interval" means **the procedure** captures μ in C% of all samples — not "μ has a C% chance of being in this specific interval".

### 9.9 Sample-size determination

For a target margin of error E (with σ known):

n = (z_{α/2} · σ / E)².  Always round **up** to the next integer.

> **2024 Makeup Q3a:** σ = 14, E = 2.5, 99% (z = 2.576).
> n = (2.576·14 / 2.5)² = (14.4256)² ≈ 208.1 → **n = 209**.

### 9.10 Sampling distribution of a proportion

p̂ ≈ N(p, p·q / n), valid when np > 15 and nq > 15.

z = (p̂ − p) / √(p·q / n).

Useful for polling problems ("what's the probability sample proportion exceeds 55%?").

---

## HOUR 10 — EXAM-DAY STRATEGY + 1-PAGE CHEAT SHEET

### 10.1 The 2-hour battle plan

| Minute | What you do |
| ------ | ----------- |
| 0–3 | Read all 6 questions. Tick the easy ones (Q1 always; whichever of Q2/Q3 looks shorter; Q4 if it's a discrete PMF). |
| 3–25 | **Q1 (Descriptive Stats).** Sort, 5-number summary, IQR, fence, outlier, mean/SD, skew comment. Banks 4–5 marks. |
| 25–50 | **Q2 (Naïve Bayes or Probability rules).** Write priors, vocab, smoothed likelihoods, scores, classify. |
| 50–75 | **Q3 (Bayes / total probability).** Define events, list given probs, compute P(E), apply formula, interpret. |
| 75–95 | **Q4 (Random variable).** Find k, P(...), E[X], Var(X). For joint PMF: marginals, independence, conditional. |
| 95–110 | **Q5 (Continuous RV / Normal).** Verify total area = 1, find P(a<X<b) via integration or z-table. |
| 110–115 | **Q6 (Distribution: Bin / Poisson / Normal approx).** Identify mould, name parameters, write event, compute. |
| 115–120 | **Review + sanity-check** every answer (probabilities in [0,1], variance ≥ 0, totals sum to 1). |

### 10.2 The 10 things you MUST write to score full marks

1. **Define every event** in plain English at the start.
2. **State the given numerical values** before substituting.
3. **Show the formula** before plugging numbers.
4. **State the rule/method** (e.g., "By Bayes' theorem", "Using the law of total probability").
5. **Show every step of substitution** — the markers grade per step.
6. **Round only at the final step** to 3–4 decimal places.
7. **State the **conclusion** in plain English** ("So the probability is ≈ 0.17, meaning 17% of …").
8. **Sanity check:** is the answer in [0, 1]? Is variance ≥ 0?
9. **Sketch a Venn / tree / bell** diagram if it clarifies — even rough sketches earn marks.
10. **Underline / box** your final answer.

### 10.3 The 7 traps the exam loves

1. **σ vs σ/√n:** "individual" → σ; "sample mean of n" → σ/√n.
2. **Continuity correction** (Normal-to-Binomial): ±0.5 half-step.
3. **Window rescale (Poisson):** λ = rate × window — recompute λ to the asked window.
4. **Inclusive vs exclusive bounds:** "fewer than 9" excludes 9 → P(X ≤ 8). "At most 2" includes 2 → P(X ≤ 2).
5. **Laplace smoothing** in Naïve Bayes: add 1 to every count, add |V| to denominator. Skipping it can make scores zero.
6. **Sample size** — always **round up** ("245.9 → 246 employees").
7. **"Mutually exclusive" ≠ "independent":** they're opposites for events with positive probability.

### 10.4 Ten-line Bayes template (write this on every Bayes Q)

```
Let A = ...
Let B = ...
Given: P(A) = ..., P(B|A) = ..., P(B|Aᶜ) = ...
By total probability:
  P(B) = P(B|A)·P(A) + P(B|Aᶜ)·P(Aᶜ)
       = ...
By Bayes' theorem:
  P(A|B) = P(B|A)·P(A) / P(B)
         = ...
Conclusion: ≈ ... %.
```

### 10.5 ONE-PAGE CHEAT SHEET

**Descriptive stats**
- Sample mean ȳ = Σyᵢ/n; sample s² = Σ(yᵢ − ȳ)²/(n−1).
- 5-number = (Min, Q1, Median, Q3, Max); IQR = Q3 − Q1.
- Fences: Q1 − 1.5·IQR, Q3 + 1.5·IQR.
- Skew: mean > median ⇒ positive skew; mean < median ⇒ negative skew.

**Probability rules**
- P(A ∪ B) = P(A) + P(B) − P(A ∩ B).
- P(A ∪ B ∪ C) = ΣP − ΣP(pairs) + P(triple).
- Mutually exclusive: P(A ∩ B) = 0.
- Independent: P(A ∩ B) = P(A)·P(B).
- Complement: P(Aᶜ) = 1 − P(A).
- "At least one": 1 − P(none).

**Conditional & Bayes**
- P(B | A) = P(A ∩ B)/P(A).
- Multiplication: P(A ∩ B) = P(A)·P(B | A).
- Total prob: P(B) = ΣP(B | Aᵢ)·P(Aᵢ).
- Bayes: P(A | B) = P(B | A)·P(A) / P(B).

**Naïve Bayes (text)**
- P(C) = #docs in class / total.
- P(w | C) = (count(w, C) + 1) / (Nᶜ + |V|).
- Score(C) = P(C)·Π P(wᵢ | C). Argmax.

**Discrete RV**
- Σ P(x) = 1.
- E[X] = Σ x·P(x); E[X²] = Σ x²·P(x).
- Var(X) = E[X²] − (E[X])².
- E[aX + b] = a·E[X] + b; Var(aX + b) = a²·Var(X).

**Continuous RV**
- ∫f(x) dx = 1; P(a < X < b) = ∫ₐᵇ f(x) dx.
- f(x) = F'(x); F(x) = ∫_{−∞}^{x} f(t) dt.
- E[X] = ∫ x·f(x) dx; Var(X) = ∫ x²·f(x) dx − μ².

**Joint distributions**
- fₓ(x) = Σᵧ f(x, y) (or ∫); f_Y(y) = Σₓ f(x, y).
- f(y | x) = f(x, y)/fₓ(x).
- Independent: f(x, y) = fₓ(x)·f_Y(y) ∀(x, y).
- E[XY] = ΣΣ xy·f(x, y); independent ⇒ E[XY] = E[X]·E[Y].

**Distribution moulds**
| Mould | Use when | E[X] | Var(X) |
| ----- | -------- | ---- | ------ |
| Bernoulli(p) | one trial | p | pq |
| Bin(n, p) | n indep trials, fixed p | np | npq |
| Poisson(λ) | rare events in window | λ | λ |
| N(μ, σ²) | continuous, sums of effects | μ | σ² |

**Approximations**
- Bin → Poisson: n ≥ 100, np ≤ 10. λ = np.
- Bin → Normal: np ≥ 15 AND nq ≥ 15. μ = np, σ = √(npq), use ±0.5.
- Poisson window: λ = rate × window size.

**Standard normal**
- Z = (X − μ)/σ.
- P(Z > z) = 1 − F(z); F(−z) = 1 − F(z).
- 68-95-99.7 rule.

**Sampling & CLT**
- x̄ ~ N(μ, σ²/n), n ≥ 30 (or population normal).
- z = (x̄ − μ)/(σ/√n).
- FPC factor √((N − n)/(N − 1)) when n/N > 5%.

**Sample size & CIs**
- n = (z·σ/E)², round UP.
- 95% CI: x̄ ± 1.96·(σ/√n).
- 99% CI: x̄ ± 2.575·(σ/√n).
- σ unknown → use t with df = n − 1.

**Z-table key values** (one-tailed)

| z | F(z) |
| - | ---- |
| 1.00 | 0.8413 |
| 1.25 | 0.8944 |
| 1.50 | 0.9332 |
| 1.645 | 0.9500 |
| 1.96 | 0.9750 |
| 2.00 | 0.9773 |
| 2.33 | 0.9901 |
| 2.50 | 0.9938 |
| 2.575 | 0.9950 |
| 2.75 | 0.9970 |
| 3.00 | 0.9986 |

By symmetry, F(−z) = 1 − F(z).

---

## Final words

1. Hour 1–4 of your prep: nail Sessions 1–4 (Q1, Q2, Q3 ≈ 15 marks, half the paper).
2. Hour 5–8: drill RV / distribution problems (Q4–Q6 ≈ 15 marks).
3. Hour 9: practice 1–2 mixed past papers under timed conditions.
4. Hour 10: review this cheat sheet, copy it onto a single sheet of paper from memory.

You will walk into the exam having already worked every problem type the paper asks. Trust the procedure, write neatly, and write the recipe steps even on questions you "see immediately" — the steps are where the partial credit lives.

**Good luck. Stir well. Stick the recipe.**




