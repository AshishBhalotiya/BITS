# ML MidSem — Final Scoring Guide (10-hour study plan)

> **Course**: AIMLCZG565 Machine Learning
> **Scope**: Modules 1–5 (Intro · Workflow · Linear Regression · Linear Classification · Decision Tree)
> **Duration**: 2 hours, Closed Book, ~30% weightage
> **Pattern**: 6 questions × 4–8 marks each (covering Intro, Workflow, Regression, Classification, Eval Metrics, Decision Tree)

This guide is built from official handouts, companion notes (CS1–CS6), and **every past midsem paper from 2022–2026 plus practice sets**. It mixes plain-English intuition, must-know formulas, fully worked numericals, and exam-day tactics. Read top-to-bottom in order — each section builds on the previous.

---

## 0. How to read this guide (15 minutes)

**Time budget for 10 hours**:

| Block | Module | Time | Why |
|---|---|---|---|
| 1 | Module 1 — Intro | 45 min | Easy marks; scenario IDs always asked |
| 2 | Module 2 — Workflow & Data | 1.5 hr | "Spot 5 quality issues" is a guaranteed Q |
| 3 | Module 3 — Linear Regression | 2.0 hr | Numerical GD step + bias/variance theory |
| 4 | Module 4a — Classification basics | 1.0 hr | Decision boundary, generative vs discriminative |
| 5 | Module 4b — Logistic Regression | 1.5 hr | Sigmoid + log-loss + GD numerical |
| 6 | Module 4c — Evaluation Metrics | 1.0 hr | Confusion matrix, P/R/F1 always 4–6 marks |
| 7 | Module 5 — Decision Tree | 2.0 hr | Entropy + IG numerical (always 5–8 marks) |
| 8 | Final review + question bank | 30 min | Skim, then attempt the bank |

**The exam pattern is fixed**: one question per module + one on metrics. Lock in the templates below and you can reach 25/30 even without panic-memorising every detail.

---

## 1. MODULE 1 — Introduction to Machine Learning

### 1.1 The single sentence to memorise (Tom Mitchell's definition)

> "A computer program is said to **learn** from experience **E** with respect to some task **T** and some performance measure **P**, if its performance on T, as measured by P, improves with experience E."

**Three pieces** — every "identify E, T, P" question is just labelling these three:

| Piece | Plain word | Example (spam filter) |
|---|---|---|
| **T** — Task | what we want done | label e-mail as spam / not spam |
| **P** — Performance measure | how we score it | % labelled correctly |
| **E** — Experience | the examples it learns from | old labelled e-mails |

**Exam tip**: If asked "name 2 P measures", pick from this short list:
- **Classification**: Accuracy, Precision, Recall, F1, AUC-ROC, Misclassification rate
- **Regression**: MSE, RMSE, MAE, R²

### 1.2 The three families of learning (memorise the trigger words)

```
Supervised        — labels available    → predict answers
   ├─ Regression       (numeric output: price, temperature, rainfall)
   └─ Classification   (categorical output: spam/not, sunny/rainy/windy)

Unsupervised      — no labels           → find structure
   ├─ Clustering       (group similar items)
   └─ Dimensionality reduction (PCA)

Reinforcement     — rewards/penalties   → learn from trial & error
                                          (self-driving car, game-playing bot)
```

**Trigger-word table for "identify the learning type"** questions (asked in EVERY paper):

| Phrase in question | Family |
|---|---|
| "labelled data", "past records with answers" | Supervised |
| "predict a category / class / label" | Supervised → Classification |
| "predict a number / amount / price / temperature" | Supervised → Regression |
| "no labels", "find hidden patterns / groups", "cluster" | Unsupervised |
| "learns from feedback / rewards / interactions / trial and error" | Reinforcement |
| "self-driving car", "game-playing", "robot navigation" | Reinforcement |

**Worked example** (2024 Reg Q1a pattern): For each scenario, name the family.

| Scenario | Answer |
|---|---|
| Self-driving car learns by interacting with environment | Reinforcement |
| Predict review sentiment (pos/neg/neutral) from labelled data | Supervised — Classification |
| Cluster DNA sequences without labels | Unsupervised — Clustering |
| Predict insurance claim amount from history | Supervised — Regression |
| Predict tomorrow's rainfall in mm | Supervised — Regression |
| Forecast Sunny/Windy/Rainy from past data | Supervised — Classification |

### 1.3 AI vs ML vs Deep Learning

```
   Artificial Intelligence (the dream of smart machines)
       └── Machine Learning  (one way: learn from data)
              └── Deep Learning (ML using big neural networks)
```

DL ⊂ ML ⊂ AI. Don't confuse the order in the exam.

### 1.4 Discriminative vs Generative classifiers (1–2 mark question)

| Aspect | **Discriminative** | **Generative** |
|---|---|---|
| What it models | P(Y \| X) directly — only the boundary | P(X, Y) — full joint distribution |
| Trains on | features → label | how each class generates the data |
| Example methods | Logistic regression, SVM, Decision Tree | Naive Bayes, GMM, LDA |
| Useful for | direct classification | density estimation, outlier/anomaly detection |
| Analogy | learns the giveaway differences | learns each language fully, then asks "which is more likely?" |

**Exam Q-pattern**: "Job applications + outlier detection via density estimation" → **Generative** (need P(X\|Y) for density).

### 1.5 Designing a learning system (Mitchell's 4 questions)

For any "design a learning system to do X":
1. **What examples will it learn from?** (Type & label availability of training data.)
2. **What exactly should it learn to predict?** (The target function.)
3. **How will the prediction be represented?** (Linear weighted sum, tree, etc.)
4. **How will it improve?** (Update rule — gradient descent.)

Running example: "Program that learns to play checkers" → board examples → score board → linear weighted sum of features → adjust weights.

### 1.6 Why ML is hard (Challenges — 1 mark phrasings)

- Too few examples (data hunger).
- Biased / unrepresentative samples.
- Messy data: missing, duplicates, outliers, typos.
- Useless / redundant features (curse of dimensionality).
- Underfitting (too simple) vs overfitting (too wiggly).
- Training–serving skew (different distributions in production).

**Mnemonic**: **DUMB-GO** — Data scarcity, Unrepresentative sample, Missing/dirty data, Bad features, Generalisation (over/underfitting), Operational skew.

---

## 2. MODULE 2 — ML Workflow and Data Preprocessing

### 2.1 The 6-step ML workflow (one-liner answer)

```
1. Gather data  →  2. Clean & explore  →  3. Pick a model
                                              ↓
6. Evaluate & iterate ←  5. Tune hyperparameters  ←  4. Train it
```

It's a **loop**, not a line. ~80% of real work lives in steps 1–2.

### 2.2 Types of attributes (decides what maths is allowed)

| Type | Plain meaning | Examples | What you can do |
|---|---|---|---|
| **Nominal** (names) | unordered labels | colour, city, ID | =, ≠ only |
| **Ordinal** (ranks) | ordered, uneven gaps | small/medium/large, grades | <, > also valid |
| **Interval** | even gaps, no true zero | temperature °C, calendar dates | + and − OK; ratios meaningless |
| **Ratio** | even gaps, true zero | height, weight, count, price | + − × ÷ all valid |

**Common pitfall**: 20°C is NOT "twice as hot" as 10°C (interval, no true zero). 20 kg IS twice 10 kg (ratio).

### 2.3 ⭐ Spotting Data Quality Issues (GUARANTEED 4–6 mark question)

The "find at least 5 issues in this messy table" question appears in **every** paper. The 6 standard categories to look for:

| # | Issue | Tell-tale sign | Fix |
|---|---|---|---|
| 1 | **Missing values** | empty cells, NaN | impute (mean / median / mode) or drop |
| 2 | **Duplicates** | same row twice | de-duplicate on key columns |
| 3 | **Outliers** | "Age = 350", crazy CGPA | IQR test or 3σ test, then cap / remove |
| 4 | **Inconsistent format** | dates "15-Aug-2022" vs "15/08/2022", "AI" vs "Artificial Intelligence" | standardize / mapping table |
| 5 | **Wrong units** | weight = 120 lbs, others in kg | unit conversion |
| 6 | **Invalid values / typos** | email "dev[at]mail.com", blood group "Iam+ve" | regex validation |
| 7 | **Wrong / mixed encoding** | covid result = 1 in one row, "Negative" in another | data smoothing / re-encode |
| 8 | **Redundant columns** | both Age and DOB present | drop redundant column |

**Template answer for the messy-table question**:
```
1. <issue>: <evidence from row Tx>. Resolve by <technique>.
2. ... etc.
```
Aim for 5–6 issues × 2 marks each.

### 2.4 Outlier detection — two formulas

**Method 1: IQR (Inter-Quartile Range)** — works on any distribution.
- Sort values, find Q1 (25th percentile), Q3 (75th percentile).
- IQR = Q3 − Q1.
- Lower fence = Q1 − 1.5·IQR. Upper fence = Q3 + 1.5·IQR.
- Anything outside the fences is an outlier.

**Worked example**: data = {2, 10, 11, 11, 11, 12, 13, 14, 14, 15, 17, 22}.
Q1 = 11, Q3 = 14.5, IQR = 3.5.
Lower = 11 − 1.5(3.5) = 5.75. Upper = 14.5 + 1.5(3.5) = 19.75.
Outliers: **2** (too small), **22** (too big).

**Method 2: 3-sigma rule** — assumes bell-shaped (normal) data.
- Compute mean μ and standard deviation σ.
- Outlier if value > μ + 3σ or < μ − 3σ (lies beyond 3 typical steps).

**Pitfall**: 3σ rule fails on **skewed** data (income, transactions). Use IQR there.

### 2.5 ⭐ Missing value imputation

| Strategy | When to use |
|---|---|
| **Mean** | numeric, symmetric (no outliers) |
| **Median** | numeric, skewed or has outliers |
| **Mode** | categorical |
| Drop row | only if missingness is rare (<5%) |
| Forward-fill | time-series |
| Predictive imputation (kNN, regression) | sophisticated; needs validation |

**Why mode is bad for unique numerics**: if all known values are different (e.g. 7, 5, 8, 4), mode is undefined / arbitrary → use mean or median.

### 2.6 ⭐ Feature scaling (always-asked theory + numerical)

**Why we scale**: when features have very different ranges (Age = 30, Income = 500,000), the larger feature drowns out the smaller. Distance-based and gradient-based methods break.

**Two formulas to memorise**:

**Min-Max Normalization** (squeeze to [0, 1]):
```
v' = (v − min) / (max − min)
```
Smallest value → 0, largest → 1, others linearly between.

**Z-score Standardization** (centre and rescale by σ):
```
v' = (v − μ) / σ
```
After z-scoring: mean = 0, std = 1.

**⭐ Worked example (Dec 2025 Q1b)** — Z-score for Total Runs of 5 teams:
Data: 820, 840, 860, 880, 900.
- μ = (820 + 840 + 860 + 880 + 900) / 5 = 4300 / 5 = **860**.
- Variance = ((−40)² + (−20)² + 0² + 20² + 40²) / 5 = (1600 + 400 + 0 + 400 + 1600) / 5 = 4000/5 = 800.
- σ = √800 ≈ **28.28**.

Normalized values:
- 820 → (820 − 860)/28.28 = −1.41
- 840 → −0.71
- 860 → 0
- 880 → +0.71
- 900 → +1.41 ✓

**⭐ Critical Common Pitfall**:
> Compute scaling parameters (μ, σ, min, max) **only on training data**, then apply same parameters to test data. Otherwise → **data leakage** → over-optimistic test scores.

### 2.7 Feature scaling — True/False statements (asked verbatim in 2024 Mkp Q2)

| Statement | T/F | Why |
|---|---|---|
| Scaling guarantees gradient descent reaches **global** minimum | **False** | Non-convex losses (NN, log-reg with non-convex reg) can have local minima |
| Scaling prevents X^T X from being singular | **False** | Singularity is about linear dependence, not scale |
| Scaling is **necessary** to prevent GD from local optima | **False** | Linear regression's MSE has no local optima — it's convex |
| Scaling **speeds up** gradient descent (fewer iterations) | **True** | Round contour → straight downhill path; long ravines → zigzag |

### 2.8 Sampling — keep it fair, keep it balanced

**Stratified sampling** is the safe default for classification:
- Preserves the class proportions of the population in your train / test / validation splits.
- Critical for imbalanced data (e.g. fraud at 0.1%).

**Split ratios** (common defaults): 70/15/15 or 80/10/10 (train/val/test).

**Three piles**:
- **Train**: model learns on this.
- **Validation**: tune hyper-parameters here.
- **Test**: locked away till the very end; touch once for the final report.

**Why keep the test set untouched** (1.5-mark Q): "If you peek at test during training/tuning, you've leaked the answers — your reported score will be over-optimistic. Reserve it for final fair evaluation."

**Imbalanced data (rare positive class)** fix list:
- Oversample the minority (SMOTE — synthetic samples).
- Undersample the majority.
- Class weights in the loss function.
- Use stratified split.
- Use precision/recall/F1/AUC instead of accuracy.

### 2.9 Feature engineering — four levers

1. **Selection** — drop useless / repeated / mostly-blank columns.
2. **Construction** — combine raw facts ("days as customer = last_visit − first_visit").
3. **Extraction (PCA)** — squash many columns into a few summary columns. Mitigates the **curse of dimensionality**.
4. **Transformation** — bucket continuous → categorical (age → child/adult/senior), one-hot encode categorical → numeric.

**One-hot vs ordinal encoding pitfall** (⭐):
- For unordered categories (Red, Green, Blue) **one-hot encode** — give each its own 0/1 column.
- Encoding Red=0, Green=1, Blue=2 wrongly tells the model "Blue > Green > Red".
- For ordered categories (small, medium, large) ordinal 0,1,2 is fine.

### 2.10 Curse of dimensionality (1-mark canned answer)

> "As feature count grows, data becomes sparse — the 'distance' between any two points becomes nearly the same, so 'nearest-neighbour' loses meaning. Patterns get diluted. Mitigations: feature selection, PCA, regularisation."

---

## 3. MODULE 3 — Linear Regression

### 3.1 The model in plain English

Predict a **number** by drawing the best line:
```
ŷ = θ₀ + θ₁ x₁ + θ₂ x₂ + … + θₙ xₙ
```
- ŷ — predicted value (price, score)
- xⱼ — j-th feature (size, age, BMI…)
- θ₀ — intercept (bias / starting value when all x = 0)
- θⱼ — weight (rate at which feature j influences ŷ)

"**Linear**" here means: multiply each fact by a weight, then add. No squaring of weights, no products of features. (Squaring **features** is fine — that's polynomial regression, see §3.10.)

In matrix form: **ŷ = Xθ**, where X has a column of 1s for the bias.

### 3.2 ⭐ Cost function: Mean Squared Error (MSE)

Take the gap (residual) on every example, square it (so + and − don't cancel), then average:

```
J(θ) = (1/2m) · Σᵢ (ŷᵢ − yᵢ)²
     = (1/2m) · Σᵢ (θᵀxᵢ − yᵢ)²
```

(The 1/2 is a convenience — it cancels the 2 from differentiation.)

Why squared? (a) Penalises sign-cancellation, (b) penalises **big** errors much more than small ones, (c) gives a smooth bowl with a unique minimum (convex).

**Spring analogy**: each data point is tied to the line by a vertical spring; the line settles where total spring energy (sum of squared stretches) is minimum.

### 3.3 ⭐ Way 1 — Closed-form (Normal Equations)

There is a one-shot formula for the best θ:

```
θ = (XᵀX)⁻¹ Xᵀ y
```

**Dimensions** (asked in 2023 Reg Q6a):
- m = number of training examples, n = number of features.
- After adding the bias column, X is **m × (n+1)**.
- y is **m × 1**.
- θ is **(n+1) × 1**.

**Worked dimension check**: m = 28 examples, n = 4 features → X is **28 × 5**, y is **28 × 1**, θ is **5 × 1**.

**When closed-form wins**: small/medium n (few features). It's exact and needs no learning rate.
**When closed-form fails**:
- Very large n (computing (XᵀX)⁻¹ is roughly O(n³) — slow when n is hundreds of thousands).
- XᵀX is non-invertible (when features are linearly dependent / m < n).
- Then switch to Way 2: Gradient Descent.

**⭐ Exam phrasing** (2023 Reg Q2b): "n=2,000,000 instances, m=300,000 features → use **gradient descent**, because computing matrix inverse is O(n³) which is infeasible."

### 3.4 ⭐ Way 2 — Gradient Descent (THE numerical question)

Walk downhill on the MSE bowl, one small step at a time:

```
Repeat until convergence:
   for each weight θⱼ:
      θⱼ ← θⱼ − α · (1/m) · Σᵢ (ŷᵢ − yᵢ) · xᵢⱼ
```

In words: **new weight = old weight − step × average (error × feature)**.

- α = learning rate (step size). Typical: 0.001, 0.01, 0.1.
- Repeat for many iterations until J stops decreasing.

**Three flavours** (memorise the differences):

| Flavour | How much data per step | Behaviour |
|---|---|---|
| **Batch GD** | All m examples | Smooth steps, slow per epoch on big data |
| **Stochastic GD (SGD)** | 1 example | Fast and noisy; never quite settles |
| **Mini-batch GD** | small batch (32, 64…) | Sweet spot — what everyone actually uses |

**Learning-rate troubleshooting** (1-mark Q):
- α too **small** → J decreases but very slowly (many iterations needed).
- α too **large** → J **oscillates or increases** — algorithm diverges.
- Verdict from "J(θ) is increasing for 15 iterations with α = 0.3" → **decrease α** (try 0.1).

### 3.5 ⭐⭐ FULLY WORKED gradient descent (one iteration, batch GD)

This template covers the question in CS3 companion notes and 2024-Reg patterns. Heart-disease risk = w₀ + w₁·BMI + w₂·Pressure. Initial w = (5, −0.03, −0.03), α = 0.02, m = 3.

| Patient | BMI | Pressure | y (real) |
|---|---|---|---|
| 1 | 35 | 80 | 1.81 |
| 2 | 25 | 80 | 1.22 |
| 3 | 30 | 100 | 1.71 |

**Step 1 — Predictions and gaps** (gap = ŷ − y):
- ŷ₁ = 5 − 0.03(35) − 0.03(80) = 5 − 1.05 − 2.40 = 1.55, gap = 1.55 − 1.81 = **−0.26**
- ŷ₂ = 5 − 0.03(25) − 0.03(80) = 5 − 0.75 − 2.40 = 1.85, gap = 1.85 − 1.22 = **+0.63**
- ŷ₃ = 5 − 0.03(30) − 0.03(100) = 5 − 0.90 − 3.00 = 1.10, gap = 1.10 − 1.71 = **−0.61**

**Step 2 — Sum (error × feature) for each weight**:
- For w₀ (bias, x₀ = 1): S₀ = −0.26 + 0.63 − 0.61 = **−0.24**
- For w₁ (BMI): S₁ = (−0.26)(35) + (0.63)(25) + (−0.61)(30) = −9.10 + 15.75 − 18.30 = **−11.65**
- For w₂ (Pressure): S₂ = (−0.26)(80) + (0.63)(80) + (−0.61)(100) = −20.80 + 50.40 − 61.00 = **−31.40**

**Step 3 — Update each weight**: nudge = (α/m) × S = (0.02/3) × S = 0.006667 × S.
- w₀ = 5 − 0.006667(−0.24) = **5.0016**
- w₁ = −0.03 − 0.006667(−11.65) = −0.03 + 0.0777 = **0.0477**
- w₂ = −0.03 − 0.006667(−31.40) = −0.03 + 0.2093 = **0.1793**

**New line**: ŷ = 5.0016 + 0.0477·BMI + 0.1793·Pressure. ✓

**What just happened**: w₀ barely moved (already about right); both rate weights flipped from negative to positive — the model corrected itself in one step.

### 3.6 ⭐ Bias–Variance Trade-off (2-3 mark theory)

Total Error = Bias² + Variance + Irreducible Noise

| | High Bias | High Variance |
|---|---|---|
| **Plain word** | underfitting — too simple | overfitting — too wiggly |
| **Symptom on data** | Train error AND test error both **high & similar** | Train error **low**, test error **high** (big gap) |
| **Cause** | model too simple, too few features | model too complex, too many features, fits noise |
| **Fix** | add features, increase model complexity, reduce λ | regularize, more data, fewer features, early stopping |

**⭐ Diagnostic rule** (asked verbatim in 2023 Reg Q2a, 2024 Reg Q3):
- "Train RMSE = $45,000, Test RMSE = $47,000 (small gap, both high)" → **High bias / underfitting**.
- "Train RMSE = $12,000, Test RMSE = $68,000 (huge gap)" → **High variance / overfitting**.
- "Train RMSE and Validation RMSE both low and close" → **Just right**.

**For Ridge with high training and high validation error** (both equal): high bias → **decrease λ** (let the weights breathe).
**For overfitting**: **increase λ** (penalise big weights more).

### 3.7 ⭐ Regularization — L1 (Lasso) vs L2 (Ridge)

Add a penalty for big weights to the cost. The hyperparameter **λ** controls how strong the penalty is.

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty term | λ Σ θⱼ² | λ Σ \|θⱼ\| |
| Cost function | MSE + λ Σ θⱼ² | MSE + λ Σ \|θⱼ\| |
| Effect on weights | shrinks all weights toward 0 (none reach exactly 0) | can drive some weights **exactly to 0** |
| Best when | many small-but-relevant features | many features, some are useless (built-in feature selection) |
| Geometric picture | round contour | diamond contour (corners = sparse solutions) |

**Elastic Net** = mix of L1 + L2 (sometimes asked).

**Effect of λ**:
- λ = 0 → plain regression (no penalty, may overfit)
- λ → ∞ → all weights shrink to 0 (severe underfitting)
- Sweet λ: pick by cross-validation.

**⭐ Use cases**:
- "High-dimensional data with many irrelevant predictors" → **Lasso** (it zeroes out useless ones).
- "Penalize large coefficients but **don't drive any to zero**" → **Ridge**.

**⭐ Effect on coefficients**:
- Ridge keeps all features (just smaller weights).
- Lasso zeros some out — explains why "Model D used only 23 of 120 features after Lasso" in 2024-Reg Q3.

### 3.8 Closed-form vs Gradient Descent — comparison table (1-2 mark Q)

| | Closed-form | Gradient Descent |
|---|---|---|
| Speed for few features | Fast (one shot) | Slow (many iterations) |
| Speed for many features | Slow (matrix inverse) | Fast (per-iter cost is linear) |
| Hyperparameter | None | Need to tune α |
| Need feature scaling | No | Yes (or it crawls) |
| Always finds optimum | Yes (linear reg has unique minimum) | Yes for convex MSE; may diverge if α too big |

### 3.9 R² — the regression report card

```
R² = 1 − (SS_res / SS_tot)
```
- SS_res = Σ (yᵢ − ŷᵢ)² (sum of squared errors)
- SS_tot = Σ (yᵢ − ȳ)² (variance of y)

Interpretation:
- R² = 1 → perfect fit (line explains all variance).
- R² = 0 → line no better than predicting mean.
- R² < 0 → line worse than predicting mean.

In words: "fraction of variance in y explained by the model".

### 3.10 Polynomial / basis function regression (curves come free)

To bend a straight line into a curve, **add new features** that are non-linear functions of the originals: x², x³, x₁·x₂, log(x), …

```
ŷ = θ₀ + θ₁ x + θ₂ x² + θ₃ x³
```
The model is still **linear in θ** (so all the same machinery works), but it now traces a curve.

**Trade-off**: high-degree polynomials → **overfit** (chase noise). Use cross-validation to pick the degree, or apply regularization (Ridge / Lasso).

### 3.11 Feature interpretation (1-mark questions, very common)

For ŷ = 50 + 200·area + 5·age:
- "200 means each unit (m²) of area adds 200 to the price, holding age constant."
- "5 is small — age has a much weaker effect than area."

**Unit change pitfall**: if area is converted from m² to ft², the **coefficient changes** (1 m² ≈ 10.76 ft²) but the **model behaviour stays the same** — the relationship is identical, only the numerical θ rescales.

---

## 4. MODULE 4 (PART 1) — Linear Models for Classification

### 4.1 From regression to classification

- **Regression** (Module 3): output is a **number** → draw the line **through** the dots.
- **Classification** (Module 4): output is a **class** → draw the line **between** the dots.

The line is called the **decision boundary**; the regions on either side are **decision regions**.

### 4.2 Linear vs non-linear decision boundaries

- **Linear classifier** — boundary is a straight line / flat hyperplane. Examples: Logistic Regression, SVM (linear kernel), LDA.
- **Non-linear classifier** — boundary is a curve / surface. Examples: Decision Tree, kNN, kernel SVM, neural net.

When does a straight line work? When classes sit in **separate clumps** with a clear gap (linearly separable). When does it fail? When one class wraps around the other (e.g. a ring of pluses around a core of circles, like XOR).

**Trick**: feed in squared / interaction features (x², x₁·x₂) and a "linear" classifier traces a **curve** in the original space — that's the kernel trick in disguise.

### 4.3 Discriminant function — the sign rule

A linear classifier just does multiply-and-add to compute a **score**:
```
score = w₀ + w₁ x₁ + w₂ x₂ + … + wₙ xₙ = wᵀx
```
Then:
- score > 0 → class +1
- score < 0 → class −1
- score = 0 → on the boundary

**Worked example**: classifier score = 0.4 + 0.3·CGPA − 0.45·IQ. Candidate (CGPA = 5, IQ = 6):
score = 0.4 + 1.5 − 2.7 = **−0.8** → "no job" side.

### 4.4 Why plain linear regression makes a bad classifier

Two problems:
1. A line can shoot above 1 or below 0 — but a "yes/no probability" must stay in [0, 1].
2. One far-away **obvious** positive dot drags the whole line and shifts the 0.5 cut-off — a once-correct point becomes mislabelled.

**The fix** (next section): pass the score through an S-shaped curve — the sigmoid.

### 4.5 Best-guess (Bayes) rule for overlapping classes

When class distributions overlap:
- **Best decision rule**: predict the class with the **higher posterior probability** at that point.
- The optimal boundary is where the two distributions **cross**.
- Sliding the boundary away from the crossing increases overall error (one exception: when one mistake is far costlier — e.g. missing a disease — then we deliberately bias the threshold).

---

## 5. MODULE 4 (PART 2) — Logistic Regression

### 5.1 The S-curve (sigmoid) — why it's the right fix

Pass the score through:
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

Properties:
- σ(very positive z) ≈ 1 ("almost certainly yes")
- σ(0) = 0.5 (coin-flip — exactly on the boundary)
- σ(very negative z) ≈ 0 ("almost certainly no")
- σ(z) ∈ (0, 1) always — a valid probability.
- σ'(z) = σ(z)·(1 − σ(z)) — handy derivative.

**Logistic regression** = wrap multiply-and-add in the sigmoid:
```
P(y = 1 | x) = h(x) = σ(wᵀx) = 1 / (1 + exp(−wᵀx))
```

### 5.2 Why the boundary is still straight

The model says "yes" when h(x) > 0.5. Sigmoid crosses 0.5 exactly when wᵀx = 0. So:
> Predict yes ⟺ chance > 0.5 ⟺ score > 0 ⟺ wᵀx > 0.

The boundary is the set of points where wᵀx = 0 — a straight line / flat hyperplane. The sigmoid only adjusts **confidence**; it doesn't bend the boundary. To get a curvy boundary, feed in squared features.

### 5.3 ⭐ Effect of weight magnitude on sigmoid (2024-Reg Q4)

Plot σ(wx) for w ∈ {1, 10, 100}:
- **w = 1**: smooth gradual S, transitions slowly from 0 to 1.
- **w = 10**: steeper S, sharp transition near x = 0.
- **w = 100**: almost a step function.

**Why this leads to overfitting**:
- Big weights → very steep sigmoid → small input changes cause big probability swings.
- Model becomes **overconfident** (predictions cluster near 0 or 1 even on borderline cases).
- Loses the smooth "probabilistic" interpretation — behaves like a hard threshold.
- Fits training noise too closely, generalises poorly.

**This is exactly why we regularise** (Ridge / Lasso) logistic regression — to keep weights small.

### 5.4 ⭐ Log-odds (logit) interpretation — coefficients are readable

Define **odds** = p / (1 − p). For p = 0.75 → odds = 3 ("3 to 1 on").

Algebra on the sigmoid gives:
```
log(p / (1 − p)) = wᵀx = w₀ + w₁ x₁ + w₂ x₂ + …
```

So logistic regression is **linear in the log-odds**, not in the probability.

**What each weight means**: increasing xⱼ by 1 unit adds wⱼ to the log-odds, which multiplies the odds by **e^wⱼ**.

**Worked example** (CS5 companion): if w(CGPA) = 0.3, then e^0.3 ≈ 1.35. Each extra CGPA point multiplies the odds by 1.35 — a 35% bump in odds, holding everything else fixed.

**⭐ Critical clarification (2024-Mkp Q5)**: "Large positive θⱼ does NOT guarantee selection". Why? Because:
- Logistic regression models log-odds, **not direct decisions**.
- Final P depends on θ₀ (bias), other θ's, and value ranges of other features.
- Even a high CGPA can be overruled by a strong negative weight on another feature.
- Selection is based on **combined weighted features**, not one alone.

### 5.5 ⭐ Cost function — Log-loss (Cross-Entropy)

Why not reuse MSE from regression? Because squaring inside the sigmoid creates a **lumpy / non-convex** surface — gradient descent gets stuck. Log-loss is **convex** for logistic regression — single bowl, unique minimum.

**Per-example penalty** = "−log(probability the model gave to the true answer)":
- True = 1, predicted h ≈ 1 → −log(1) = 0 (no penalty, confident & right)
- True = 1, predicted h ≈ 0.5 → −log(0.5) ≈ 0.69 (medium penalty)
- True = 1, predicted h ≈ 0 → −log(0) → ∞ (huge penalty, confident & wrong)

**Total log-loss (binary cross-entropy)**:
```
J(θ) = − (1/m) · Σᵢ [ yᵢ log(hᵢ) + (1 − yᵢ) log(1 − hᵢ) ]
```
where hᵢ = σ(θᵀxᵢ), and yᵢ ∈ {0, 1}.

The yᵢ acts as a switch: when y = 1, only the first term fires; when y = 0, only the second.

### 5.6 ⭐ Logistic Regression Gradient Descent

Beautifully, the update rule is **identical** to linear regression — only h(x) is now the sigmoid:
```
θⱼ ← θⱼ − α · (1/m) · Σᵢ (hᵢ − yᵢ) · xᵢⱼ
```
"Nudge each weight against its average miss." Same sentence. The "miss" is now (predicted probability − true label).

**Vector form**:
```
∇θ J = (1/m) · Xᵀ (h − y)
θ ← θ − α · ∇θ J
```

### 5.7 ⭐⭐ ONE-ITERATION Logistic Regression (Dec 2025 Q1c — full template)

**Setup**: 4 weights θ = [0.1, 0.1, 0.1, 0.2] (bias + 3 features), α = 0.1, m = 5 teams.
After Z-score normalization (from §2.6), 5 examples have features (x₁, x₂, x₃) and labels y.

**Step 1 — Compute z = θᵀx for each example**.
**Step 2 — Compute h = σ(z) for each example**.
**Step 3 — Compute the average gradient**:
```
∂J/∂θⱼ = (1/m) · Σᵢ (hᵢ − yᵢ) · xᵢⱼ
```
**Step 4 — Update each θⱼ ← θⱼ − α · (∂J/∂θⱼ)**.

For Dec 2025 the gradients came out as:
- ∂J/∂θ₀ = 0.065
- ∂J/∂θ₁ = −0.1946
- ∂J/∂θ₂ = 0.197
- ∂J/∂θ₃ = 0.197

So updates:
- θ₀ = 0.1 − 0.1(0.065) = **0.0935**
- θ₁ = 0.1 − 0.1(−0.1946) = **0.1195**
- θ₂ = 0.1 − 0.1(0.197) = **0.0803**
- θ₃ = 0.2 − 0.1(0.197) = **0.1803**

**Compute prediction for Team D** (x₁ = 8 raw, normalized x₂ = 0.71, x₃ = 0.71):
z = 0.0935 + 0.1195(8) + 0.0803(0.71) + 0.1803(0.71)
  = 0.0935 + 0.956 + 0.057 + 0.128 = **1.2342**
h = 1 / (1 + e^(−1.2342)) ≈ **0.7746**

**Interpretation**: Team D has a ~77.5% probability of qualifying. ✓

### 5.8 Multi-class extensions

| Approach | How it works |
|---|---|
| **One-vs-Rest** (OvR) | Train one binary classifier per class ("is it cat or not?", "is it dog or not?", …). At test time, pick the class with the highest score. |
| **Softmax (Multinomial)** | Direct generalization of sigmoid: K parallel scores, then softmax = exp(zₖ) / Σⱼ exp(zⱼ). Outputs sum to 1 — a clean probability over all classes. |

**Softmax formula**:
```
P(y = k | x) = exp(wₖᵀ x) / Σⱼ exp(wⱼᵀ x)
```

---

## 6. MODULE 4 (PART 3) — Evaluation Metrics for Classification

This is its own 4–6 mark question in EVERY paper.

### 6.1 ⭐ Confusion Matrix

|  | Predicted **Positive** | Predicted **Negative** |
|---|---|---|
| **Actual Positive** | TP (caught it) | FN (missed it) |
| **Actual Negative** | FP (false alarm) | TN (correctly cleared) |

- TP = positives correctly predicted positive.
- FN = positives wrongly predicted negative ("misses").
- FP = negatives wrongly predicted positive ("false alarms").
- TN = negatives correctly predicted negative.

Total = TP + FN + FP + TN = N.

### 6.2 ⭐ Core Metrics — formulas to memorise

| Metric | Formula | Plain meaning |
|---|---|---|
| **Accuracy** | (TP + TN) / N | overall correctness |
| **Precision** | TP / (TP + FP) | when it says yes, how often is it right? |
| **Recall** (Sensitivity, TPR) | TP / (TP + FN) | of all real positives, how many caught? |
| **Specificity** (TNR) | TN / (TN + FP) | of all real negatives, how many correctly cleared? |
| **F1-score** | 2·P·R / (P + R) | harmonic mean — balances P & R |
| **FPR** | FP / (FP + TN) | of real negatives, how many wrongly flagged? |
| **Misclassification** | (FP + FN) / N | = 1 − Accuracy |

**Why harmonic mean for F1?** Because if either P or R is near 0, F1 also drops near 0 — it punishes imbalance.

### 6.3 Three guaranteed sub-questions

**(a) Compute the metrics from a given confusion matrix.**

⭐ Worked example (Jan 2026 Q4, medical diagnosis): TP=120, FN=30, FP=85, TN=765 (1000 patients):
- Accuracy = (120 + 765) / 1000 = **88.5%**
- Precision = 120 / (120 + 85) = 120 / 205 ≈ **58.5%**
- Recall = 120 / (120 + 30) = 120 / 150 = **80.0%**
- F1 = 2 · (0.585 · 0.80) / (0.585 + 0.80) = 0.936 / 1.385 ≈ **67.6%**
- Specificity = 765 / (765 + 85) = 765 / 850 = **90.0%**

**(b) Pick the metric that matters here.**
- "Imbalanced data, want to catch all positives" → **Recall**.
- "Want few false alarms" → **Precision**.
- "Care about both equally" → **F1**.
- "Compare two models across thresholds" → **AUC-ROC** (or **AUC-PR** for very imbalanced).

**(c) Cost-aware threshold tuning.**

⭐ Worked example (Jan 2026 Q4c, 1000 patients): cost(FN) = $50,000, cost(FP) = $2,000.
- Cost of FN: 30 × $50,000 = $1,500,000
- Cost of FP: 85 × $2,000 = $170,000
- **Total** = $1,670,000

**Recommendation**: cost(FN) is **25× higher** than cost(FP) → **lower the threshold** to catch more positives (recall goes up at the expense of more false alarms).

### 6.4 ⭐ Accuracy paradox — why accuracy ALONE is misleading

Imbalanced data: 990 non-fraud + 10 fraud. A "predict everything not-fraud" dummy gets 99% accuracy but catches **zero** frauds. **Recall and AUC** tell the truth.

**⭐ Canned phrasing for "Model A 95% acc vs Model B 84% acc, which is better?"** (2024-Reg Q3):
> "Accuracy alone is misleading on imbalanced data (only 8% approved loans here). Model A may simply be predicting all-rejected. Without precision, recall, F1 and confusion matrix, we can't decide. **Model B may actually be better** if it identifies the minority class more effectively. Need to evaluate using **Precision, Recall, F1, and AUC-ROC** for the minority class."

### 6.5 Per-class metrics (asked in 2023-Reg Q4 / 2023-Reg-2 Q4)

You can compute precision/recall **with respect to the negative class** by re-labelling it as the positive:
- Precision_neg = TN / (TN + FN)
- Recall_neg = TN / (TN + FP)

For a confusion matrix with TP=350 (Pos=Yes), FN=150, FP=200, TN=800:
- Pos: P = 350/(350+200) = **63.6%**, R = 350/(350+150) = **70.0%**, F1 ≈ **66.6%**
- Neg: P = 800/(800+150) = **84.2%**, R = 800/(800+200) = **80.0%**, F1 ≈ **82.0%**

### 6.6 When precision matters > recall, and vice-versa

| Scenario | Optimize | Why |
|---|---|---|
| Spam filter (don't lose real mail) | **Precision** | a real e-mail in spam folder is unforgivable |
| Cancer screening / fraud detection | **Recall** | missing a positive case is catastrophic |
| Search engine top-10 results | **Precision** | users only see top-10; quality matters |
| Disease outbreak case-finding | **Recall** | better false alarms than missed cases |

### 6.7 ROC and AUC

- **ROC curve** plots TPR (recall) vs FPR for **every threshold**.
- **AUC** = area under the ROC curve.
  - AUC = 1 → perfect.
  - AUC = 0.5 → coin flip.
  - AUC < 0.5 → worse than random (just flip the labels!).
- Useful when you want a **threshold-independent** comparison.
- For very imbalanced data, **AUC-PR** (precision-recall curve) is more informative than AUC-ROC.

### 6.8 Effect of threshold on metrics (1.5 mark Q)

Lower threshold → predict "yes" more often → recall ↑, precision ↓, FPR ↑.
Raise threshold → predict "yes" less often → precision ↑, recall ↓, FPR ↓.

So **yes** — Precision, Recall, Sensitivity, Specificity all change with the threshold. Show one calculation:
> "With threshold 0.5, TP=40, FP=10, FN=20 → Precision = 40/50 = 0.80. With threshold 0.7, fewer get classified positive — say TP=30, FP=4, FN=30 → Precision = 30/34 ≈ 0.88, Recall = 30/60 = 0.50. Both shifted."

---

## 7. MODULE 5 — Decision Trees

### 7.1 The big idea

A **flowchart** the computer builds for you. Each internal node asks one yes/no-ish question; each leaf gives a final answer. To classify a new example, walk down the answers.

**The strategy** (Twenty Questions): at each node, pick the **question that clears the most confusion** about the answer.

### 7.2 ⭐ Entropy — measure of confusion

For a binary label (yes / no):
```
H(S) = − p_yes · log₂(p_yes) − p_no · log₂(p_no)
```

For K classes:
```
H(S) = − Σₖ p_k · log₂(p_k)
```

**Quick checks**:
- All same class (all yes or all no): H = 0 (zero confusion — pure).
- 50-50 mix: H = 1 bit (maximum confusion for 2 classes).
- Uniform over K classes: H = log₂(K).

**Useful values to memorise**:
- −log₂(0.5) = 1
- −log₂(1/3) ≈ 1.585
- −log₂(1/4) = 2
- −log₂(1/8) = 3

**Quick log₂ values for fractions** (3-class, 4-class):
- log₂(2/3) ≈ −0.585, log₂(1/3) ≈ −1.585
- log₂(3/4) ≈ −0.415, log₂(1/4) = −2
- log₂(5/14) ≈ −1.485, log₂(9/14) ≈ −0.637

**Tennis-data sanity check** (9 yes, 5 no):
H = −(9/14)·log₂(9/14) − (5/14)·log₂(5/14) ≈ −(0.643)(−0.637) − (0.357)(−1.485) ≈ 0.410 + 0.531 = **0.94 bits** ✓

### 7.3 Information Gain (IG) — the splitting rule

When splitting set S on attribute A into child sets S₁, S₂, …, Sᵥ:

```
IG(S, A) = H(S) − Σᵥ (|Sᵥ| / |S|) · H(Sᵥ)
                ↑              ↑
           "before"     "after" (weighted average over children)
```

**Greedy rule (ID3)**: at each node, pick the attribute with **largest IG**.

**⭐ #1 Mistake**: forgetting the weight |Sᵥ|/|S| — each child must be weighted by how many examples landed in it.

### 7.4 ⭐⭐ Tennis dataset — full ID3 (Mitchell's classic)

14 days, 9 play / 5 don't. Parent entropy H(S) = 0.94 bits.

**Test attribute Wind** (8 weak, 6 strong):
- Weak (8 days, 6/2): H = −(6/8)log₂(6/8) − (2/8)log₂(2/8) ≈ 0.811
- Strong (6 days, 3/3): H = −0.5(−1) − 0.5(−1) = 1.000
- Weighted = (8/14)(0.811) + (6/14)(1.000) ≈ 0.464 + 0.429 = **0.892**
- IG(Wind) = 0.94 − 0.892 = **0.048**

**Test Humidity** (7 high, 7 normal):
- High (7 days, 3/4): H ≈ 0.985
- Normal (7 days, 6/1): H ≈ 0.592
- Weighted = (7/14)(0.985) + (7/14)(0.592) ≈ **0.789**
- IG(Humidity) = 0.94 − 0.789 = **0.151**

**Test Outlook** (5 sunny, 4 overcast, 5 rain):
- Sunny (5 days, 2/3): H ≈ 0.971
- Overcast (4 days, 4/0): H = **0** (already pure)
- Rain (5 days, 3/2): H ≈ 0.971
- Weighted = (5/14)(0.971) + (4/14)(0) + (5/14)(0.971) ≈ **0.694**
- IG(Outlook) = 0.94 − 0.694 = **0.246**

**Test Temperature** (4 hot, 6 mild, 4 cool): IG ≈ **0.029**.

**Final ranking**: Outlook (0.246) > Humidity (0.151) > Wind (0.048) > Temp (0.029).

→ **Root = Outlook**. Recurse on each branch:
- **Overcast** branch → already pure (4/4 yes) → leaf "Play".
- **Sunny** branch (2/3) → split on **Humidity** (high → don't, normal → play, both pure).
- **Rain** branch (3/2) → split on **Wind** (strong → don't, weak → play, both pure).

### 7.5 ⭐ Other classic ID3 problem (Jan 2026 Q2)

Loan dataset: 14 samples, 9 Repay, 5 Default. Attribute Income has High/Low.
- Income = High: 6 Repay, 2 Default → H ≈ −(6/8)log₂(6/8) − (2/8)log₂(2/8) ≈ **0.811**
- Income = Low: 3 Repay, 3 Default → H = **1.000**
- Weighted = (8/14)(0.811) + (6/14)(1.000) ≈ **0.892**
- Parent H ≈ **0.940**
- **IG = 0.940 − 0.892 ≈ 0.048** → "Income Level is a poor split (gain almost zero)."

### 7.6 ⭐ Computing IG for a 3-class problem (2023 Reg Q5)

Performance is one of {Excellent, Good, Poor}. 15 students, 5/5/5 → uniform.

Parent H = −(1/3)log₂(1/3) × 3 = **log₂(3) ≈ 1.585 bits** (with base-2).

For each attribute, group rows by attribute value, compute child entropies, take weighted average, subtract from parent. Pick attribute with **maximum IG** (= **minimum** weighted-child entropy).

**Note from 2023 Reg-2 Q5a**: When K=3 classes some answer keys allow log base 3 normalization (max H = 1 instead of 1.585). The relative ranking of attributes is unchanged.

### 7.7 ⭐ Limitation of IG: bias toward many-valued attributes

IG **over-rewards** attributes with many distinct values. Extreme case: an "ID" or "Date" attribute splits the data into N pure singletons → IG looks huge → totally useless model (memorisation).

**Fix: Gain Ratio** (used in C4.5):
```
SplitInfo(A) = − Σᵥ (|Sᵥ|/|S|) · log₂(|Sᵥ|/|S|)
GainRatio(S, A) = IG(S, A) / SplitInfo(A)
```

SplitInfo is just **entropy of the split sizes** — it grows with the number of children, so it discounts attributes that fragment too much.

### 7.8 Gini Impurity (CART trees)

Alternative to entropy:
```
Gini(S) = 1 − Σₖ p_k²
```

- All-same: Gini = 0 (pure)
- 50-50: Gini = 1 − (0.25 + 0.25) = 0.5 (most impure for 2 classes)

Same story as entropy — used by CART because it's cheaper to compute (no log).

### 7.9 ⭐ Continuous attributes (C4.5 — asked verbatim in practice Q5)

How to split on TestScore = 55, 60, 75, 80, 85, 95?

1. **Sort** by attribute.
2. **Find candidate thresholds**: midpoints between adjacent values where the **class label changes**.
   - 55 (No) → 60 (No) → 75 (Yes): change between 60 and 75 → T₁ = 67.5
   - 75 (Yes) → 80 (Yes) → 85 (No): change between 80 and 85 → T₂ = 82.5
   - 85 (No) → 95 (Yes): change → T₃ = 90
3. For each threshold, create a binary split (≤ T, > T) and compute IG.
4. Pick threshold with **highest IG**.

For TestScore example, IG(T₁=67.5) = 0.459, IG(T₂=82.5) = 0, IG(T₃=90) = 0.191 → **best split: TestScore ≤ 67.5**.

### 7.10 Missing values handling (C4.5 way)

- **Training**: distribute the missing-attribute examples **fractionally** down each branch in proportion to known-value examples. Compute IG using fractional counts.
- **Prediction**: send the example down all branches; final prediction is a weighted average of leaf predictions.

Other simple options:
- Fill with mode (categorical) / mean / median (numeric).
- Treat "missing" as its own category.

### 7.11 ⭐ Overfitting in Decision Trees

A fully grown DT can memorise the training set perfectly → performs poorly on new data.

**Two cures**:

| Strategy | What it does | Trade-off |
|---|---|---|
| **Pre-pruning** (early stopping) | Stop splitting if depth > D, or node has < N samples, or IG < ε | Fast; may stop too early on XOR-like patterns |
| **Post-pruning** | Grow full tree → prune branches that don't help on validation set | Usually better; XOR-friendly |

**Reduced-error pruning**: walk bottom-up, replace subtree with majority leaf if it improves validation accuracy.

**Cost-complexity pruning** (CART): add α × |leaves| penalty to the loss; tune α by cross-validation.

### 7.12 ⭐ MDL Principle (Minimum Description Length)

> Prefer the model that lets you describe the data **most briefly**:
>
> total cost = (cost to write down the tree) + (cost to write down the leftover errors)
>
> Smaller tree → simpler description but more errors. Bigger tree → more errors fixed but tree itself is longer to encode. The MDL-optimal tree balances these.

This is Occam's razor for trees, a principled way to avoid overfitting. (It's the rationale behind cost-complexity pruning.)

### 7.13 ⭐ Robustness comparison — Linear vs Logistic vs Decision Tree (Jan 2026 Q6)

This question asks how each model reacts to **(A) noisy/irrelevant features** and **(B) outliers**.

**Dataset A — Noisy / Irrelevant Features Added**:

| Model | Affected? | Why |
|---|---|---|
| Linear Regression | Affected | Fits a non-zero weight to every feature; spurious correlations distort. |
| Logistic Regression | Affected | Same — noisy features distort the decision boundary. |
| Decision Tree | **Strongly affected** | Greedy splits on any small impurity reduction → branches that fit random quirks → overfits. |

**Dataset B — Outliers Introduced**:

| Model | Affected? | Why |
|---|---|---|
| Linear Regression | **Strongly affected** | MSE squares errors → extreme outliers pull the line dramatically. |
| Logistic Regression | Less affected | Sigmoid saturates near 0/1 → extreme points don't dominate. |
| Decision Tree | Moderately–strongly affected | Outliers can cause unusual splits / extra branches; locally overfits. |

**Memorise pattern**:
- DT: terrible with noise, manageable with outliers.
- Linear Regression: terrible with outliers, manageable with noise.
- Logistic Regression: better in both because of the sigmoid saturation.

### 7.14 Pros / Cons of Decision Trees (one-mark phrasings)

| ✅ Strengths | ❌ Weaknesses |
|---|---|
| Easy to read like rules ("white-box") | Greedy — may miss the globally best tree |
| No need to scale features | Unstable — small data changes can re-shape the whole tree |
| Handles mixed types and missing values | Boundaries are axis-parallel rectangles only (no slants) |
| Fast prediction | Overfit if not pruned |

**The instability fix**: average many trees (Random Forest, Module 9 — out of midsem scope but worth mentioning).

### 7.15 Why training-set accuracy alone is detrimental (1-2 mark Q)

> "A fully-grown DT can hit 100% training accuracy by memorising every example, including noise. That number says nothing about generalisation. Always evaluate on a **held-out validation/test set** that the tree never saw."

---

## 8. ⭐ Quick-reference cheat sheets (memorise night-before)

### 8.1 Formulas at a glance

```
LINEAR REGRESSION
  Model:        ŷ = θᵀx
  Cost (MSE):   J = (1/2m) Σ (ŷᵢ − yᵢ)²
  Closed form:  θ = (XᵀX)⁻¹ Xᵀy
  GD update:    θⱼ ← θⱼ − α·(1/m) Σ(ŷᵢ − yᵢ) xᵢⱼ
  R²:           1 − SSres/SStot

LOGISTIC REGRESSION
  Sigmoid:      σ(z) = 1 / (1 + e⁻ᶻ)
  Model:        h(x) = σ(θᵀx)
  Cost (BCE):   J = −(1/m) Σ [y log h + (1−y) log(1−h)]
  GD update:    θⱼ ← θⱼ − α·(1/m) Σ(hᵢ − yᵢ) xᵢⱼ   (same form!)
  Log-odds:     log(p/(1−p)) = θᵀx
  Odds change:  exp(θⱼ) per unit increase in xⱼ

REGULARIZATION
  Ridge (L2):   J + λ Σ θⱼ²        (shrinks all)
  Lasso (L1):   J + λ Σ |θⱼ|       (zeros some — sparsity)

CLASSIFICATION METRICS
  Accuracy   = (TP + TN) / N
  Precision  = TP / (TP + FP)
  Recall     = TP / (TP + FN)
  Specificity= TN / (TN + FP)
  F1         = 2·P·R / (P + R)
  FPR        = FP / (FP + TN)

DECISION TREES
  Entropy:    H(S) = − Σ p_k log₂(p_k)
  Info Gain:  IG(S, A) = H(S) − Σ (|Sᵥ|/|S|) H(Sᵥ)
  Gain Ratio: IG / SplitInfo
  SplitInfo:  − Σ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)
  Gini:       1 − Σ p_k²
```

### 8.2 Common log₂ values

| x | log₂(x) | x | log₂(x) |
|---|---|---|---|
| 1/16 | −4 | 1/2 | −1 |
| 1/8 | −3 | 1 | 0 |
| 1/4 | −2 | 2 | 1 |
| 1/3 | −1.585 | 3 | 1.585 |
| 1/5 | −2.322 | 4 | 2 |
| 2/3 | −0.585 | 5 | 2.322 |
| 3/4 | −0.415 | 8 | 3 |

### 8.3 Decision-tree numerical template (use this in the exam)

```
Step 1: Parent entropy H(S)
    Count classes in S → p_k = countₖ/|S|
    H(S) = − Σ p_k log₂(p_k)

Step 2: For each candidate attribute A:
    a) Partition S into Sᵥ (one per value of A)
    b) For each Sᵥ, compute H(Sᵥ)
    c) Weighted child entropy = Σ (|Sᵥ|/|S|) · H(Sᵥ)
    d) IG(S, A) = H(S) − weighted child entropy

Step 3: Pick attribute with MAX IG → root node.

Step 4 (if asked): recurse on each branch / continue building tree
       OR: compute Gain Ratio = IG / SplitInfo to break ties.
```

### 8.4 Logistic-regression GD numerical template

```
Step 1: For each example i, compute zᵢ = θᵀxᵢ (don't forget bias!)
Step 2: hᵢ = σ(zᵢ) = 1 / (1 + e⁻ᶻⁱ)
Step 3: Average gradient per weight:
        ∂J/∂θⱼ = (1/m) · Σᵢ (hᵢ − yᵢ) · xᵢⱼ
Step 4: Update each θⱼ ← θⱼ − α · (∂J/∂θⱼ)
Step 5: (If asked) BCE log-loss = −(1/m) Σᵢ [yᵢ log hᵢ + (1−yᵢ) log(1−hᵢ)]
Step 6: (If asked) Predict for new x → z = θᵀx → h = σ(z) → predict 1 if h > 0.5
```

### 8.5 ⭐ "Kill list" of must-know facts (skim 5 min before exam)

- ML = learn from E, with task T, scored by P (Tom Mitchell).
- Supervised → numeric = regression, categorical = classification. Unsupervised = no labels. Reinforcement = trial+reward.
- Discriminative = boundary only (Logistic, SVM, Tree). Generative = full P(X,Y) (Naive Bayes, GMM).
- Z-score: v' = (v − μ)/σ. Min-max: v' = (v − min)/(max − min). Fit on training data only.
- Linear regression closed-form: θ = (XᵀX)⁻¹ Xᵀy. Slow when n is huge → use GD.
- GD update: θⱼ ← θⱼ − α·(1/m) Σ(hᵢ − yᵢ) xᵢⱼ. Same form for linear AND logistic regression.
- α too big → J oscillates / diverges → decrease α. α too small → very slow.
- Ridge (L2): shrinks weights; Lasso (L1): can zero out features.
- High bias = underfit (both errors high & similar). High variance = overfit (train low, test high).
- Sigmoid: σ(z) = 1/(1+e⁻ᶻ). σ(0) = 0.5. Big positive z → 1; big negative z → 0.
- Logistic regression's boundary is **straight** (where wᵀx = 0). Sigmoid only handles confidence.
- Log-odds: log(p/(1−p)) = θᵀx. Each weight gives the multiplicative effect on odds = exp(θⱼ).
- Log-loss is convex; MSE+sigmoid is not convex → don't use MSE for classification.
- Confusion matrix: TP / FN / FP / TN. Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = harmonic mean.
- Accuracy paradox: 99% accuracy can mean **zero** recall on imbalanced data.
- Entropy: H = − Σ p log₂(p). 50-50 → 1 bit. All-same → 0.
- Information Gain = parent entropy − weighted average of children entropies. Pick max IG.
- IG biased to many-valued attributes → use Gain Ratio.
- Continuous features: try midpoints where class label changes; pick the threshold with max IG.
- Pre-prune (early stop) vs post-prune (grow then chop). MDL = shortest tree + errors.
- DT: "white-box", no scaling, but unstable; weakest against noisy features. Linear regression: weakest against outliers (MSE squares them).

---

## 9. Exam-day strategy (last 15 minutes)

### 9.1 Time budget (2 hours, 6 questions × ~5 marks each)

- ~15 min per question on average. **Reserve last 10 min for review.**
- Skim all 6 first; tackle the easiest two first to lock confidence.
- For each numerical: **show every intermediate step** (formulas + substitutions + final value). Most rubrics give partial marks step-wise.

### 9.2 Numerical question template (works for ANY numerical)

```
1. State the formula.
2. Identify variables from the question.
3. Substitute values.
4. Compute step-by-step (don't skip lines).
5. State the final answer with units / interpretation.
```

### 9.3 Theory question template (4-6 mark)

```
1. Define the term in one sentence.
2. Why / when used (give one concrete reason).
3. Mention any formula or mechanism (1 line).
4. Trade-off / limitation (1 line).
5. (If asked) tie back to the scenario in the question.
```

### 9.4 If you get stuck — partial-credit hooks

- **Identification questions** (E/T/P, supervised/unsupervised, discriminative/generative): even if unsure, write the definition first → guaranteed 0.5–1 mark.
- **Numericals**: write the formula. Even if numbers are wrong, formula + structure = ≥30% marks.
- **Decision tree IG**: even partial calculation of parent entropy is 1 mark; pick whichever attribute makes one child obviously pure for partial credit.
- **Confusion matrix calc**: correctly placing TP/FN/FP/TN is 1 mark even if your divisions are off.

### 9.5 Common traps to avoid

- Forget the **|Sᵥ|/|S|** weight in IG (#1 mistake!).
- Mix up Precision and Recall denominators: P = TP+**FP**, R = TP+**FN**.
- Use MSE with logistic regression (it's NOT convex with sigmoid → use log-loss).
- Forget the **bias term** column of 1's when stating dimensions of X (it becomes m × (n+1), not m × n).
- Use 3σ outlier rule on skewed data (use IQR instead).
- Conclude a model is good purely from accuracy on imbalanced data.
- Encode unordered categories as 0,1,2 (use one-hot instead).

---

## 10. Companion: Question Bank

For exhaustive variants of each question seen across all past papers, see **`ML_MIDSEM_QUESTION_BANK.md`** (organised by topic). This guide explains the **why**; the question bank drills the **how**.

**Good luck — you've got this.**