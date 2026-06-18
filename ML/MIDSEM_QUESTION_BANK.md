# ML MidSem — Master Question Bank (all variants)

> Compiled from official midsem papers: 2023 Reg, 2023 Reg-2, 2024 Reg, 2024 Mkp, Dec 2025 Reg, Jan 2026 Mkp, Sample QP, plus practice problems (MFML SEC9). Organised by topic so you can drill weak spots.
> **Scope**: Modules 1–5 only (Intro, Workflow, Linear Regression, Linear Classification, Decision Tree). Out-of-scope topics (kNN, SVM, Naive Bayes, Ensembles) are excluded.

## How to use this bank

1. Pick a topic where you feel weakest.
2. Try the question without looking at the solution.
3. Compare with the worked solution.
4. Note the marking-rubric clues so your exam answer maps to the marking key.

---

## TABLE OF CONTENTS

1. **Module 1 — Introduction**
   - 1.1 Tom Mitchell's E, T, P framework
   - 1.2 Identify the learning type (supervised/unsup/RL)
   - 1.3 Discriminative vs Generative
2. **Module 2 — ML Workflow & Data**
   - 2.1 Spot data quality issues
   - 2.2 Z-score normalization (numerical)
   - 2.3 Outlier detection (IQR / 3σ)
   - 2.4 Missing value imputation
   - 2.5 Feature scaling — True/False
   - 2.6 Train/Val/Test split & class imbalance
3. **Module 3 — Linear Regression**
   - 3.1 Closed-form & matrix dimensions
   - 3.2 Gradient descent — one iteration
   - 3.3 Learning rate diagnosis
   - 3.4 Bias-variance diagnosis
   - 3.5 Ridge vs Lasso — choose & explain
   - 3.6 Coefficient interpretation
4. **Module 4 — Classification & Logistic Regression**
   - 4.1 Discriminant function (sign rule)
   - 4.2 Sigmoid plotting & overfitting
   - 4.3 Log-odds interpretation
   - 4.4 One-iteration logistic GD
   - 4.5 Effect of weights / θ values
   - 4.6 Cost J(θ) = 0 corollaries
5. **Module 4b — Evaluation Metrics**
   - 5.1 Compute P, R, F1, Specificity from confusion matrix
   - 5.2 Class-imbalance & accuracy paradox
   - 5.3 Cost-aware threshold tuning
   - 5.4 ROC / TPR / FPR computation
   - 5.5 Per-class metrics (negative class)
6. **Module 5 — Decision Tree**
   - 6.1 Compute parent entropy + IG, pick root
   - 6.2 Continuous attribute thresholding (C4.5)
   - 6.3 Pre vs Post pruning
   - 6.4 Limitations of IG / Gain Ratio
   - 6.5 Robustness to noise / outliers comparison
7. **Cross-cutting** — model comparisons (linear vs logistic vs tree)

---

## TOPIC 1 — Module 1: Introduction

### 1.1 Tom Mitchell's E, T, P framework

**Pattern**: given a problem, identify Experience (E), Task (T), Performance measure (P), and name 2 P metrics.

#### Variant 1.1.1 — Weather classification [2024-Mkp Q1, 2023-Reg Q1a; 2 marks]
**Q**: Historical weather data → predict Sunny / Windy / Rainy. Identify E, T, P. Name 2 P measures.

**Answer**:
- **T** = predicting weather category for given input parameters (a **classification** task).
- **P** = the probability of correctly predicting future weather / number of new instances correctly classified.
- **E** = the historical weather data the algorithm examines.
- **2 P measures**: Accuracy, Precision, Recall, F1-score, AUC-ROC, Misclassification rate (any 2).

**Marking**: 1.5 marks for E/T/P (0.5 each); 0.25 marks each for two valid metrics.

#### Variant 1.1.2 — Rainfall regression [2024-Mkp Q1.2, 2023-Reg Q1a.2; 1 mark]
**Q**: Predict tomorrow's rainfall in mm. Which learning technique? Justify. Name 2 P measures.

**Answer**:
- Technique: **Regression** (target variable rainfall is continuous).
- **2 P measures for regression**: MSE, RMSE, MAE, R² (any 2).

#### Variant 1.1.3 — Spam classifier [Standard textbook]
**Q**: Build a system to label e-mail spam vs not spam. Identify E, T, P.

**Answer**:
- **T** = classify each e-mail as spam / not spam.
- **P** = % of correctly classified e-mails (or precision-on-spam, depending on cost).
- **E** = past e-mails already labelled spam / not spam.

#### Variant 1.1.4 — Game-playing transition [2023-Reg Q1b; 2 marks]
**Q**: First version of Go-playing AI watched humans play; second version learnt by self-play with rewards. Describe the transition.

**Answer**:
- **First version → Supervised Learning** (1 mark) — observed and imitated labelled human moves.
- **Second version → Reinforcement Learning** (1 mark) — learnt from reward signal during self-play.

### 1.2 Identify the type of learning (supervised/unsupervised/RL)

**Pattern**: Given several scenarios, classify each. Asked in EVERY paper (2024 Reg Q1a, others).

#### Variant 1.2.1 — Mixed scenarios [2024-Reg Q1a; 2 marks]

| Scenario | Answer |
|---|---|
| Self-driving car learns to navigate, avoid obstacles, minimize fuel through environmental feedback | Reinforcement Learning |
| Online retailer predicts review sentiment (positive/negative/neutral) using labelled customer reviews | Supervised — Classification |
| Geneticist clusters DNA sequences without labels to find disease links | Unsupervised — Clustering |
| Insurance company predicts claim amount based on age, driving history, vehicle type | Supervised — Regression |

**Marking**: 0.25 mark for correct family + 0.25 mark for justification (each scenario).

#### Variant 1.2.2 — More scenarios

| Scenario | Answer |
|---|---|
| Group customers into "frequent / occasional / inactive" without prior labels | Unsupervised — Clustering |
| Predict house price from area, age, bedrooms | Supervised — Regression |
| Detect anomalous network traffic (no labels available) | Unsupervised — Anomaly detection |
| Robot arm learns by trial and error to grasp objects | Reinforcement Learning |
| Classify handwritten digits from labelled pixel data | Supervised — Classification |
| Recommend movies based on click history (ratings as feedback over time) | RL or Supervised (depends on framing) |

**Trigger words to lock in**:
- "labelled" / "past data with answers" → Supervised
- "predict number / amount" → Regression
- "predict class / category / label" → Classification
- "no labels / cluster / find groups / hidden patterns" → Unsupervised
- "feedback / rewards / interactions / trial and error / agent" → Reinforcement

### 1.3 Discriminative vs Generative classifier

#### Variant 1.3.1 — Outlier detection scenario [2023-Reg-2 Q1b, Sample Q1b; 2 marks]
**Q**: Friend wants to (a) classify job applications as good/bad, AND (b) detect liar applicants using density estimation. Recommend discriminative or generative? Why?

**Answer**: **Generative classifier** (1 mark). Reason: density estimation requires modelling P(X | y), which is what generative classifiers compute. Discriminative classifiers only model P(y | X) — the boundary — and cannot do density-based outlier detection (1 mark).

#### Variant 1.3.2 — Quick comparison
**Q**: Briefly state the difference between discriminative and generative classifiers and give one example of each.

**Answer**:
- **Discriminative**: models P(y | X) directly; learns the decision boundary. Examples: Logistic Regression, SVM, Decision Tree.
- **Generative**: models P(X, y) — the joint distribution. Uses Bayes rule: P(y | X) ∝ P(X | y)·P(y). Can sample from the model and detect outliers. Examples: Naive Bayes, GMM, LDA.

---

## TOPIC 2 — Module 2: ML Workflow & Data Preprocessing

### 2.1 Spot data quality issues (4–5 marks; APPEARS IN EVERY PAPER)

**Pattern**: Given a messy table, identify ≥5 issues and suggest fixes.

#### Variant 2.1.1 — Patient COVID dataset [2023-Reg-2 Q1a, Sample Q1a; 5 marks]

| TXN-ID | NAME | AGE | HEIGHT | WEIGHT | BLOOD | COVID-RESULT |
|---|---|---|---|---|---|---|
| T001 | RAMA | 45 | 145 | 62kg | O+ve | Positive |
| T003 | Akbar | 38 | 172 | 60kg | Iam+ve | Positive |
| T005 | THenali | 22 | 157 | 78kg | B-ve | 1 |
| T007 | Rajuu | 350 | 132 | 48kg | O+ve | Positive |
| T008 | HARI | 32 | 180 | 120lbs | AB-ve | Negative |
| T009 | Inba | 25 | (missing) | 85kg | O+ve | 0 |
| T010 | SysUsr789 | 20 | 165 | 68kg | O-ve | Negative |

**Issues + fixes**:
1. **Invalid name** — T010 "SysUsr789" looks like a system ID, not a name. **Fix**: replace with correct name or drop record.
2. **Outlier in Age** — T007 has Age = 350, impossible. **Fix**: replace with mean/median age, or remove.
3. **Missing value** — T009 has missing height. **Fix**: impute with mean/median (median is safer if outliers exist).
4. **Inconsistent unit** — T008 weight in lbs, others in kg. **Fix**: convert lbs → kg (×0.4536).
5. **Invalid blood group format** — T003 "Iam+ve" is malformed. **Fix**: replace with NULL and use mode imputation, or apply binning.
6. **Encoding mismatch** — T005 covid result "1" and T009 "0" instead of "Positive"/"Negative". **Fix**: data smoothing / re-encode to consistent labels.

**Marking**: 0.5 marks per issue identified + 0.5 marks per fix proposed (5 issues × 1 mark = 5 total).

#### Variant 2.1.2 — Student admissions dataset [2024-Reg Q2; 4 marks]

| Student | Age | Email | Specialization | Admission Date | CGPA |
|---|---|---|---|---|---|
| Rina | 22 | rina@example.com | AI | 15-Aug-2022 | 8.2 |
| Dev | 23 | dev[at]mail.com | Cybersecurity | 2022-08-15 | 7.8 |
| Aarav | 22 | aarav@example.com | Data Science | 15/08/2022 | 9.1 |
| Shweta | 22 | (missing) | Cybersecurity | 15-Aug-2022 | 6.4 |
| Rina | 22 | rina@example.com | Artificial Intelligence | 15-Aug-2022 | 8.2 |
| Mansi | 58 | mansi123@gmail.com | Data Science | 15-Aug-2022 | 7.0 |

**Issues + fixes** (find ≥6):
1. **Inconsistent date formats** ("15-Aug-2022", "2022-08-15", "15/08/2022"). **Fix**: standardize to ISO YYYY-MM-DD.
2. **Duplicate records** (Rina appears twice with identical data). **Fix**: deduplicate on (name, email, specialization).
3. **Age outlier** — Mansi at 58 is unlikely for a regular M.Tech student. **Fix**: investigate, cap or remove.
4. **Missing email** — Shweta has no email. **Fix**: collect or impute "unknown".
5. **Invalid email format** — Dev's "dev[at]mail.com". **Fix**: regex validation.
6. **Inconsistent category names** — "AI" vs "Artificial Intelligence". **Fix**: mapping table to unify.

#### Variant 2.1.3 — Student CGPA dataset [2023-Reg Q2; 3 marks]

| Name | Age | DOB | Course | CGPA |
|---|---|---|---|---|
| Aishwarya | 24 | 01-Jan-1995 | CS104 | 7.4 |
| Bhargav | 23 | Dec-01-1996 | CS102 | 7.5 |
| Chandra | 25 | 01-Nov-1994 | (missing) | 6.7 |
| Bhargav | 23 | Dec-01-1996 | CS102 | 8.1 |
| Eshan | 24 | 01-Jul-1995 | CS103 | 87.5 |
| Francis | 54 | 01-01-1959 | CS105 | 7.0 |

**Issues + fixes**:
1. **Missing Course ID** for Chandra. **Fix**: drop or impute mode.
2. **Redundant columns** — Age can be derived from DOB. **Fix**: drop one (keep DOB; compute Age dynamically).
3. **Inconsistent DOB formats**. **Fix**: parse to ISO date.
4. **Duplicate-key conflict** — Bhargav appears twice with different CGPA. **Fix**: investigate; pick latest or average; deduplicate.
5. **CGPA outlier** — Eshan = 87.5 (likely typo, should be 8.75). **Fix**: cap to scale (0–10) or correct.
6. **Age outlier** — Francis = 54 in B.Tech program. **Fix**: investigate; possibly remove.

### 2.2 Z-score normalization (numerical)

#### Variant 2.2.1 — IPL teams dataset [Dec 2025 Q1b; 3 marks]
**Q**: Normalize Total Runs and Total Wickets using z-score.

| Team | Total Runs | Total Wickets |
|---|---|---|
| A | 820 | 40 |
| B | 840 | 44 |
| C | 860 | 48 |
| D | 880 | 52 |
| E | 900 | 56 |

**Solution — Total Runs**:
- μ = (820 + 840 + 860 + 880 + 900) / 5 = 4300 / 5 = **860**
- Variance = (40² + 20² + 0² + 20² + 40²) / 5 = (1600 + 400 + 0 + 400 + 1600) / 5 = 4000/5 = **800**
- σ = √800 ≈ **28.28**
- Z-scores: 820 → −1.41, 840 → −0.71, 860 → 0, 880 → +0.71, 900 → +1.41 ✓

**Solution — Total Wickets**:
- μ = (40 + 44 + 48 + 52 + 56) / 5 = 240 / 5 = **48**
- Variance = (8² + 4² + 0² + 4² + 8²) / 5 = (64 + 16 + 0 + 16 + 64) / 5 = 160/5 = **32**
- σ = √32 ≈ **5.66**
- Z-scores: 40 → −1.41, 44 → −0.71, 48 → 0, 52 → +0.71, 56 → +1.41 ✓

**Normalized dataset**:

| Team | Mean(Batsmen) | Runs(z) | Wickets(z) |
|---|---|---|---|
| A | 7 | −1.41 | −1.41 |
| B | 5 | −0.71 | −0.71 |
| C | 6 | 0.00 | 0.00 |
| D | 8 | +0.71 | +0.71 |
| E | 4 | +1.41 | +1.41 |

#### Variant 2.2.2 — Min-Max normalization
**Q**: Normalize income from {12000, 50000, 73000, 86000, 98000} to [0, 1].

**Solution**: Use v' = (v − min) / (max − min). Min = 12000, max = 98000, range = 86000.
- 12000 → 0
- 50000 → 38000/86000 ≈ 0.442
- 73000 → 61000/86000 ≈ 0.709
- 86000 → 74000/86000 ≈ 0.860
- 98000 → 1

### 2.3 Outlier detection

#### Variant 2.3.1 — IQR method [from CS2 companion]
**Q**: Find outliers in {2, 10, 11, 11, 11, 12, 13, 14, 14, 15, 17, 22}.

**Solution**:
- Sorted: 2, 10, 11, 11, 11, 12, 13, 14, 14, 15, 17, 22.
- Q1 (25th percentile) = 11; Q3 (75th percentile) = 14.5.
- IQR = 14.5 − 11 = 3.5.
- Lower fence = 11 − 1.5(3.5) = **5.75**.
- Upper fence = 14.5 + 1.5(3.5) = **19.75**.
- Outliers: **2** (below 5.75), **22** (above 19.75).

#### Variant 2.3.2 — 3-sigma method
**Q**: Mean = 50, std dev = 5. Are values 70 and 48 outliers under the 3σ rule?

**Solution**:
- Lower bound = μ − 3σ = 50 − 15 = 35.
- Upper bound = μ + 3σ = 50 + 15 = 65.
- 70 > 65 → **outlier**.
- 48 ∈ [35, 65] → **not an outlier**.

### 2.4 Missing value imputation

#### Variant 2.4.1 — Numeric column with all-unique values [Dec 2025 Q1a; 1 mark]
**Q**: Mean Number of Batsmen Played has values 7, 5, 8, 4 known + 1 missing. Suggest method.

**Answer**: **Mean or median imputation** is appropriate (continuous numeric, symmetric).
- Mean = (7+5+8+4)/4 = **6**
- Median (sorted: 4, 5, 7, 8) = (5+7)/2 = **6**

Both give 6. **Mode imputation is inappropriate** since values are all unique (mode is undefined / arbitrary).

#### Variant 2.4.2 — Categorical column
**Q**: Customer occupation column has 5% missing. Suggest method.

**Answer**: Use **mode** (most frequent occupation) — categorical features have no meaningful mean. Alternative: treat "missing" as its own category if missingness is informative.

### 2.5 Feature scaling — True/False statements

#### Variant 2.5.1 — All four standard claims [2024-Mkp Q2; 6 marks]

| Statement | Answer | Reasoning |
|---|---|---|
| Scaling guarantees gradient descent reaches global minimum. | **False** | Convergence to global minimum is determined by the **loss surface**, not the input scale. Non-convex losses (e.g., NN, log-reg with non-convex penalty) can have local minima irrespective of scaling. |
| Scaling prevents X^T X from being singular. | **False** | Singularity is caused by **linear dependence among columns**, not by scale. Scaling rescales columns but doesn't change rank. |
| Scaling is necessary to prevent gradient descent from local optima. | **False** | Linear regression's MSE is **convex** — it has a single global minimum. Scaling doesn't affect the existence of optima, only the speed of convergence. |
| Scaling speeds up gradient descent (fewer iterations). | **True** | When features have very different scales, contour lines of the loss become long ellipses → GD zig-zags. Scaling makes contours roughly circular → straight downhill path → faster convergence. |

**Marking**: 0.5 mark for True/False + 1 mark for explanation each.

### 2.6 Train/Validation/Test split & class imbalance

#### Variant 2.6.1 — Imbalanced loan dataset [2024-Reg Q3; 5 marks]
**Q**: Dataset of 10,000 loan applications, only 800 approved. (a) How to split? Justify. (b) Two models: A=95% acc, B=84% acc. Which is reliable? Other metrics? (c) Why keep test set untouched?

**Solutions**:

(a) **Stratified split**: preserve the ~8% approved / 92% rejected ratio across train, validation, test sets. Without stratification, a small validation set might end up with ~0% approved → useless evaluation. Common ratios: 70/15/15 or 80/10/10. Justify using the rare-positive imbalance. (1.5 marks)

(b) Accuracy alone is misleading. Trivial "predict all rejected" achieves 92% accuracy. Without additional metrics:
- **Model B may actually be better** if it identifies more approved loans correctly.
- Need **Precision, Recall, F1, AUC-ROC, confusion matrix** — especially for the minority (approved) class. (2 marks)

(c) Test set must be locked away to give an honest estimate of generalisation:
- Using test data during training/tuning **leaks the answer key** → over-optimistic test scores.
- Reserve it for the **final** evaluation only. (1.5 marks)

---

## TOPIC 3 — Module 3: Linear Regression

### 3.1 Closed-form solution & matrix dimensions

#### Variant 3.1.1 — Dimensions of θ, X, y [2023-Reg Q6a; 2 marks]
**Q**: m = 28 training examples, n = 4 features (excluding the bias / all-ones feature). For closed form θ = (XᵀX)⁻¹ Xᵀy, what are the dimensions of θ, X, y?

**Solution**:
- After adding the bias column, X has n+1 = **5 columns** → X is **28 × 5**.
- y is the target vector → **28 × 1**.
- θ has one weight per feature (incl. bias) → **5 × 1**.

**Marking**: X = 1 mark, y = 0.5 marks, θ = 0.5 marks. Wrong dimensions = 0 for that parameter.

#### Variant 3.1.2 — When to prefer GD over closed form [2023-Reg-2 Q2b, Sample Q2b; 3 marks]
**Q**: n = 2,000,000 instances, m (features) = 300,000. Use gradient descent or normal equations? Why?

**Answer**: **Gradient Descent** (1 mark).
- Normal equations require computing (XᵀX)⁻¹, which is **O(d³)** where d is the number of features.
- With d = 300,000, d³ ≈ 2.7 × 10¹⁶ operations — completely infeasible (slow & memory-heavy).
- GD scales linearly per iteration in d, so it remains tractable. (2 marks for explanation)

#### Variant 3.1.3 — Edge cases when closed form fails
**Q**: When does the closed-form θ = (XᵀX)⁻¹ Xᵀy fail?

**Answer**:
1. **Computational**: very large n (matrix inverse is O(n³), memory issues).
2. **Singular X^T X**: when columns of X are linearly dependent (multicollinearity), or when m < n+1 (more features than examples). Inverse doesn't exist.
3. **Numerical instability**: near-singular X^T X gives unreliable inverse. Use SVD-based pseudo-inverse, or switch to GD with regularization (Ridge).

### 3.2 Gradient descent — one iteration (numerical)

#### Variant 3.2.1 — Heart disease risk [CS3 companion]
**Q**: Predict heart-disease risk = w₀ + w₁·BMI + w₂·Pressure. Initial w = (5, −0.03, −0.03), α = 0.02, m = 3. Patients:

| Patient | BMI | Pressure | y (real risk) |
|---|---|---|---|
| 1 | 35 | 80 | 1.81 |
| 2 | 25 | 80 | 1.22 |
| 3 | 30 | 100 | 1.71 |

**Solution**:
Step 1 — Predictions and gaps (gap = ŷ − y):
- ŷ₁ = 5 − 0.03(35) − 0.03(80) = 5 − 1.05 − 2.40 = 1.55, gap = 1.55 − 1.81 = **−0.26**
- ŷ₂ = 5 − 0.03(25) − 0.03(80) = 1.85, gap = 1.85 − 1.22 = **+0.63**
- ŷ₃ = 5 − 0.03(30) − 0.03(100) = 1.10, gap = 1.10 − 1.71 = **−0.61**

Step 2 — Sums of (gap × feature):
- S₀ = −0.26 + 0.63 − 0.61 = **−0.24**
- S₁ = (−0.26)(35) + (0.63)(25) + (−0.61)(30) = −9.10 + 15.75 − 18.30 = **−11.65**
- S₂ = (−0.26)(80) + (0.63)(80) + (−0.61)(100) = −20.80 + 50.40 − 61.00 = **−31.40**

Step 3 — Updates (nudge = (α/m)·S = 0.006667·S):
- w₀ = 5 − 0.006667(−0.24) = **5.0016**
- w₁ = −0.03 − 0.006667(−11.65) = −0.03 + 0.0777 = **0.0477**
- w₂ = −0.03 − 0.006667(−31.40) = −0.03 + 0.2093 = **0.1793**

**New line**: ŷ = 5.0016 + 0.0477·BMI + 0.1793·Pressure.

#### Variant 3.2.2 — Tabulating one mini-batch step
**Q**: Show one batch GD update for ŷ = θ₀ + θ₁·x with initial θ = (0, 0), α = 0.1, m = 4 examples (1, 2), (2, 3), (3, 5), (4, 4).

**Solution**:
- ŷ₁ = 0, gap = 0 − 2 = −2.
- ŷ₂ = 0, gap = 0 − 3 = −3.
- ŷ₃ = 0, gap = 0 − 5 = −5.
- ŷ₄ = 0, gap = 0 − 4 = −4.
- S₀ = −14. S₁ = (−2)(1) + (−3)(2) + (−5)(3) + (−4)(4) = −2 − 6 − 15 − 16 = −39.
- θ₀ ← 0 − (0.1/4)(−14) = +0.35. θ₁ ← 0 − (0.1/4)(−39) = +0.975.
- New: ŷ = 0.35 + 0.975·x.

### 3.3 Learning rate diagnosis (must-know)

#### Variant 3.3.1 — J(θ) increasing [2023-Reg Q6b; 2 marks]
**Q**: Run GD for 15 iterations with α = 0.3, MSE cost. J(θ) **increases** over time. What conclusion? Should α be increased or decreased?

**Answer**: **Learning rate is too high** → algorithm is overshooting the minimum and diverging.
- **Decrease α** (try α = 0.1 or 0.01) and re-run. (0.5 mark for direction; 1.5 marks for justification)
- Sketch the bouncing-out-of-bowl picture if asked.

#### Variant 3.3.2 — J flat / barely changing
**Q**: J(θ) decreases but very slowly over 1000 iterations. Diagnosis?

**Answer**: α is **too small** → tiny steps, slow convergence. **Increase α** (try 0.01 → 0.1).

### 3.4 Bias-Variance diagnosis & overfitting

#### Variant 3.4.1 — Diagnosing from RMSE table [2024-Mkp Q3 part A; 2 marks]
**Q**: 4 models tested on real-estate prediction.

| Model | Train RMSE | Test RMSE | R² (test) | Features |
|---|---|---|---|---|
| A: Linear, 3 features | $45,000 | $47,000 | 0.66 | 3 |
| B: Poly-8, 15 features (120 expanded) | $12,000 | $68,000 | 0.42 | 120 |
| C: Poly-8 + Ridge λ=100 | $28,000 | $36,500 | 0.76 | 120 |
| D: Poly-8 + Lasso λ=50 | $31,000 | $34,800 | 0.79 | 23 |

Identify bias/variance and underfitting/overfitting.

**Solution**:
- **Model A**: Train RMSE ($45k) ≈ Test RMSE ($47k), both high → **High bias, low variance → UNDERFITTING**. (Small gap, both high.)
- **Model B**: Train RMSE ($12k) << Test RMSE ($68k), gap = $56k → **Low bias, high variance → SEVERE OVERFITTING**. (Big gap.)
- **Model C**: Moderate train ($28k), better test ($36.5k), R² = 0.76. Better balance.
- **Model D**: Best test RMSE ($34.8k) and R² = 0.79. **Best generalising model**.

#### Variant 3.4.2 — Ridge with high train + high val error [2023-Reg-2 Q2a, Sample Q2a; 3 marks]
**Q**: Ridge regression: training error and validation error are almost equal and fairly high. Bias or variance? λ direction?

**Answer**: 
- **High bias / underfitting** (1 mark) — both errors high & similar → model is too constrained.
- **Decrease λ** (1 mark) — large λ is over-penalising the weights, forcing the model to be too simple.
- **Justification** (1 mark): higher λ → higher bias, lower variance. Lower λ → opposite. Use cross-validation to find the sweet spot.

#### Variant 3.4.3 — Doubling dataset, val error stays high [Dec 2025 Q2; 2 marks]
**Q**: Student says: "Doubled dataset size, validation RMSE still high while training RMSE keeps dropping. So increasing data doesn't help." Critique.

**Answer**:
- Model has **low bias, high variance** (overfitting).
- More data should reduce variance, but the gap persists, suggesting the model is **too complex** OR the data distribution shifted, OR noise increased.
- **Corrective actions**: reduce polynomial order, apply regularization, normalize inputs, recheck data quality.

#### Variant 3.4.4 — Logistic regression overfitting symptoms [2024-Mkp Q5; 4 marks]
**Q**: Symptoms: train acc 0.94, val acc 0.70; train log-loss 0.15, val 0.55; β₁=85, β₂=−72; small data perturbations cause large coefficient swings.

**Answer**:
(a) **Overfitting** (high variance) — large gap in metrics, huge weights, instability under perturbation.

(b) Why arises: log-loss optimisation pushes weights to **large magnitudes** when data is (nearly) linearly separable, high-dimensional, or noisy. Without regularization, the model becomes too flexible and fits noise.

(c) Mitigations:
- **L1/L2 regularization** — adds penalty term to log-loss, shrinking weights.
- **Early stopping** — halt training when validation log-loss starts rising.
- **More data** / **fewer features**.

### 3.5 Ridge vs Lasso — choose & explain

#### Variant 3.5.1 — High-dim noisy features [2024-Reg Q3; 2+2 marks]
**Q**: Hundreds of predictors, many redundant/irrelevant. Ridge or Lasso?

**Answer**: **Lasso (L1)** because:
- Lasso adds Σ|θⱼ| penalty → drives many coefficients to **exactly zero** → built-in feature selection.
- Reduces complexity, improves interpretability.
- Ridge keeps all features (only shrinks); doesn't help when many features are useless.

**Mathematical expressions**:
- Ridge: J + λ Σ θⱼ²
- Lasso: J + λ Σ |θⱼ|

#### Variant 3.5.2 — Penalize but keep all coefficients [2023-Reg Q3a; 1 mark]
**Q**: Want to penalize large coefficients but NOT set them to 0. Which technique?

**Answer**: **Ridge regularization (L2)** — shrinks all weights smoothly toward zero without zeroing any out.

#### Variant 3.5.3 — Why number of features differs across regularizers [2024-Mkp Q3 part E]
**Q**: Model C (Ridge) used 120 features; Model D (Lasso) used 23 features. Justify.

**Answer**:
- **Ridge** retains all coefficients but reduces their magnitudes → all 120 features still contribute.
- **Lasso** zeros out non-informative coefficients → only 23 useful features remain; 97 set to exactly 0.

#### Variant 3.5.4 — Reducing bias of regularized model [2024-Mkp Q3 part D]
**Q**: Model C (Ridge) shows moderate bias. How to reduce bias further?

**Answer**: **Decrease λ** — lower regularization allows weights to grow → model fits training data better → lower bias (at cost of higher variance).

### 3.6 Coefficient interpretation

#### Variant 3.6.1 — Predict house price [Dec 2025 Q5A; 2 marks]
**Q**: Model: price = 50 + 200·area + 5·age. Interpret coefficients. What if area is in ft² instead of m²?

**Answer**:
(a) Coefficient interpretation:
- **200**: each additional m² of area adds 200 to the price, holding age constant.
- **5**: each additional year of age adds 5 to the price (or subtracts, depending on sign).
- **50** (intercept): predicted price when area = 0 and age = 0 (baseline; usually not physically meaningful).

(b) **Unit change**: changing area from m² to ft² **rescales the coefficient** but not the model behaviour. Since 1 m² ≈ 10.76 ft², the new θ_area ≈ 200 / 10.76 ≈ 18.6 per ft². The relationship stays identical.

#### Variant 3.6.2 — Feature influence comparison [Dec 2025 Q5C; 1 mark]
**Q**: Which feature has stronger influence — area (θ=200) or age (θ=5)?

**Answer**: **Area** has a much stronger influence because its coefficient is far larger (40× the age coefficient). Each unit increase in area moves the prediction far more than a unit increase in age, **assuming features are on similar scales** (or have been standardized).

#### Variant 3.6.3 — Gradient comparison [Dec 2025 Q5B; 1 mark]
**Q**: ∂J/∂θ₁ is large while ∂J/∂θ₂ ≈ 0. Inference?

**Answer**: θ₁ is **far from optimal** (large gradient indicates much room for improvement). θ₂ is **already near-optimal** (gradient zero means no improvement possible by moving θ₂) OR θ₂ corresponds to an irrelevant feature that doesn't affect the loss.

### 3.7 J(θ) = 0 corollaries

#### Variant 3.7.1 — Logistic regression cost = 0 [2023-Reg-2 Q3, Sample Q3; 6 marks]
**Q**: Found θ₀, θ₁ such that J(θ₀, θ₁) = 0. For each statement, T/F + justification:

(a) **Model will work perfectly on unseen data.**
**False** — J = 0 on training set means **overfitting**. No guarantee of generalisation; the model may have memorised noise.

(b) **If J = 0, then h(x⁽ⁱ⁾) = y⁽ⁱ⁾ for every training example.**
**True** — log-loss is 0 only when each training prediction matches the true label perfectly (well, with σ approaching 1 for y=1 and 0 for y=0).

(c) **For J = 0, θ₀ and θ₁ must be 0.**
**False** — θ₀ = θ₁ = 0 gives σ(0) = 0.5 for every input, which has non-zero log-loss in general. Many specific (θ₀, θ₁) values can drive J to 0.

(d) **J(θ₀, θ₁) cannot be 0.**
**False** — J = 0 is achievable when training data is **linearly separable** and weights are pushed to ±∞ (or large enough that σ ≈ 0 or 1 perfectly).

**Marking**: 0.5 marks for T/F + 1 mark for justification (each part).

---

## TOPIC 4 — Module 4: Classification & Logistic Regression

### 4.1 Discriminant function (sign rule)

#### Variant 4.1.1 — Score sign decision [from CS4 companion]
**Q**: Linear classifier: score = 0.4 + 0.3·CGPA − 0.45·IQ. Classify candidate (CGPA = 5, IQ = 6).

**Solution**:
- score = 0.4 + 0.3(5) − 0.45(6) = 0.4 + 1.5 − 2.7 = **−0.8**
- score < 0 → "no job" side.

#### Variant 4.1.2 — Boundary equation
**Q**: For the same classifier, write the equation of the decision boundary in (CGPA, IQ) space.

**Answer**: 0.4 + 0.3·CGPA − 0.45·IQ = 0 → 0.45·IQ = 0.4 + 0.3·CGPA → IQ = (0.4 + 0.3·CGPA) / 0.45.

### 4.2 Sigmoid plotting & overfitting

#### Variant 4.2.1 — Sigmoid for various weights [2024-Reg Q4; 4 marks]
**Q**: Qualitatively sketch σ(wx) = 1 / (1 + e⁻ʷˣ) for w ∈ {1, 10, 100}. Explain how high weights cause overfitting.

**Answer**:

Sketch description (2 marks):
- **w = 1**: smooth, gradual S, slow transition from 0 to 1 around x = 0.
- **w = 10**: noticeably steeper S, sharp transition near x = 0.
- **w = 100**: nearly a step function — very abrupt jump from 0 to 1 at x = 0.

How high weights → overfitting (2 marks):
- Steeper sigmoid → small input changes cause large probability swings.
- Model becomes **overconfident** (predictions cluster near 0 or 1 even on borderline data).
- Loses smooth probabilistic interpretation — behaves like a hard threshold.
- Fits training noise → poor generalisation.

This motivates **regularization** to keep weights small.

### 4.3 Log-odds interpretation

#### Variant 4.3.1 — Coefficients explain log-odds [2023-Reg Q3c; 3 marks]
**Q**: Explain how logistic regression coefficients relate to log-odds. Mathematical justification.

**Answer**:
- Logistic regression models p = σ(θᵀx) = 1 / (1 + e⁻θᵀˣ).
- Take logit: log(p / (1 − p)) = **θᵀx** = θ₀ + θ₁x₁ + θ₂x₂ + …
- So each coefficient θⱼ represents the **change in log-odds** of the outcome per unit increase in xⱼ.
- Equivalently, multiplying the **odds** by exp(θⱼ).

**Mathematical derivation**:
```
p = 1 / (1 + e⁻ᶻ)   where z = θᵀx
1 − p = e⁻ᶻ / (1 + e⁻ᶻ)
p / (1 − p) = 1 / e⁻ᶻ = eᶻ
log(p/(1−p)) = z = θᵀx ✓
```

#### Variant 4.3.2 — Large positive θ does NOT guarantee selection [Jan 2026 Q5; 2 marks]
**Q**: A student claims a large positive θ₂ for CGPA "guarantees" high-CGPA candidates always get selected. Correct?

**Answer**: **No, this is incorrect** (0.5 marks).
- Logistic regression models **log-odds**, not direct decisions.
- A large positive θ_CGPA increases log-odds **per unit of CGPA**, but the final probability h(x) depends on:
  - θ₀ (bias term).
  - Other weights θ₁, θ₃, … and their feature values.
  - Actual ranges of features (e.g., a low IQ value with a strong negative weight could outweigh CGPA).
- Selection happens when σ(θᵀx) > 0.5, i.e., when the **combined weighted features** push θᵀx > 0 — not based on one feature alone. (1.5 marks)

#### Variant 4.3.3 — Computing odds change
**Q**: If w_CGPA = 0.3, what does a unit increase in CGPA do to the odds?

**Answer**: Multiplies odds by **e^0.3 ≈ 1.35** → a **35% increase** in odds, holding other features fixed.

### 4.4 One-iteration logistic GD (numerical — high-yield)

#### Variant 4.4.1 — IPL playoff prediction [Dec 2025 Q1c; 4 marks]
**Q**: 4 weights θ = [0.1, 0.1, 0.1, 0.2] (bias + 3 features), α = 0.1, m = 5 teams. Z-score normalized features (from §2.2.1):

| Team | x₁ (Batsmen) | x₂ (Runs-z) | x₃ (Wickets-z) | y (qualified) |
|---|---|---|---|---|
| A | 7 | −1.41 | −1.41 | 0 |
| B | 5 | −0.71 | −0.71 | 0 |
| C | 6 | 0 | 0 | 1 |
| D | 8 | +0.71 | +0.71 | 1 |
| E | 4 | +1.41 | +1.41 | 1 |

**Solution outline**:
- For each team, compute z = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃, then h = σ(z).
- Compute average gradient per weight: ∂J/∂θⱼ = (1/5) Σᵢ (hᵢ − yᵢ) xᵢⱼ.
- Update each θⱼ ← θⱼ − α · (∂J/∂θⱼ).

Following the official answer key, the gradients evaluate to:
- ∂J/∂θ₀ = 0.065
- ∂J/∂θ₁ = −0.1946
- ∂J/∂θ₂ = 0.197
- ∂J/∂θ₃ = 0.197

Updates:
- θ₀ = 0.1 − 0.1(0.065) = **0.0935**
- θ₁ = 0.1 − 0.1(−0.1946) = **0.1195**
- θ₂ = 0.1 − 0.1(0.197) = **0.0803**
- θ₃ = 0.2 − 0.1(0.197) = **0.1803**

#### Variant 4.4.2 — Predict Team D after iteration [Dec 2025 Q1e; 1 mark]
**Q**: After Variant 4.4.1's update, predict for Team D (x₁=8, x₂=0.71, x₃=0.71).

**Solution**:
- z = 0.0935 + 0.1195(8) + 0.0803(0.71) + 0.1803(0.71)
- z = 0.0935 + 0.956 + 0.0570 + 0.1280 = **1.2342**
- h = 1 / (1 + e^(−1.2342)) ≈ 1 / (1 + 0.291) ≈ **0.7746**

**Interpretation**: Team D has **77.46%** probability of qualifying for the playoffs.

#### Variant 4.4.3 — Compute log-loss after iteration [Dec 2025 Q1d; 1 mark]
**Q**: Compute the BCE log-loss after the update.

**Formula**:
```
J = − (1/m) Σᵢ [ yᵢ log hᵢ + (1 − yᵢ) log(1 − hᵢ) ]
```

**Approach**: For each example use updated weights → compute z, h, then per-example penalty. Sum and divide by m.
(0.5 mark for stating the formula; 0.5 mark for the final value, which depends on full computation.)

### 4.5 Effect of weights / θ values

#### Variant 4.5.1 — Feature scaling for logistic regression [2023-Reg-2 Q3.2, Sample Q3.2; 2 marks]
**Q**: Importance of feature scaling for logistic regression with GD.

**Answer**: When using gradient descent, features must have similar scales — otherwise GD takes much longer to converge. Different scales → elongated cost contours → GD zig-zags rather than going straight to the optimum. With scaling, contours become roughly circular, GD descends efficiently.

### 4.6 J(θ) = 0 corollaries (logistic regression)

See Topic 3.7.1 for the full True/False question (asked in 2023-Reg-2 Q3 / Sample Q3).

---

## TOPIC 5 — Module 4b: Evaluation Metrics

### 5.1 Compute Precision/Recall/F1 from confusion matrix

#### Variant 5.1.1 — Medical diagnosis [Jan 2026 Q4; 5 marks]
**Q**: 1000 patients, 150 have disease. Model correctly identifies 120 diseased; flags 85 healthy as diseased.

**Solution**:

(a) **Confusion matrix**:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** (150) | 120 (TP) | 30 (FN) |
| **Actual Negative** (850) | 85 (FP) | 765 (TN) |

(b) **Metrics**:
- **Accuracy** = (120 + 765) / 1000 = 885/1000 = **88.5%**
- **Precision** = 120 / (120 + 85) = 120/205 ≈ **58.5%**
- **Recall (Sensitivity)** = 120 / (120 + 30) = 120/150 = **80.0%**
- **F1** = 2 × (0.585 × 0.80) / (0.585 + 0.80) = 0.936/1.385 ≈ **67.6%**
- **Specificity (TNR)** = 765 / (765 + 85) = 765/850 = **90.0%**

(c) **Cost-aware threshold tuning**:
- Cost of FN: 30 × $50,000 = **$1,500,000**
- Cost of FP: 85 × $2,000 = **$170,000**
- Total cost: **$1,670,000**

**Recommendation**: cost(FN) is **25× higher** than cost(FP). **Lower the classification threshold** to increase recall (catch more diseased patients), even at the expense of more false alarms.

#### Variant 5.1.2 — Three medical models [2024-Mkp Q4; 5 marks]
**Q**: Disease prevalence 2%; 10,000 test samples.

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| A | 180 | 200 | 20 | 9600 |
| B | 150 | 50 | 50 | 9750 |
| C | 190 | 500 | 10 | 9300 |

**Metrics**:

| Model | Precision | Recall | F1 | Specificity |
|---|---|---|---|---|
| A | 180/(180+200) = **47.4%** | 180/(180+20) = **90.0%** | **62.0%** | 9600/(9600+200) = **98.0%** |
| B | 150/(150+50) = **75.0%** | 150/(150+50) = **75.0%** | **75.0%** | 9750/(9750+50) = **99.5%** |
| C | 190/(190+500) = **27.5%** | 190/(190+10) = **95.0%** | **42.7%** | 9300/(9300+500) = **94.9%** |

**Choice**: For medical diagnosis, **false negatives are more costly** (missed disease). All three have high recall but Model B has the **best balance** with high precision (75%) and reasonable recall (75%) — minimizing both error types. Model C catches more cases (95% recall) at huge precision cost (27.5%, 500 FPs); Model A is between. (2 marks)

#### Variant 5.1.3 — Per-class metrics [2023-Reg Q4; 6 marks]

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 350 (TP) | 150 (FN) |
| **Actual Negative** | 200 (FP) | 800 (TN) |

(a) Compute Precision, Recall, F1 for the **negative class**:
- **Precision_neg** = TN / (TN + FN) = 800 / (800 + 150) = 800/950 ≈ **84.2%**
- **Recall_neg** = TN / (TN + FP) = 800 / (800 + 200) = 800/1000 = **80.0%**
- **F1_neg** = 2 × (0.842 × 0.80) / (0.842 + 0.80) = 1.347 / 1.642 ≈ **82.0%**

(b) Most informative metric: **F1-score** — balances precision and recall, gives a single summary even when class sizes differ.

(c) When **Precision** matters more: **spam filters, search results** — false positive (good email tagged spam) is costly.

(d) When **Recall** matters more: **cancer screening, fraud detection** — false negative (missed disease/fraud) is catastrophic.

#### Variant 5.1.4 — Two models compared [Dec 2025 Q4; 5 marks]

For two models on test data:

| | M1 Predicted + | M1 Predicted − |
|---|---|---|
| Actual + | 3 (TP) | 2 (FN) |
| Actual − | 1 (FP) | 4 (TN) |

| | M2 Predicted + | M2 Predicted − |
|---|---|---|
| Actual + | 1 (TP) | 4 (FN) |
| Actual − | 1 (FP) | 4 (TN) |

(a) Construct confusion matrices: as shown above.

(b) Compute metrics at threshold t = 0.5:

**M1**:
- Precision = 3/(3+1) = **0.75**
- Recall = 3/(3+2) = **0.60**
- F1 = 2(0.75)(0.60)/(0.75+0.60) = 0.90/1.35 ≈ **0.667**

**M2**:
- Precision = 1/(1+1) = **0.50**
- Recall = 1/(1+4) = **0.20**
- F1 = 2(0.50)(0.20)/(0.50+0.20) = 0.20/0.70 ≈ **0.286**

**Verdict**: M1 has higher precision, higher recall, and ~2.3× the F1 — clearly better.

#### Variant 5.1.5 — Fraud detection table [2023-Reg-2 Q4, Sample Q4; 4 marks]

|  | Predicted Fraud | Predicted Not Fraud |
|---|---|---|
| **Actual Fraud** | 60 (TP) | 0 (FN) |
| **Actual Not Fraud** | 120 (FP) | 20 (TN) |

**Metrics w.r.t. Fraud class**:
- Precision = 60/(60+120) = **33.33%**
- Recall = Sensitivity = 60/(60+0) = **100%**
- (Caught every fraud, but raised many false alarms.)

**Metrics w.r.t. Not-Fraud class**:
- Precision = 20/(20+0) = **100%**
- Recall = 20/(20+120) = **14.29%**
- (When says "not fraud", always right; but misses most legitimate transactions.)

**Observation**: Highly imbalanced/biased classifier — over-predicts fraud. Catches all real fraud but at huge cost (lots of false flags). Threshold needs to be raised, or model needs re-training with class weights.

### 5.2 Class-imbalance & accuracy paradox

#### Variant 5.2.1 — High-accuracy models compared [2024-Reg Q3b; 1.5+1 marks]
**Q**: Model A: 95% acc, Model B: 84% acc on test. Which is reliable for deployment? Other metrics?

**Answer**:
- Accuracy alone is misleading on imbalanced data (only 8% positives in original loan dataset).
- **Model A** could be predicting all-rejected → 95% acc but useless.
- **Model B** may actually be better if it identifies more approved (positive) loans correctly.
- Need: **Precision, Recall, F1** for the minority class; **Confusion Matrix**; **AUC-ROC**. Any 2 metrics get full credit.

#### Variant 5.2.2 — Why accuracy is poor on rare classes
**Q**: Disease prevalence 0.1%. Why is accuracy a poor metric?

**Answer**: A trivial "predict everyone healthy" classifier gets **99.9% accuracy** but **zero recall** — useless for finding sick patients. We need precision/recall/F1/AUC to capture true performance.

### 5.3 Cost-aware threshold tuning

See Variant 5.1.1c above. The general framework:

```
Total Cost = (#FN × cost_per_FN) + (#FP × cost_per_FP)
```

If cost_FN >> cost_FP → **lower** the threshold (more positive predictions, more recall).
If cost_FP >> cost_FN → **raise** the threshold (more conservative, more precision).

### 5.4 ROC / TPR / FPR computation

#### Variant 5.4.1 — TPR/FPR at two thresholds [practice Q34; 4 marks]
**Q**: 5 instances with true labels [1, 0, 1, 0, 1] and predicted positive-class probabilities [0.9, 0.7, 0.6, 0.3, 0.2]. Compute TPR and FPR at thresholds T > 0.8 and T > 0.5.

**Setup**: Total Positives P = 3, Total Negatives N = 2.

**Threshold T > 0.8**:
- Probability 0.9 > 0.8 → predict 1; true=1 → **TP**
- 0.7, 0.6, 0.3, 0.2 ≤ 0.8 → predict 0
  - true=0 (prob 0.7) → TN; true=1 (prob 0.6) → FN; true=0 (prob 0.3) → TN; true=1 (prob 0.2) → FN
- TP=1, FP=0, FN=2, TN=2.
- **TPR = 1/3 ≈ 0.33**, **FPR = 0/2 = 0.00**.

**Threshold T > 0.5**:
- Probabilities 0.9, 0.7, 0.6 > 0.5 → predict 1
  - true=1 (0.9) → TP; true=0 (0.7) → **FP**; true=1 (0.6) → TP
- 0.3, 0.2 ≤ 0.5 → predict 0
  - true=0 (0.3) → TN; true=1 (0.2) → FN
- TP=2, FP=1, FN=1, TN=1.
- **TPR = 2/3 ≈ 0.67**, **FPR = 1/2 = 0.50**.

**ROC points**: (FPR=0, TPR=0.33), (FPR=0.5, TPR=0.67). Plot these to start sketching the curve.

### 5.5 Effect of threshold on metrics

#### Variant 5.5.1 — Are metrics affected by threshold? [2023-Reg Q3b; 2 marks]
**Q**: Are Precision, Recall, Sensitivity, Specificity affected by changing the decision threshold? Justify with one calculation.

**Answer**: **Yes** — all four metrics depend on the threshold because changing it changes which examples become TP/FN/FP/TN.

- Lowering threshold → predict positive more often → recall ↑, precision ↓.
- Raising threshold → predict positive less often → precision ↑, recall ↓.

**Example calculation**: at T = 0.5, TP=40, FP=10, FN=20, TN=130 → Precision = 40/(40+10) = **0.80**. Raising T to 0.7 might give TP=30, FP=4, FN=30, TN=136 → Precision = 30/34 ≈ **0.88**, Recall = 30/(30+30) = **0.50**. Both shifted with threshold change.

### 5.6 Bonus metrics (good to mention)

- **AUC-ROC**: Area under TPR-vs-FPR curve. Threshold-independent. 1 = perfect, 0.5 = random.
- **AUC-PR**: For very imbalanced data, more informative than AUC-ROC (focuses on positive class).
- **Misclassification rate** = 1 − Accuracy.
- **Balanced accuracy** = (Recall + Specificity) / 2 — fair on imbalanced data.

---

## TOPIC 6 — Module 5: Decision Trees

### 6.1 Compute parent entropy + IG, pick root

#### Variant 6.1.1 — Loan repayment with Income split [Jan 2026 Q2; 6 marks]
**Q**: 14 samples, 9 Repay / 5 Default. Income splits: High = 6 Repay, 2 Default; Low = 3 Repay, 3 Default. Determine if Income is a good split.

**Solution**:

Step 1 — Parent entropy:
- p(Repay) = 9/14, p(Default) = 5/14.
- H(S) = −(9/14)·log₂(9/14) − (5/14)·log₂(5/14)
- log₂(9/14) ≈ −0.637, log₂(5/14) ≈ −1.485.
- H(S) ≈ −(0.643)(−0.637) − (0.357)(−1.485) ≈ 0.410 + 0.531 ≈ **0.94 bits**.

Step 2 — Child entropies:
- Income = High (8 samples, 6/2): H = −(6/8)log₂(6/8) − (2/8)log₂(2/8) = −0.75(−0.415) − 0.25(−2) = 0.311 + 0.500 ≈ **0.811**.
- Income = Low (6 samples, 3/3): H = −0.5(−1) − 0.5(−1) = **1.000**.

Step 3 — Weighted entropy after split:
- H_after = (8/14)(0.811) + (6/14)(1.000) = 0.464 + 0.429 ≈ **0.892**.

Step 4 — Information Gain:
- IG = 0.940 − 0.892 = **0.048**.

**Conclusion**: IG ≈ 0.05 — very low. **Income Level is NOT a good attribute** for splitting.

**Marking**: Parent entropy 1.5; child entropies 2; weighted 1; IG 1; interpretation 0.5.

#### Variant 6.1.2 — 2-class node split with attribute B [2024-Mkp Q6; 5 marks]
**Q**: Parent = {12, 8} (class 1, class 2). Split on attribute B: Left {9, 1}, Right {3, 7}.

**Solution**:

Step 1 — Parent entropy (20 samples):
- p₁ = 12/20 = 0.6, p₂ = 8/20 = 0.4.
- H(S) = −0.6 log₂(0.6) − 0.4 log₂(0.4) = −0.6(−0.737) − 0.4(−1.322) = 0.442 + 0.529 ≈ **0.971**.

Step 2 — Child entropies:
- Left (10 samples, 9/1): H = −0.9 log₂(0.9) − 0.1 log₂(0.1) = −0.9(−0.152) − 0.1(−3.322) = 0.137 + 0.332 ≈ **0.469**.
- Right (10 samples, 3/7): H = −0.3 log₂(0.3) − 0.7 log₂(0.7) = 0.521 + 0.360 ≈ **0.881**.

Step 3 — Weighted child entropy:
- (10/20)(0.469) + (10/20)(0.881) = 0.235 + 0.441 = **0.676**.

Step 4 — IG:
- IG(B) = 0.971 − 0.676 = **0.295**.

(c) **Limitation of IG**: it favours attributes with **many distinct values** (e.g., an "ID" column gives perfect splits → infinite IG, but useless for prediction). **Address by using Gain Ratio** (used in C4.5): GainRatio = IG / SplitInfo, which penalizes high-cardinality attributes.

#### Variant 6.1.3 — Choose root from 3 attributes [2023-Reg Q5; 5 marks]
**Q**: 15 students, performance ∈ {Excellent, Good, Poor} with 5/5/5. Three features: Exam Preparation Time, Attendance Rate, Participation. Use ID3 to find root.

**Solution**:
- Parent entropy (uniform 3-class): H(S) = log₂(3) ≈ **1.585** (using base-2).
- Compute IG for each attribute by partitioning the 15 examples by attribute values, getting child entropies, weighted-averaging them.
- (Per the answer key) Exam Preparation Time has weighted child entropy ≈ 0.260, so IG(Exam Prep) ≈ 1.585 − 0.260 = **1.325**, the largest.

**Conclusion**: **Exam Preparation Time** is selected as the root (highest IG, equivalently lowest weighted child entropy).

**Note (alternative answer key)**: For 3-class problems some answer keys allow log base 3 normalization (max H = 1). The relative ranking of attributes is unchanged.

#### Variant 6.1.4 — Choose root for "Crash Severity" [Endsem-style, 6 marks]
**Q**: 12 records, 8 Major / 4 Minor. Attributes: Weather Condition, Driver Condition, Rule Violation, Seat Belt.

**Solution**:

Parent entropy:
- H = −(8/12)log₂(8/12) − (4/12)log₂(4/12) ≈ 0.39 + 0.53 = **0.92**.

For each attribute, compute weighted child entropy (only the answers shown):
- Weather (Good vs Bad) → weighted 0.91 → **IG = 0.01**.
- Driver Condition → weighted 0.88 → IG = 0.04.
- Seat Belt → weighted 0.67 → IG = 0.25.
- Rule Violation → weighted 0.23 → IG = **0.69** (best).

**Choose Rule Violation as root** — highest IG.

#### Variant 6.1.5 — Customer purchasing decision tree [Dec 2025 Q3; 8 marks]
**Q**: 8 customers; columns Agegroup, Income Level, Occupation, Purchased.

(i) Calculate parent entropy.
(ii) Suppose Agegroup is the root. Compute entropy of "Young" branch. Is it pure? If not, which attribute splits it best?
(iii) Without full IG: which is more useful — Occupation or Income Level?
(iv) Classify a new customer {Young, Low, Professional} by majority vote in "Young" subset.

**Solution sketch**:
- (i) Parent entropy: count Yes/No, apply formula H = −p_yes log₂(p_yes) − p_no log₂(p_no).
- (ii) For "Young" subset of 4 (3 Yes, 1 No): H = −(3/4)log₂(3/4) − (1/4)log₂(1/4) ≈ **0.811**. Not pure (entropy ≠ 0). Split on **Income Level** since it perfectly separates the 4 instances (Low → No, Medium → Yes, High → Yes).
- (iii) **Income Level** is better. Reason: it perfectly separates classes (Low → No; Medium/High → Yes), while Occupation gives mixed results in every category.
- (iv) Majority vote in "Young" = 3 Yes vs 1 No → **Yes**.

#### Variant 6.1.6 — Funding vs Team Size [Practice Q1]
**Q**: 8 startups, 4 Profitable / 4 Not. Compute IG for "Funding" (3 levels: High/Med/Low) and "Team Size" (Small/Large). Pick root.

**Solution**:

Parent entropy: 4/4 → H(S) = **1**.

Funding splits:
- High (3 records, 3/0): H = **0**.
- Medium (3 records, 1/2): H = −(1/3)log₂(1/3) − (2/3)log₂(2/3) ≈ **0.918**.
- Low (2 records, 0/2): H = **0**.
- Weighted = (3/8)(0) + (3/8)(0.918) + (2/8)(0) = **0.344**.
- **IG(Funding) = 1 − 0.344 = 0.656**.

Team Size splits:
- Small (4 records, 2/2): H = **1**.
- Large (4 records, 2/2): H = **1**.
- Weighted = (4/8)(1) + (4/8)(1) = **1**.
- **IG(Team Size) = 1 − 1 = 0**.

**Pick Funding as root** (IG 0.656 vs 0).

### 6.2 Continuous attribute thresholding (C4.5)

#### Variant 6.2.1 — Test Score split [Practice Q5; 4-6 marks]
**Q**: TestScore: 55(No), 60(No), 75(Yes), 80(Yes), 85(No), 95(Yes). Find best threshold for binary split.

**Solution**:

Step 1 — Find candidate thresholds (midpoints where class label changes):
- 60 → 75 (No → Yes): T₁ = (60 + 75)/2 = **67.5**.
- 80 → 85 (Yes → No): T₂ = (80 + 85)/2 = **82.5**.
- 85 → 95 (No → Yes): T₃ = (85 + 95)/2 = **90.0**.

Step 2 — Parent entropy: 6 samples, 3/3 → H = **1.0**.

Step 3 — IG for each threshold:

**T₁ = 67.5**:
- Left (≤ 67.5): {55, 60} → 2 No → H = 0.
- Right (> 67.5): {75, 80, 85, 95} → 3 Yes, 1 No → H = −(3/4)log₂(3/4) − (1/4)log₂(1/4) ≈ 0.811.
- Weighted = (2/6)(0) + (4/6)(0.811) = 0.541.
- IG(T₁) = 1 − 0.541 = **0.459**.

**T₂ = 82.5**:
- Left (≤ 82.5): {55, 60, 75, 80} → 2 No, 2 Yes → H = 1.
- Right (> 82.5): {85, 95} → 1 No, 1 Yes → H = 1.
- Weighted = (4/6)(1) + (2/6)(1) = 1.
- IG(T₂) = 1 − 1 = **0**.

**T₃ = 90.0**:
- Left (≤ 90): {55, 60, 75, 80, 85} → 3 No, 2 Yes → H = −(3/5)log₂(3/5) − (2/5)log₂(2/5) ≈ 0.971.
- Right (> 90): {95} → 1 Yes → H = 0.
- Weighted = (5/6)(0.971) + (1/6)(0) = 0.809.
- IG(T₃) = 1 − 0.809 = **0.191**.

**Best split: TestScore ≤ 67.5** (IG = 0.459).

#### Variant 6.2.2 — General continuous-attribute procedure [Practice Q2 conceptual]
**Q**: How does C4.5 handle a continuous attribute like AverageMonthlySpend?

**Steps**:
1. Sort all unique values of the attribute.
2. For each adjacent pair where the **class label changes**, compute the midpoint as a candidate threshold.
3. For each candidate T, partition into (≤ T) vs (> T) and compute IG (or Gain Ratio).
4. Pick the threshold T with the **highest gain**.
5. The continuous attribute can be reused at deeper splits with different thresholds.

### 6.3 Pre vs Post pruning

#### Variant 6.3.1 — Compare strategies [Practice Q4; 4 marks]
**Q**: Compare pre-pruning and post-pruning. When is post-pruning more effective?

**Comparison**:

| | Pre-pruning (early stopping) | Post-pruning |
|---|---|---|
| When | During training | After full tree is grown |
| How | Stop at depth/sample/IG threshold | Snip subtrees that don't help on validation |
| Pros | Fast, simple | Often gives better accuracy |
| Cons | Greedy / myopic — may stop too early on patterns that need depth (XOR-like) | Slower; needs validation set |

**Scenario where post-pruning wins**: XOR-like feature interaction. Single-feature splits show low IG → pre-pruning stops. But post-pruning lets the tree grow → second split reveals the XOR pattern → keep that subtree.

#### Variant 6.3.2 — Why training-set accuracy alone is detrimental [2023-Reg-2 Q5b, Sample Q5b; 2 marks]
**Q**: "Assessing decision tree using only training data is detrimental." Justify.

**Answer**: A fully-grown tree can hit **100% training accuracy** by memorising every example, including noise. This says nothing about generalisation. Always evaluate on a **held-out validation/test set** that the tree never saw. Otherwise, overfitting goes undetected and the deployed model performs poorly.

**Marking**: 1 mark for the reason; 1 mark for an example showing training/test split with diverse distributions.

### 6.4 Limitations of IG / Gain Ratio

#### Variant 6.4.1 — IG bias and fix [2024-Mkp Q6c; 2 marks]
**Q**: Limitations of Information Gain and how to address them.

**Answer**:
- **Limitation**: IG favours attributes with **many distinct values** (e.g., ID, Date, Phone Number). These split the data into many small pure groups → huge IG → useless for prediction (memorisation).
- **Fix**: **Gain Ratio** (used in C4.5):
  ```
  GainRatio(A) = IG(A) / SplitInfo(A)
  SplitInfo(A) = − Σᵥ (|Sᵥ|/|S|) · log₂(|Sᵥ|/|S|)
  ```
  SplitInfo grows when an attribute fragments the data into many parts → divides IG back down → discourages high-cardinality splits.

#### Variant 6.4.2 — Missing attribute handling
**Q**: 15% of CustomerAge values are missing. How does C4.5 handle them at training and prediction?

**Answer**:
- **Training**: distribute missing-attribute examples **fractionally** down each branch in proportion to known examples' split. E.g., if 60/85 known are >30 and 25/85 are ≤30, then each missing example sends 60/85 fraction down >30 branch and 25/85 down ≤30. IG is computed using these fractional counts.
- **Prediction**: send the missing-attribute example **down all branches** simultaneously; final prediction = weighted average of leaf predictions, weighted by branch proportions.

### 6.5 Robustness — comparing models on noise vs outliers

#### Variant 6.5.1 — Noise + Outliers grid [Jan 2026 Q6; 6 marks]
**Q**: For Linear Regression (L1), Logistic Regression (L2), Decision Tree (T): assess effect of (A) noisy/irrelevant features and (B) outliers.

**Dataset A — Noisy / Irrelevant Features**:

| Model | Affected? | Justification |
|---|---|---|
| L1: Linear Regression | Affected | Fits a non-zero weight to every feature; noise causes spurious correlations → reduces train error but increases variance → poor test performance. |
| L2: Logistic Regression | Affected | Same — noise distorts the decision boundary as model finds spurious patterns. Overfits → lower test accuracy. |
| T: Decision Tree | **Strongly affected** | Greedy splits on **any** feature with even small impurity reduction. Noise creates extra branches that fit random quirks → very high train accuracy but **high variance**. |

**Dataset B — Outliers**:

| Model | Affected? | Justification |
|---|---|---|
| L1: Linear Regression | **Strongly affected** | MSE squares errors → outliers pull the regression line toward themselves, distorting both train and test fit. |
| L2: Logistic Regression | Less affected (not immune) | Sigmoid saturates near 0 and 1 → extreme points have bounded influence. Single outliers don't dominate as much. Boundary may shift slightly. |
| T: Decision Tree | Moderately–strongly affected | Outliers can cause unusual splits (a separate branch for a few extreme points). Trees are less globally pulled, but locally overfit those rare cases. |

**Memorise pattern**:
- DT: weakest against **noise**; manages outliers OK.
- Linear Regression: weakest against **outliers** (squared loss).
- Logistic Regression: most robust overall thanks to sigmoid saturation.

### 6.6 Decision tree implications & best practices

#### Variant 6.6.1 — Overfitting implications & mitigations [2024-Reg Q1b; 4 marks]
**Q**: Implications of overfitting on model performance? How can regularization and early stopping help, in the context of linear regression or logistic regression?

**Answer**:

Implications of overfitting:
- Loss of generalisability — high training acc, low test acc.
- Unreliable predictions on unseen data.
- Increased model complexity → harder to interpret, larger memory, slower inference.

Regularization (in linear/logistic regression):
- Adds a penalty term to the loss function (Σ θ² for L2, Σ |θ| for L1).
- Discourages overly complex models by constraining the magnitude of weights.
- Result: smoother predictions, better generalization.

Early stopping (in iterative training):
- Monitors validation loss during training.
- Halts when validation loss starts increasing while training loss continues to decrease.
- Prevents the model from learning noise.

#### Variant 6.6.2 — Strengths & weaknesses of DT
**Q**: List 3 strengths and 3 weaknesses of decision trees.

**Strengths**:
- Highly interpretable ("white-box") — each path is a rule.
- No need to scale features.
- Handle mixed types (numerical + categorical) and missing values gracefully.

**Weaknesses**:
- Greedy splits may miss the globally best tree.
- Unstable: small changes in data can re-shape the entire tree.
- Axis-parallel decision boundaries only — can't draw slants.
- Overfit if not pruned.

---

## TOPIC 7 — Cross-cutting questions

### 7.1 Polynomial regression overfitting [Jan 2026 Q1; 3 marks]
**Q**: Manager says "always pick higher-order polynomial since it gives lower training error." Critique.

**Answer**:
(a) Higher-order models can **memorize** training data → very low training error but high validation error.
(b) On unseen data → poor generalization (overfitting).
(c) Three ways to ensure generalization:
1. **Regularization** (Ridge/Lasso) — penalize large weights.
2. **Cross-validation** — pick model complexity that gives best held-out score.
3. **Early stopping** — train iteratively and stop before overfitting begins.
4. (Bonus) **Simplify model** (lower polynomial degree); **data augmentation**; **collect more data**.

### 7.2 Comparing linear models with regularization [2024-Mkp Q3; 6 marks]
**Q**: Real estate prediction; 4 models tested (see §3.4.1). Analyze:

A) Bias-variance for Models A and B → see §3.4.1 above.

B) Why does Model B's training error drop but test error rise after iteration 5?
- The model is **memorising training data** (learning noise) → overfitting.
- Test error rises because patterns fit don't generalize.
- Helpful technique (excluding L1/L2): **Early stopping** (halt at iteration 5).

C) R² role:
- Measures goodness of fit — proportion of variance in y explained by the model.
- Model D has highest R² (0.79) → best generalisation.

D) Reduce bias of Model C (Ridge):
- **Decrease λ** → less regularization → more flexibility → lower bias (but higher variance trade-off).

E) Justification of feature counts:
- Model C (Ridge): retains all 120 features, just shrinks coefficients.
- Model D (Lasso): zeros 97 coefficients, keeps 23 most informative.

F) Improve Model A:
- Suffers from high bias (only 3 features). **Add more features** (use all 15 instead of 3); explore polynomial features for non-linearity. Lower bias.

---

## APPENDIX A — Useful constants and log₂ values

| x | log₂(x) | x | log₂(x) |
|---|---|---|---|
| 1/16 | −4 | 1 | 0 |
| 1/8 | −3 | 2 | 1 |
| 1/4 | −2 | 3 | 1.585 |
| 1/3 | −1.585 | 4 | 2 |
| 1/2 | −1 | 5 | 2.322 |
| 2/3 | −0.585 | 6 | 2.585 |
| 3/4 | −0.415 | 7 | 2.807 |
| 4/5 | −0.322 | 8 | 3 |
| 5/8 | −0.678 | 9 | 3.170 |
| 3/8 | −1.415 | 10 | 3.322 |
| 9/14 | −0.637 | 14 | 3.807 |
| 5/14 | −1.485 | 16 | 4 |

**Useful entropy values**:
- H(50/50) = 1 bit
- H(75/25) ≈ 0.811
- H(80/20) ≈ 0.722
- H(90/10) ≈ 0.469
- H(uniform over K) = log₂(K)
- H(9/14, 5/14) ≈ 0.940 (the famous tennis-data entropy)

---

## APPENDIX B — Formula sheet (final exam reference)

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
  GD update:    θⱼ ← θⱼ − α·(1/m) Σ(hᵢ − yᵢ) xᵢⱼ
  Log-odds:     log(p/(1−p)) = θᵀx
  Odds change per Δxⱼ = 1: ×exp(θⱼ)

REGULARIZATION
  Ridge (L2):   J + λ Σ θⱼ²        (shrinks all)
  Lasso (L1):   J + λ Σ |θⱼ|       (zeros some — sparsity)

PREPROCESSING
  Z-score:      v' = (v − μ)/σ
  Min-Max:      v' = (v − min)/(max − min)
  IQR fences:   [Q1 − 1.5·IQR, Q3 + 1.5·IQR]
  3-sigma:      [μ − 3σ, μ + 3σ]

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

---

## APPENDIX C — Final exam-day checklist

**5 minutes before**:
- Refresh the formula sheet (Appendix B).
- Skim the 8.5 Kill List in the scoring guide.

**As soon as you receive the paper**:
- Read all questions; mark difficulty (easy / medium / hard).
- Tackle easiest 2 first; build confidence and bank time.

**For every numerical**:
- Write the formula explicitly.
- Substitute step-by-step; show every intermediate calculation.
- State the final answer with units / interpretation.

**For every theory question**:
- Define key terms.
- Cite formulas where relevant.
- Connect back to the scenario / dataset in the question.

**Time-keeping**: ~15 minutes per question average. **Strict 10-minute reserve** for review.

---

**Best of luck — drill these patterns and the midsem becomes a recognition test, not a discovery one.**