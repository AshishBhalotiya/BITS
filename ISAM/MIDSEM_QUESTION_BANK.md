# ISM Mid-Semester Question Bank — Solved Variants by Topic

**Course:** AIMLCZC418 — Introduction to Statistical Methods
**Scope:** Sessions 1–7 (with primary emphasis on Sessions 1–6).

This bank compiles **every distinct question variant** from past mid-semester papers (2023 Regular, 2023 Makeup, 2024 Regular, 2024 Makeup, 2024 Makeup-2, 2025 Regular, Dec 2025 Regular, Jan 2026 Makeup), the practice problem set, and webinar notes. Each problem is fully worked, organised by topic so you can practise an entire weakness in a single sitting. All math is in **plain Unicode** for universal Markdown rendering.

---

## How to use this bank

- **Skim section headers** to spot any topic you're shaky on.
- **Try each problem under timed conditions** before peeking at the solution.
- Cross-check your numerical answers; the marker rewards step-by-step working, so practise writing out every step.
- Recurring problem types are tagged with the papers they appeared in (e.g., *Dec 2025 Q1*).

---

## TABLE OF CONTENTS

1. Descriptive Statistics — Section 1 (10 problems)
2. Probability Rules: Set Ops, Independence, Inclusion-Exclusion — Section 2 (10)
3. Conditional Probability & Total Probability — Section 3 (8)
4. Bayes' Theorem — Section 4 (10)
5. Naïve Bayes Classifier — Section 5 (8)
6. Discrete Random Variables (PMF, E[X], Var(X)) — Section 6 (10)
7. Continuous Random Variables (PDF, CDF, expectations) — Section 7 (10)
8. Joint Distributions (joint PMF/PDF, marginals, independence) — Section 8 (8)
9. Binomial & Poisson Distributions — Section 9 (10)
10. Normal Distribution & Approximations — Section 10 (10)
11. Sampling, CLT, Sample Size — Section 11 (6)

---

## SECTION 1 — DESCRIPTIVE STATISTICS (Q1 in every paper)

### Q1.1 — Fast-food fat content (*Dec 2025 Regular Q1*)

**Data:** 7, 4, 20, 23, 25, 19, 30, 30, 40, 56.

(a) Mean, median, sample SD, IQR; identify outliers via 1.5 × IQR.
(b) Comment on central tendency, dispersion, variability for nutritional consistency.

**Solution.**

Sorted: 4, 7, 19, 20, 23, 25, 30, 30, 40, 56. n = 10.

- Mean = 254/10 = **25.4 g**.
- Median = (23 + 25)/2 = **24 g**.
- Q1 (median of {4,7,19,20,23}) = **19**.
- Q3 (median of {25,30,30,40,56}) = **30**.
- IQR = 30 − 19 = **11**.
- Fences: 19 − 16.5 = 2.5; 30 + 16.5 = 46.5. **Outlier: 56**.
- Sum of squared deviations: 458.16 + 338.56 + 40.96 + 29.16 + 5.76 + 0.16 + 21.16 + 21.16 + 213.16 + 936.36 = 2064.6.
- s² = 2064.6/9 ≈ 229.4. **s ≈ 15.15 g**.

(b) The mean (25.4 g) exceeds the median (24 g) by ~1.4 g, indicating slight right-skew driven by 56 g. The IQR of 11 g shows the middle 50% of sandwiches cluster between 19 and 30 g, but the outlier (56 g) signals a single chain whose product is far fattier than the rest. From a quality-control perspective, fat content in chicken sandwiches across chains is **not consistent**; reducing variability would require reformulation by the high-fat brand, since one extreme value moves the brand-wide average noticeably.

---

### Q1.2 — Production output stability (*Jan 2026 Makeup Q1*)

**Data (units/day):** 190, 250, 240, 270, 230, 280, 245, 265, 285, 380.

(a) Five-number summary; identify outliers.
(b) Comment on stability/consistency.

**Solution.**

Sorted: 190, 230, 240, 245, 250, 265, 270, 280, 285, 380.

Using (n+1)/4 percentile method:
- Q1 position = 0.25·(11) = 2.75 → between 230 and 240. Q1 = 230 + 0.75·10 = **237.5**.
- Median = (250 + 265)/2 = **257.5**.
- Q3 position = 0.75·11 = 8.25 → Q3 = 280 + 0.25·5 = **281.25**.
- Min = 190; Max = 380.

- IQR = 281.25 − 237.5 = 43.75.
- Upper fence = 281.25 + 1.5·43.75 = **346.875**. → 380 > 346.875, **380 is an outlier**.
- Lower fence = 237.5 − 65.625 = 171.875. No low outlier.

(b) Output for 9 of the 10 days lies between 190–285, suggesting a generally stable post-automation production. The 380-unit day is a special-cause event — perhaps an extra shift, double-shift, or data-entry error. Investigation is needed before accepting the 380-day as routine. Aside from the outlier, daily variation is moderate.

---

### Q1.3 — API response times (*Practice Set Q1*)

**Data (ms):** 12, 15, 18, 14, 15, 12, 85, 16, 15, 19, 17.

(a) Five-number summary.
(b) Outliers via 1.5×IQR.
(c) Mean and SD.
(d) Comment on skew.

**Solution.**

Sorted: 12, 12, 14, 15, 15, 15, 16, 17, 18, 19, 85. n = 11.

- Min = 12; Max = 85.
- Median = 6th value = **15**.
- Lower half = {12, 12, 14, 15, 15} → Q1 = **14**.
- Upper half = {16, 17, 18, 19, 85} → Q3 = **18**.
- IQR = 4. Fences: 14 − 6 = 8; 18 + 6 = 24. → **85 is an outlier**.
- Mean = 238/11 ≈ **21.64**.
- Σx² = 144 + 144 + 196 + 225 + 225 + 225 + 256 + 289 + 324 + 361 + 7225 = 9614. (recompute: actually 144+144+196+225+225+225+256+289+324+361+7225 = 9614)
- s² = (Σx² − n·x̄²)/(n − 1) = (9614 − 11·468.5)/10 = (9614 − 5153.5)/10 ≈ 446.05. s ≈ **21.12**.
- Mean (21.64) ≫ median (15) → **strongly positively skewed** due to outlier 85.

---

### Q1.4 — SpaceX launch times (*2024 Regular Q1b*)

**Data (hours):** 9, 195, 241, 301, 216, 260, 7, 244, 192, 147, 10, 295, 142.

Find mean and SD.

**Solution.**

n = 13.
Sum = 9+195+241+301+216+260+7+244+192+147+10+295+142 = **2259**.
Mean = 2259/13 ≈ **173.77 hours**.

Σ(xᵢ − 173.77)² = (164.77)² + (21.23)² + (67.23)² + (127.23)² + (42.23)² + (86.23)² + (166.77)² + (70.23)² + (18.23)² + (26.77)² + (163.77)² + (121.23)² + (31.77)²
= 27149 + 451 + 4520 + 16188 + 1783 + 7436 + 27812 + 4932 + 332 + 717 + 26821 + 14697 + 1009 ≈ 133846.

s² = 133846/12 ≈ 11154. s ≈ **105.61 hours**.

The huge SD reflects bimodal mission lengths (very short 7–10 h vs very long 142–301 h).

---

### Q1.5 — Astronaut training times (*2025 Regular Q1b*)

**Data (min):** 18, 25, 19, 32, 27, 22, 29, 30, 24, 26.

Find mean, SD; interpret variability and central tendency.

**Solution.**

Sum = 252. Mean = **25.2 min**.

Squared deviations: 51.84 + 0.04 + 38.44 + 46.24 + 3.24 + 10.24 + 14.44 + 23.04 + 1.44 + 0.64 = 189.6.

s² = 189.6/9 ≈ 21.07. s ≈ **4.59 min**.

Central tendency: typical astronaut takes ~25 min.
Variability: most times within ±5 min of the mean (i.e., ~20–30 min). Reasonable consistency, but a 14-min spread (18 → 32) suggests some performers are notably faster/slower.

---

### Q1.6 — Eleven weights (*2025 Regular Q1a*)

**Data (kg):** 60, 72, 65, 68, 70, 100, 62, 75, 78, 80, 83.

Find Q1, Q2, Q3, IQR; identify outliers.

**Solution.**

Sorted: 60, 62, 65, 68, 70, 72, 75, 78, 80, 83, 100. n = 11.

- Median = 6th value = **72**.
- Lower half = {60, 62, 65, 68, 70} → Q1 = **65**.
- Upper half = {75, 78, 80, 83, 100} → Q3 = **80**.
- IQR = 80 − 65 = **15**.
- Lower fence = 65 − 22.5 = 42.5; Upper = 80 + 22.5 = 102.5.
- All data in [42.5, 102.5]. **No outliers** (100 is just below the upper fence).

---

### Q1.7 — Student marks symmetry (*2024 Makeup Q1*)

**Data (out of 30):** 28, 12, 10, 22, 18, 18, 17, 15, 16, 15, 14, 15, 16, 17, 18.

Find mean, 5-number summary; comment on symmetry.

**Solution.**

Sorted: 10, 12, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 18, 22, 28. n = 15.

- Sum = 251 → Mean = 251/15 ≈ **16.73**.
- Median = 8th value = **16**.
- Q1 = median of lower 7 ({10,12,14,15,15,15,16}) = 15.
- Q3 = median of upper 7 ({17,17,18,18,18,22,28}) = 18.
- Min = 10, Max = 28.

5-number summary: (10, 15, 16, 18, 28).

Mean (16.73) > Median (16) → **slightly positively skewed**. The high-end 28 pulls the mean up.

---

### Q1.8 — Four-dataset consistency (*2023 Regular Q1*)

Given summary table for HHV (n=908, mean 150, SD 10, 5-num: 30, 90, 120, 130, 160), WBN (mean 65, SD 5, range 15–75 with quartiles 25, 65, 70), BNC (mean 150, SD 8, 5-num: 75, 90, 125, 135, 180), HBCN (mean 68, SD 4, 5-num: 30, 30, 60, 63, 90).

Provide three inferences.

**Solution.**

(1) **Consistency via CV** = (SD/mean)×100:
- HHV = 10/150·100 ≈ 6.67%.
- WBN = 5/65·100 ≈ 7.69%.
- BNC = 8/150·100 ≈ 5.33%.
- HBCN = 4/68·100 ≈ 5.88%.
→ **BNC is most consistent** (lowest CV).

(2) **Skew comparison** (mean vs median):
- HHV: 150 vs 120 → positive skew.
- WBN: 65 vs 65 → symmetric.
- BNC: 150 vs 125 → positive skew.
- HBCN: 68 vs 60 → positive skew.
→ Only WBN is symmetric; the others lean right.

(3) **Outlier check** via 1.5·IQR fences:
- HHV: IQR = 130 − 90 = 40; fences [30, 190]. Min = 30 sits exactly on the lower fence (potentially marginal outlier).
- WBN: IQR = 70 − 25 = 45; fences [−42.5, 137.5]. No outliers (data in [15, 75]).
- BNC: IQR = 135 − 90 = 45; fences [22.5, 202.5]. No outliers (range [75, 180]).
- HBCN: IQR = 63 − 30 = 33; fences [−19.5, 112.5]. No outliers (range [30, 90]).
→ Only HHV has a borderline minimum.

---

### Q1.9 — Time-to-get-ready (*Practice Set 1.1*)

**Data (min):** 39, 29, 43, 52, 39, 44, 40, 31, 44, 35.

Find mean, median, mode; range, variance, SD; comment on skew.

**Solution.**

n = 10.
Sum = 396. **Mean = 39.6**. Median = (39 + 40)/2 = **39.5**.
Modes: 39 and 44 (each appears twice) → **bimodal**.

Range = 52 − 29 = **23**.

Σ(xᵢ − 39.6)² = 112.36 + 73.96 + 21.16 + 0.36 + 0.36 + 0.16 + 11.56 + 19.36 + 19.36 + 153.76 = 412.4.
s² = 412.4/9 ≈ **45.82**. s ≈ **6.77 min**.

Mean ≈ Median → essentially symmetric (slight right-lean).

---

### Q1.10 — Outlier-resistant median (*Practice Set 1.8*)

Dataset 1: $50k, 52k, 55k, 58k, 60k.
Dataset 2: $50k, 52k, 55k, 58k, 150k.

Compute mean and median for each; comment.

**Solution.**

| | Sum | Mean | Median |
|---|---|---|---|
| D1 | 275 | $55k | $55k |
| D2 | 365 | $73k | $55k |

The single outlier ($150k) drags the mean from 55k → 73k while the median is unmoved. **For Dataset 2, the median is the more honest measure of central tendency.**

---

## SECTION 2 — PROBABILITY RULES, SET OPS, INDEPENDENCE

### Q2.1 — Independent vs mutually exclusive validation (*2023 Regular Q2*)

A and B independent with P(A) = 0.35, P(B) = 0.30. Validate or refute:

(a) P(A ∪ B) = 0 because they are independent.
(b) P(A ∩ B) = 0 because they are mutually exclusive.
(c) Find P(Aᶜ ∩ Bᶜ).

**Solution.**

P(A ∩ B) = 0.35·0.30 = 0.105 (since independent).

(a) FALSE. P(A ∪ B) = 0.35 + 0.30 − 0.105 = **0.545**, not 0. Independence does not eliminate the union.
(b) FALSE. P(A ∩ B) = 0.105 ≠ 0; A and B can both occur.
(c) Aᶜ and Bᶜ also independent (complements of independents). P(Aᶜ) = 0.65; P(Bᶜ) = 0.70. P(Aᶜ ∩ Bᶜ) = 0.65·0.70 = **0.455**.

---

### Q2.2 — Six-part independence drill (*2023 Makeup Q1*)

A, B independent, P(A) = 0.25, P(B) = 0.20. Find:

(a) P(A ∪ B); (b) P(A ∩ B); (c) P(A | B); (d) P(B | A); (e) P(Aᶜ | B); (f) P(Aᶜ ∩ Bᶜ).

**Solution.**

(a) P(A ∪ B) = 0.25 + 0.20 − 0.05 = **0.40**.
(b) P(A ∩ B) = 0.25·0.20 = **0.05**.
(c) P(A | B) = P(A) (independent) = **0.25**.
(d) P(B | A) = P(B) = **0.20**.
(e) P(Aᶜ | B) = 1 − P(A | B) = **0.75**.
(f) P(Aᶜ ∩ Bᶜ) = (1 − 0.25)(1 − 0.20) = 0.75·0.80 = **0.60**.

---

### Q2.3 — Set table reconstruction (*2024 Makeup-2 Q1b*)

Given P(M ∪ Q) = 0.40, P(only M) = 0.15, P(only Q) = 0.10. Find P(M ∩ Q), P(M), P(Q); validate consistency.

**Solution.**

P(M ∪ Q) = P(only M) + P(only Q) + P(M ∩ Q) ⇒ 0.40 = 0.15 + 0.10 + P(M ∩ Q) ⇒ **P(M ∩ Q) = 0.15**.

P(M) = P(only M) + P(M ∩ Q) = 0.15 + 0.15 = **0.30**.
P(Q) = P(only Q) + P(M ∩ Q) = 0.10 + 0.15 = **0.25**.

Consistency check: P(M ∪ Q) ≤ P(M) + P(Q) → 0.40 ≤ 0.55 ✓. All values in [0,1] ✓. **Consistent**.

---

### Q2.4 — Three-event inclusion-exclusion (*Dec 2025 Q6*)

Three independent weather alerts A, B, C; P(A) = P(B) = P(C) = p; P(A ∩ B) = P(B ∩ C) = P(A ∩ C) = 1/8; P(A ∩ B ∩ C) = 1/16; P(Aᶜ ∩ Bᶜ ∩ Cᶜ) = 1/4.

Find: (a) P(A); (b) P(exactly two); (c) P(at least one); (d) P(exactly one).

**Solution.**

By De Morgan: P(at least one) = 1 − P(none) = 1 − 1/4 = **3/4**.

By inclusion-exclusion:
3p − 3·(1/8) + 1/16 = 3/4.
3p = 3/4 + 3/8 − 1/16 = 12/16 + 6/16 − 1/16 = 17/16.
p = **17/48** ≈ 0.354.

(b) P(exactly two) = ΣP(pair) − 3·P(all three) = 3·(1/8) − 3·(1/16) = 3/8 − 3/16 = **3/16** = 0.1875.

(c) **3/4 = 0.75** (computed above).

(d) P(exactly one) = P(at least one) − P(exactly two) − P(all three) = 3/4 − 3/16 − 1/16 = 12/16 − 4/16 = **8/16 = 1/2**.

---

### Q2.5 — Three-set probability puzzle (*2023 Regular Q3*)

P(A) = 0.50, P(B) = 0.55, P(C) = 0.45, P(A ∩ B) = 0.20, P(A ∩ C) = 0.20, P(B ∩ C) = 0.15, P(A ∩ B ∩ C) = 0.05.

Find: (a) P(A | A ∪ B); (b) P(B | A ∩ B); (c) P(A ∩ B | A ∪ B); (d) P(A ∪ B | A ∩ B ∩ C).

**Solution.**

P(A ∪ B) = 0.50 + 0.55 − 0.20 = 0.85.

(a) P(A | A ∪ B) = P(A ∩ (A ∪ B)) / P(A ∪ B) = P(A) / P(A ∪ B) = 0.50/0.85 ≈ **0.588**.
(b) P(B | A ∩ B) = P(B ∩ (A ∩ B)) / P(A ∩ B) = P(A ∩ B) / P(A ∩ B) = **1**. (B is certain given A ∩ B.)
(c) P(A ∩ B | A ∪ B) = P((A ∩ B) ∩ (A ∪ B)) / P(A ∪ B) = P(A ∩ B) / P(A ∪ B) = 0.20/0.85 ≈ **0.235**.
(d) Since A ∩ B ∩ C ⊆ A ∪ B, P(A ∪ B | A ∩ B ∩ C) = **1**.

---

### Q2.6 — Brake-disc defects inclusion-exclusion (*Jan 2026 Q6b*)

n = 10000 discs; W = 1200, C = 850, S = 1500; W∩C = 300, W∩S = 400, C∩S = 250; W∩C∩S = 100.

(i) P(at least one defect)?
(ii) Number defect-free?

**Solution.**

|W ∪ C ∪ S| = 1200 + 850 + 1500 − 300 − 400 − 250 + 100 = **2700**.

(i) P(at least one) = 2700/10000 = **0.27 (27%)**.
(ii) Defect-free = 10000 − 2700 = **7300 discs**.

---

### Q2.7 — Workplace honesty contingency (*2024 Regular Q2*)

P(quiet) = 0.35; P(sick when well | quiet) = 0.75; P(quiet | sick when well) = 0.40.

Find: (a) P(both); (b) P(at least one); (c) P(not quiet | sick); (d) P(neither).

**Solution.**

P(both) = P(quiet) · P(sick | quiet) = 0.35·0.75 = **0.2625**.

P(sick when well) from P(quiet | sick) = P(both)/P(sick) → 0.40 = 0.2625 / P(sick) → **P(sick) = 0.65625**.

(b) P(at least one) = 0.35 + 0.65625 − 0.2625 = **0.74375**.
(c) P(not quiet | sick) = 1 − 0.40 = **0.60**.
(d) P(neither) = 1 − 0.74375 = **0.25625**.

---

### Q2.8 — Truth-tellers contradiction (*Companion S2 Practice 3*)

A truthful 80%, B truthful 60%; chance they contradict?

**Solution.**

Contradiction = exactly one lies = P(A truthful) · P(B lies) + P(A lies) · P(B truthful)
= 0.8·0.4 + 0.2·0.6 = 0.32 + 0.12 = **0.44 (44%)**.

---

### Q2.9 — Newspaper Venn (*Practice Set 1.10*)

40% read A, 25% read B, 15% both. Find: (a) P(at least one); (b) P(exactly one); (c) P(B | A).

**Solution.**

(a) 0.40 + 0.25 − 0.15 = **0.50**.
(b) 0.50 − 0.15 = **0.35**.
(c) 0.15 / 0.40 = **0.375**.

---

### Q2.10 — Two dice — sum and 6-on-die (*Practice Set 1.5*)

A: sum > 8; B: at least one 6.

(a) P(A); (b) P(B); (c) P(A | B).

**Solution.**

|A|: sums 9,10,11,12 → 4 + 3 + 2 + 1 = 10. P(A) = **10/36 = 5/18**.
|B|: 6 + 5 = 11 outcomes. P(B) = **11/36**.
|A ∩ B|: from A's 10 outcomes, those containing a 6: (3,6), (6,3), (4,6), (6,4), (5,6), (6,5), (6,6) = **7**.
P(A | B) = 7/11.

---

## SECTION 3 — CONDITIONAL & TOTAL PROBABILITY

### Q3.1 — Loan default conditional table (*2024 Makeup-2 Q3a*)

|  | Low | Middle | High | Total |
|--|-----|--------|------|-------|
| No default | 8000 | 15000 | 7000 | 30000 |
| Yes default | 2000 | 5000 | 1000 | 8000 |
| Total | 10000 | 20000 | 8000 | 38000 |

(i) P(no default | Middle).
(ii) P(Middle | no default).

**Solution.**

(i) 15000/20000 = **0.75 (75%)**.
(ii) 15000/30000 = **0.50 (50%)**.

(Same numerator, different denominators — pick the right "world".)

---

### Q3.2 — Spam through 3 accounts (*2024 Regular Q3b*)

P(A₁) = 0.70, P(A₂) = 0.20, P(A₃) = 0.10.
P(S | A₁) = 0.01, P(S | A₂) = 0.02, P(S | A₃) = 0.05.

P(spam)?

**Solution.**

P(S) = 0.01·0.70 + 0.02·0.20 + 0.05·0.10 = 0.007 + 0.004 + 0.005 = **0.016 (1.6%)**.

---

### Q3.3 — Click-through tree (*Companion S3 Practice 4*)

Devices: desktop 0.5 (P(click)=0.04), tablet 0.2 (0.06), mobile 0.3 (0.10). Overall click rate?

**Solution.**

P(click) = 0.04·0.5 + 0.06·0.2 + 0.10·0.3 = 0.020 + 0.012 + 0.030 = **0.062 (6.2%)**.

---

### Q3.4 — Mining job on time (*Companion S3 worked example*)

P(rain) = 0.45. P(on time | rain) = 0.42; P(on time | no rain) = 0.90. P(on time)?

**Solution.**

P(on time) = 0.45·0.42 + 0.55·0.90 = 0.189 + 0.495 = **0.684**.

---

### Q3.5 — Drawing two queens (*Companion S3 Practice 2*)

(a) Without replacement. (b) With replacement. From a 52-card deck.

**Solution.**

(a) 4/52 · 3/51 = 12/2652 = **1/221 ≈ 0.0045**.
(b) 4/52 · 4/52 = 16/2704 = **1/169 ≈ 0.0059**.

With replacement is larger because the second draw still has 4 queens (no shrinking).

---

### Q3.6 — Right-handed sample (*Companion S3 Practice 1*)

300 adults, 272 right-handed. With replacement, pick 3.

(a) P(all three right-handed); (b) P(at least one right-handed).

**Solution.**

p = 272/300 ≈ 0.9067.

(a) p³ = 0.9067³ ≈ **0.7456**.
(b) 1 − P(none) = 1 − (1 − p)³ = 1 − (0.0933)³ ≈ 1 − 0.000813 ≈ **0.9992**.

---

### Q3.7 — Spam features independence (*Companion S3 Ex 4.3*)

P(A) = 0.6, P(B) = 0.4, independent.

(a) P(both); (b) P(either); (c) P(neither).

**Solution.**

(a) 0.6·0.4 = **0.24**.
(b) 0.6 + 0.4 − 0.24 = **0.76**.
(c) 1 − 0.76 = **0.24**.

---

### Q3.8 — Independence test from contingency (*Companion S3 Ex 4.2*)

100 flights surveyed; 20 late; 45 from airport A, of which 9 late. Are "from A" and "late" independent?

**Solution.**

P(late) = 20/100 = 0.20.
P(late | A) = 9/45 = 0.20.

Equal ⇒ **independent**. (If different, dependent.)

---

## SECTION 4 — BAYES' THEOREM

### Q4.1 — Vice-Chancellor research promotion (*Jan 2026 Q2*)

P(A) = 0.5, P(B) = 0.3, P(P) = 0.2.
P(R | A) = 0.8, P(R | B) = 0.6, P(R | P) = 0.4.

R is observed. Find P(A | R).

**Solution.**

P(R) = 0.8·0.5 + 0.6·0.3 + 0.4·0.2 = 0.40 + 0.18 + 0.08 = **0.66**.
P(A | R) = (0.8·0.5) / 0.66 = 0.40/0.66 = 20/33 ≈ **0.606 (60.6%)**.

---

### Q4.2 — Hardik / Rishabh / Surya (*2023 Regular Q4*)

Priors 0.2, 0.5, 0.3. Likelihoods 0.3, 0.6, 0.5.

If fee increases, find P(each captain).

**Solution.**

P(F) = 0.2·0.3 + 0.5·0.6 + 0.3·0.5 = 0.06 + 0.30 + 0.15 = **0.51**.

(a) P(H | F) = 0.06/0.51 = 2/17 ≈ **0.118**.
(b) P(R | F) = 0.30/0.51 = 10/17 ≈ **0.588**.
(c) P(S | F) = 0.15/0.51 = 5/17 ≈ **0.294**.

(Sum check: 0.118 + 0.588 + 0.294 = 1.0 ✓.)

---

### Q4.3 — Rare disease 1-in-1000 (*2024 Regular Q6a*)

P(D) = 0.001; P(+ | D) = 0.99; P(+ | Dᶜ) = 0.02.

P(D | +)?

**Solution.**

P(+) = 0.99·0.001 + 0.02·0.999 = 0.00099 + 0.01998 = **0.02097**.
P(D | +) = 0.00099/0.02097 ≈ **0.0472 (4.72%)**.

---

### Q4.4 — Rare genetic disorder (*2025 Regular Q4*)

P(D) = 0.001; sensitivity 0.98; specificity 0.97 (so P(+ | Dᶜ) = 0.03).

P(D | +)?

**Solution.**

P(+) = 0.98·0.001 + 0.03·0.999 = 0.00098 + 0.02997 = **0.03095**.
P(D | +) = 0.00098/0.03095 ≈ **0.0316 (3.16%)**.

---

### Q4.5 — DNA match (*2024 Makeup Q4*)

Region of 5,000,000 people. P(match | guilty) = 1; P(match | innocent) = 1/1,000,000. Prior P(G) = 1/5,000,000.

P(G | match)?

**Solution.**

Numerator: 1 · 1/5,000,000 = 1/5,000,000.
P(match | innocent) · P(innocent) = (1/1,000,000) · (4,999,999/5,000,000) ≈ 5/5,000,000.

P(G | match) ≈ 1/(1 + 5) = **1/6 ≈ 16.7%**.

(Lesson: a 1-in-a-million match in a 5-million-person region still leaves ~5 random matches; DNA alone isn't conclusive.)

---

### Q4.6 — Garage robbery (*2024 Makeup-2 Q2b*)

P(open) = 0.25. P(robbed | open) = 0.05; P(robbed | closed) = 0.01.

If robbed, P(open)?

**Solution.**

P(R) = 0.05·0.25 + 0.01·0.75 = 0.0125 + 0.0075 = **0.020**.
P(open | R) = 0.0125/0.020 = **0.625 (62.5%)**.

---

### Q4.7 — Bolt machines (*Practice Set 1.4*)

A 40%, B 35%, C 25%; defect rates 5%, 3%, 2%. Defective bolt — P(from A)?

**Solution.**

P(D) = 0.05·0.40 + 0.03·0.35 + 0.02·0.25 = 0.02 + 0.0105 + 0.005 = **0.0355**.
P(A | D) = 0.02/0.0355 ≈ **0.5634 (56.34%)**.

---

### Q4.8 — Editor → syntax error (*Practice Q2*)

P(A) = 0.30, P(B) = 0.50, P(C) = 0.20.
P(E | A) = 0.05; P(E | B) = 0.03; P(E | C) = 0.08.

(a) P(E); (b) P(C | E).

**Solution.**

(a) P(E) = 0.05·0.30 + 0.03·0.50 + 0.08·0.20 = 0.015 + 0.015 + 0.016 = **0.046**.
(b) P(C | E) = 0.016/0.046 ≈ **0.348 (34.8%)**.

---

### Q4.9 — Spam filter, one word (*Companion S4 Ex 2.2*)

P(spam) = 0.30; P(offer | spam) = 0.80; P(offer | not spam) = 0.10.

P(spam | offer)?

**Solution.**

P(offer) = 0.80·0.30 + 0.10·0.70 = 0.24 + 0.07 = **0.31**.
P(spam | offer) = 0.24/0.31 ≈ **0.774 (77.4%)**.

(Up from a 30% prior — strong evidence.)

---

### Q4.10 — Spam detector with high false positive (*Companion S4 Practice 2*)

P(spam) = 0.5; sensitivity 0.99; false-positive 0.05.

P(not spam | flagged)?

**Solution.**

P(flagged) = 0.99·0.5 + 0.05·0.5 = 0.495 + 0.025 = **0.520**.
P(not spam | flagged) = 0.025/0.520 ≈ **0.0481 (4.81%)**.

(With base rate 50%, accuracy translates well; 95% of flagged messages are real spam.)

---

## SECTION 5 — NAÏVE BAYES CLASSIFIER

### Q5.1 — "Assignment urgent" (*Dec 2025 Q2*)

Training set:
| Msg | Words | Label |
|-----|-------|-------|
| 1 | assignment due tomorrow please help | Urgent |
| 2 | lunch plans today | Not Urgent |
| 3 | exam rescheduled urgent attention | Urgent |
| 4 | thanks for the notes | Not Urgent |

Classify "assignment urgent" with α = 1.

**Solution.**

Vocabulary V (16 words): {assignment, due, tomorrow, please, help, lunch, plans, today, exam, rescheduled, urgent, attention, thanks, for, the, notes}. |V| = 16.

P(U) = P(N) = **0.5**.
N_U = 9; N_N = 7.

count(assignment, U) = 1; count(urgent, U) = 1; count(assignment, N) = 0; count(urgent, N) = 0.

Smoothed:
- P(assignment | U) = 2/(9+16) = 2/25.
- P(urgent | U) = 2/25.
- P(assignment | N) = 1/(7+16) = 1/23.
- P(urgent | N) = 1/23.

Score_U = 0.5·(2/25)·(2/25) = 0.5·4/625 = **0.0032**.
Score_N = 0.5·(1/23)·(1/23) = 0.5/529 ≈ **0.000945**.

→ **Classify as Urgent.**

---

### Q5.2 — "Win free money" (*2024 Makeup-2 Q2*)

Training:
| Msg | Label |
|-----|-------|
| Win money | Spam |
| Free prize | Spam |
| See you soon | Ham |
| Call me now | Ham |

Classify "Win free money".

**Solution.**

Vocab V = {win, money, free, prize, see, you, soon, call, me, now}. |V| = 10.

P(Spam) = P(Ham) = 0.5.
N_Spam = 4 (win, money, free, prize); N_Ham = 6 (see, you, soon, call, me, now).

Counts in Spam: win=1, money=1, free=1; in Ham: 0 each.

Smoothed:
- P(win | Spam) = 2/(4+10) = 2/14 = 1/7.
- P(free | Spam) = 1/7.
- P(money | Spam) = 1/7.
- P(win | Ham) = 1/(6+10) = 1/16.
- P(free | Ham) = 1/16.
- P(money | Ham) = 1/16.

Score_Spam = 0.5·(1/7)³ = 0.5/343 ≈ **0.001458**.
Score_Ham = 0.5·(1/16)³ = 0.5/4096 ≈ **0.000122**.

→ **Classify as Spam.**

---

### Q5.3 — "Congratulations! You have won money quickly" (*2024 Makeup Q3*)

Training (4 mails, 2 Spam, 2 Not Spam):
| Email | Label |
|-------|-------|
| Congratulations, you have won a free prize | Spam |
| Monthly meeting scheduled for Monday | Not Spam |
| Earn money quickly with this simple trick | Spam |
| Project report attached. Review by Friday | Not Spam |

Classify "Congratulations! You have won money quickly".

**Solution.**

Vocab |V| = 25 (after lowercasing/tokenising). N_Spam = 14, N_NotSpam = 14.

Counts in Spam:
- congratulations 1, you 1, have 1, won 1, money 1, quickly 1.

Counts in NotSpam: all zero for the test words.

Test tokens: congratulations, you, have, won, money, quickly (6 tokens).

Smoothed Spam likelihoods (denominator = 14 + 25 = 39):
- Each of those words: 2/39.
- Score_Spam = 0.5 · (2/39)⁶ = 0.5 · 6.27×10⁻⁸ ≈ **3.14×10⁻⁸**.

Smoothed NotSpam likelihoods (denominator = 14 + 25 = 39):
- Each test word: 1/39.
- Score_NotSpam = 0.5 · (1/39)⁶ = 0.5 · 1/(39⁶) ≈ **0.5×9.79×10⁻¹⁰ ≈ 4.9×10⁻¹⁰**.

Score_Spam / Score_NotSpam = 2⁶ = 64. → **Classify as Spam.**

(Numerical answers depend on exact vocab size; approach is what gets marks.)

---

### Q5.4 — "Win free call" (*Practice Set Q7*)

Training:
| Msg | Label |
|-----|-------|
| Win cash now | Spam |
| Call me later | Ham |
| Win free cash | Spam |
| Meeting at five | Ham |

Classify "Win free call".

**Solution.**

Vocab = {win, cash, now, call, me, later, free, meeting, at, five}; |V| = 10.

P(Spam) = P(Ham) = 0.5. N_Spam = 6 (win, cash, now, win, free, cash); N_Ham = 6.

Counts in Spam: win=2, cash=2, now=1, free=1.
Counts in Ham: call=1, me=1, later=1, meeting=1, at=1, five=1.

Smoothed (denom 6+10=16):
Spam: P(win)=3/16; P(free)=2/16; P(call)=1/16. Score_Spam = 0.5·(3·2·1)/16³ = 0.5·6/4096 ≈ **7.32×10⁻⁴**.
Ham: P(win)=1/16; P(free)=1/16; P(call)=2/16. Score_Ham = 0.5·(1·1·2)/16³ = 0.5·2/4096 ≈ **2.44×10⁻⁴**.

→ **Classify as Spam** (Score_Spam ≈ 3× Score_Ham).

---

### Q5.5 — Weather → Player Play (*2024 Regular Q5*)

14-row table; Snowy class — will player play?

**Solution.**

P(Play = Yes) = 7/14 = 0.5; P(Play = No) = 7/14 = 0.5.

Out of 14 rows, snowy = 4 rows (rows 6, 12, 13, 14). Among those snowy rows, "Yes" appears in 1 row (row 14: "Snowy Yes"). So:
- P(Snowy | Yes) = 1/7.
- P(Snowy | No) = 3/7 (3 snowy "No" rows).

By Bayes:
P(Yes | Snowy) = (1/7 · 0.5) / [(1/7)(0.5) + (3/7)(0.5)] = (1/14) / (4/14) = **0.25 (25%)**.

→ Less likely to play; **predict No** (since P(No|Snowy) = 0.75).

---

### Q5.6 — Promotion type (*2024 Makeup-2 Q4a*)

Given table (Email/SocialMedia/TVAd → Yes/No purchases). For Promotion = "Social Media", P(purchase = Yes) by Naïve Bayes.

**Solution (counts from typical 14-row variant):**

Total rows: 14 (8 Yes, 6 No assumed).
Among Yes rows, Social Media occurrences: 2.
Among No rows, Social Media occurrences: 3.

P(Yes | SM) ∝ P(SM | Yes) · P(Yes) = (2/8)·(8/14) = 16/112.
P(No | SM) ∝ P(SM | No) · P(No) = (3/6)·(6/14) = 18/84.

Normalising: P(Yes | SM) ≈ (2/14) / [(2/14) + (3/14)] = 2/5 = **0.40 (40%)**.

(Counts vary by exact dataset version; structure is the answer.)

---

### Q5.7 — Tennis dataset (*Companion S4 Ex 3.1*)

14 days; P(Yes) = 9/14, P(No) = 5/14. Test (Sunny, Hot, Normal, ¬Windy).

**Solution.**

Yes-counts (out of 9): Sunny=3; Hot=2; Normal=6; ¬Windy=6.
No-counts (out of 5): Sunny=3; Hot=2; Normal=1; ¬Windy=2.

Score_Yes = (9/14)·(3/9)·(2/9)·(6/9)·(6/9) = (9/14)·(216/6561) ≈ 0.0212.
Score_No = (5/14)·(3/5)·(2/5)·(1/5)·(2/5) = (5/14)·(12/625) ≈ 0.00686.

→ **Predict Yes.**

---

### Q5.8 — Identifying salaried vs business (*2023 Makeup Q3*)

8-row dataset (Vehicle/Card/Travel/Employment). Test: Vehicle=scooter, CreditCard=yes, Travel=Bus.

**Solution.** 4 Salaried, 3 Business, 1 Unknown.

P(Salaried) = 4/8 = 0.5; P(Business) = 3/8 ≈ 0.375. (Treat Unknown as not classified.)

For Salaried (4 rows): Scooter=1, Credit-card=1, Bus=1.
For Business (3 rows): Scooter=0, Credit-card=1, Bus=1.

With Laplace (|V| = 3 categorical features × distinct values, smoothing each category):

Score_Salaried = 0.5·(2/(4+3))·(2/7)·(2/7) ≈ 0.5·(8/343) ≈ **0.012**.
Score_Business = 0.375·(1/(3+3))·(2/6)·(2/6) ≈ 0.375·(4/216) ≈ **0.007**.

→ **Predict Salaried.**

---

## SECTION 6 — DISCRETE RANDOM VARIABLES

### Q6.1 — PMF table with k², k (*Dec 2025 Q3a*)

| x | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| p(x) | 0 | k | 2k | 2k | 3k | k² | 2k² | 7k² + k |

(i) Find k.
(ii) P(X < 6); P(X ≥ 6).
(iii) P(0 < X < 5).

**Solution.**

Sum = 0 + k + 2k + 2k + 3k + k² + 2k² + 7k² + k = 10k² + 9k = 1.

Solving 10k² + 9k − 1 = 0: k = (−9 ± √(81 + 40))/20 = (−9 ± 11)/20.
k = 1/10 (rejecting negative) → **k = 0.1**.

PMF: 0, 0.1, 0.2, 0.2, 0.3, 0.01, 0.02, 0.17.

(ii) P(X < 6) = 0 + 0.1 + 0.2 + 0.2 + 0.3 + 0.01 = **0.81**. P(X ≥ 6) = **0.19**.
(iii) P(0 < X < 5) = 0.1 + 0.2 + 0.2 + 0.3 = **0.8**.

---

### Q6.2 — Symmetric PMF k·x² (*2024 Makeup Q3b*)

P(X = x) = k·x² for x ∈ {−1, 0, 1}.

(a) k; (b) E[X]; (c) Var(X).

**Solution.**

Sum: k(1 + 0 + 1) = 2k = 1 ⇒ **k = 1/2**.

P(−1) = 0.5, P(0) = 0, P(1) = 0.5.
E[X] = (−1)(0.5) + 0 + 1(0.5) = **0**.
E[X²] = 1(0.5) + 0 + 1(0.5) = **1**.
Var(X) = 1 − 0² = **1**.

---

### Q6.3 — PMF with parameter c (*Practice Q4*)

| X | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| P(X) | 0.1 | 2c | 0.3 | c |

(a) c; (b) E[X], Var(X); (c) E[2X − 3].

**Solution.**

0.1 + 2c + 0.3 + c = 1 ⇒ 3c = 0.6 ⇒ **c = 0.2**.

PMF: 0.1, 0.4, 0.3, 0.2.

E[X] = 0(0.1) + 1(0.4) + 2(0.3) + 3(0.2) = 0 + 0.4 + 0.6 + 0.6 = **1.6**.
E[X²] = 0 + 0.4 + 1.2 + 1.8 = **3.4**.
Var(X) = 3.4 − 1.6² = 3.4 − 2.56 = **0.84**.

E[2X − 3] = 2(1.6) − 3 = **0.2**.

---

### Q6.4 — Var(2X + 3) and PMF (*2025 Regular Q5*)

|X | 0 | 1 | 2 | 3 | 4|
|--|---|---|---|---|---|
|P(X)| 1/10 | 2/10 | 4/10 | 2/10 | 1/10 |

Find: (a) verify PMF; (b) E[X]; (c) interpret; (d) Var(2X + 3).

**Solution.**

(a) Sum = 0.1 + 0.2 + 0.4 + 0.2 + 0.1 = 1 ✓.

(b) E[X] = 0 + 0.2 + 0.8 + 0.6 + 0.4 = **2**.

(c) On average, the model predicts 2 errors per document.

(d) E[X²] = 0 + 0.2 + 1.6 + 1.8 + 1.6 = 5.2. Var(X) = 5.2 − 4 = **1.2**.
Var(2X + 3) = 4·Var(X) = **4.8**.

> Or if PMF is the simpler {0:0.1, 1:0.2, 2:0.4, 3:0.2, 4:0.1} with Var(X)=1, then Var(2X+3) = 4. Confirm with the actual table.

---

### Q6.5 — PMF with k(x+1)/4 (*2023 Makeup Q4*)

p(x) = k(x + 1)/4 for x ∈ {−1, 0, 1, 2}.

(a) k; (b) E[X]; (c) E[X²]; (d) Var(X); (e) P(−1 < X < 2).

**Solution.**

p(−1)=0; p(0)=k/4; p(1)=k/2; p(2)=3k/4. Sum = 6k/4 = 1 ⇒ **k = 2/3**.

PMF: 0, 1/6, 1/3, 1/2.

E[X] = (−1)(0) + 0(1/6) + 1(1/3) + 2(1/2) = 0 + 0 + 1/3 + 1 = **4/3**.
E[X²] = 0 + 0 + 1(1/3) + 4(1/2) = 1/3 + 2 = **7/3**.
Var(X) = 7/3 − (4/3)² = 7/3 − 16/9 = 21/9 − 16/9 = **5/9**.
P(−1 < X < 2) = p(0) + p(1) = 1/6 + 1/3 = **1/2**.

---

### Q6.6 — Defectives drawn without replacement (*2024 Regular Q4b*)

10 items, 3 defective; sample 4 without replacement. X = defectives in sample.

PMF and P(X ≥ 1).

**Solution.**

Hypergeometric: P(X = k) = [C(3, k)·C(7, 4 − k)] / C(10, 4).
C(10, 4) = 210.

- P(X = 0) = C(3,0)·C(7,4)/210 = 1·35/210 = **1/6**.
- P(X = 1) = C(3,1)·C(7,3)/210 = 3·35/210 = **1/2**.
- P(X = 2) = C(3,2)·C(7,2)/210 = 3·21/210 = **3/10**.
- P(X = 3) = C(3,3)·C(7,1)/210 = 1·7/210 = **1/30**.

Check: 1/6 + 1/2 + 3/10 + 1/30 = 5/30 + 15/30 + 9/30 + 1/30 = 30/30 = 1 ✓.

P(X ≥ 1) = 1 − 1/6 = **5/6**.

---

### Q6.7 — Linear transform (*Practice 2.8*)

E[X] = 10, Var(X) = 4. Y = 3X − 5.

E[Y], Var(Y)?

**Solution.**

E[Y] = 3(10) − 5 = **25**.
Var(Y) = 3²·Var(X) = 9·4 = **36**.

---

### Q6.8 — Driving test attempts (*Companion S5 Practice 1*)

P(X = i) ∝ rule. Suppose proportional to p^(i−1)·q. Confirm sum to 1; find E[X].

**Solution (specific case).** Suppose passing probability per attempt p = 0.6 (so q = 0.4), attempts 1, 2, 3:
P(1) = 0.6, P(2) = 0.4·0.6 = 0.24, P(3) = 0.4²·1 = 0.16 (since attempt 3 is the last). Sum = 1.

E[X] = 1·0.6 + 2·0.24 + 3·0.16 = 0.6 + 0.48 + 0.48 = **1.56 attempts**.

(Actual numerics depend on the specific exam version; method is the same.)

---

### Q6.9 — Coin-toss winnings (*2025 Regular Q6*)

3 fair coin tosses. X = number of tails. Y = winnings ($3 if first tail at toss 1, $2 at toss 2, $1 at toss 3, −$2 if no tails).

(a) Joint PMF; (b) Marginal of X; (c) E(X).

**Solution.**

Outcomes (each prob 1/8):
- HHH: X=0, Y=−2.
- HHT: X=1, Y=1 (first tail at toss 3).
- HTH: X=1, Y=2.
- THH: X=1, Y=3.
- HTT: X=2, Y=2.
- THT: X=2, Y=3.
- TTH: X=2, Y=3.
- TTT: X=3, Y=3.

| X\Y | −2 | 1 | 2 | 3 |
|-----|----|---|----|---|
| 0 | 1/8 | 0 | 0 | 0 |
| 1 | 0 | 1/8 | 1/8 | 1/8 |
| 2 | 0 | 0 | 1/8 | 2/8 |
| 3 | 0 | 0 | 0 | 1/8 |

Marginal X: P(0) = 1/8, P(1) = 3/8, P(2) = 3/8, P(3) = 1/8 (Binomial(3, 0.5)).

E[X] = 0·1/8 + 1·3/8 + 2·3/8 + 3·1/8 = (0 + 3 + 6 + 3)/8 = **12/8 = 1.5**.

---

### Q6.10 — Fair-game dice winnings (*Companion S5 Ex 3.1*)

Two dice rolled; +$3 if sum 7, −$2 if 8, −$4 if 3, else $0. Is the game fair?

**Solution.**

P(7) = 6/36, P(8) = 5/36, P(3) = 2/36.

E[X] = 3·(6/36) + (−2)·(5/36) + (−4)·(2/36)
    = (18 − 10 − 8)/36 = 0/36 = **0**.

→ **Game is fair**.

---

## SECTION 7 — CONTINUOUS RANDOM VARIABLES

### Q7.1 — Triangular PDF (*2024 Makeup Q6*)

f(x) = c·x² for 0 ≤ x ≤ 1, f(x) = c·(2 − x) for 1 < x ≤ 2.

(a) c; (b) P(0.5 < X < 1.5); (c) E[X].

**Solution.**

(a) ∫₀¹ x² dx + ∫₁² (2 − x) dx = 1/3 + 0.5 = 5/6. c = **6/5 = 1.2**.

(b) ∫_{0.5}^{1} 1.2 x² dx + ∫_{1}^{1.5} 1.2(2 − x) dx = 1.2·(1/3 − 1/24) + 1.2·(0.375) = 1.2·(7/24) + 0.45 = 0.35 + 0.45 = **0.8**.

(c) E[X] = ∫₀¹ x·1.2x² dx + ∫₁² x·1.2(2 − x) dx
       = 1.2·(1/4) + 1.2·(2/3) = 0.3 + 0.8 = **1.1**.

---

### Q7.2 — Tent on (0, 2) (*Companion S5 Ex 4.1*)

f(x) = x for 0 < x < 1, f(x) = 2 − x for 1 < x < 2.

P(0.6 < X < 1.2)? E[X]?

**Solution.**

Total area = (1/2)·(2)·(1) = 1 ✓.

P(0.6 < X < 1.2) = ∫_{0.6}^{1} x dx + ∫_{1}^{1.2} (2 − x) dx
   = [x²/2]_{0.6}^{1} + [2x − x²/2]_{1}^{1.2}
   = (0.5 − 0.18) + (2.4 − 0.72 − (2 − 0.5))
   = 0.32 + (1.68 − 1.5)
   = 0.32 + 0.18 = **0.5**.

E[X] = 1 (by symmetry around x = 1).

---

### Q7.3 — Uniform-like on [-3, 3] (*2024 Makeup-2 Q4b*)

f(x) = k/6 on [−3, 3].

(a) k; (b) E[X]; (c) Var(X).

**Solution.**

(a) (k/6)·6 = k = 1 ⇒ **k = 1**.
(b) E[X] = 0 (symmetric).
(c) E[X²] = (1/6)·∫_{−3}^{3} x² dx = (1/6)·(2·27/3) = (1/6)·18 = **3**. Var(X) = **3**.

---

### Q7.4 — Linear PDF k(x+1)/4 (*2023 Regular Q5*)

f(x) = k(x + 1)/4 on [0, 2].

(a) k; (b) E[X]; (c) E[X²]; (d) Var(X); (e) P(0.5 < X < 1.5).

**Solution.**

∫₀² k(x + 1)/4 dx = (k/4)·[x²/2 + x]₀² = (k/4)·(2 + 2) = k = 1 ⇒ **k = 1**.

f(x) = (x + 1)/4, 0 ≤ x ≤ 2.

E[X] = ∫₀² x(x + 1)/4 dx = (1/4)·[x³/3 + x²/2]₀² = (1/4)·(8/3 + 2) = (1/4)·(14/3) = **7/6 ≈ 1.167**.

E[X²] = ∫₀² x²(x + 1)/4 dx = (1/4)·[x⁴/4 + x³/3]₀² = (1/4)·(4 + 8/3) = (1/4)·(20/3) = **5/3 ≈ 1.667**.

Var(X) = 5/3 − (7/6)² = 5/3 − 49/36 = 60/36 − 49/36 = **11/36 ≈ 0.306**.

P(0.5 < X < 1.5) = ∫_{0.5}^{1.5} (x + 1)/4 dx = (1/4)·[x²/2 + x]_{0.5}^{1.5}
   = (1/4)·[(1.125 + 1.5) − (0.125 + 0.5)] = (1/4)·(2.625 − 0.625) = (1/4)·2 = **0.5**.

---

### Q7.5 — CDF → PDF (*Practice Set 2.7*)

F(x) = 0 for x < 0; x³/8 for 0 ≤ x ≤ 2; 1 for x > 2.

(a) f(x); (b) P(X > 1); (c) median.

**Solution.**

(a) f(x) = F'(x) = **3x²/8 for 0 ≤ x ≤ 2**, else 0.

(b) P(X > 1) = 1 − F(1) = 1 − 1/8 = **7/8**.

(c) Median: F(m) = 0.5 ⇒ m³/8 = 0.5 ⇒ m³ = 4 ⇒ m = ∛4 ≈ **1.587**.

---

### Q7.6 — Overheating index CDF (*Jan 2026 Q5*)

CDF F(x) for X on [0, 3]; piece given f(x) = k·g(x). (Specific formula varies; use general method.)

**Method:** (a) F(3) = 1 → solve for k. (b) P(1 ≤ X ≤ 2) = F(2) − F(1). (c) f(x) = F'(x).

> **Worked variant:** F(x) = k·x³ on [0, 3].
> F(3) = 27k = 1 ⇒ k = 1/27. f(x) = 3x²/27 = x²/9.
> P(1 ≤ X ≤ 2) = F(2) − F(1) = 8/27 − 1/27 = 7/27 ≈ 0.259.

---

### Q7.7 — Density 1/4·x on [0, √8] (*Practice Set 2.5*)

f(x) = x/4 for 0 ≤ x ≤ √8, else 0.

(a) Validate; (b) P(X < 1); (c) E[X].

**Solution.**

(a) ∫₀^{√8} x/4 dx = (1/4)·[x²/2]₀^{√8} = (1/4)·(8/2) = 1 ✓.

(b) ∫₀¹ x/4 dx = (1/4)·(1/2) = **1/8 = 0.125**.

(c) E[X] = ∫₀^{√8} x·(x/4) dx = (1/4)·[x³/3]₀^{√8} = (1/4)·(8√8/3) = (16√2)/12 = (4√2)/3 ≈ **1.886**.

---

### Q7.8 — Piecewise PDF k·x and k(4−x) (*Practice Q3*)

f(x) = k·x for 0 ≤ x ≤ 2, f(x) = k(4 − x) for 2 < x ≤ 4.

(a) k; (b) P(1 < X < 3); (c) E[X].

**Solution.**

∫₀² k·x dx + ∫₂⁴ k(4 − x) dx = k·2 + k·2 = 4k = 1 ⇒ **k = 0.25**.

P(1 < X < 3) = ∫₁² 0.25·x dx + ∫₂³ 0.25(4 − x) dx = 0.25·1.5 + 0.25·1.5 = **0.75**.

By symmetry around x = 2 → **E[X] = 2**. (Verify: ∫₀² x·(0.25x) dx + ∫₂⁴ x·0.25(4 − x) dx = (0.25/3)·8 + 0.25·∫₂⁴(4x − x²) dx = 2/3 + 0.25·[2x² − x³/3]₂⁴ = 2/3 + 0.25·(32 − 64/3 − 8 + 8/3) = 2/3 + 0.25·(24 − 56/3) = 2/3 + 0.25·(16/3) = 2/3 + 4/3 = 2 ✓.)

---

### Q7.9 — Joint conditional integration (*Practice Set 2.9*)

f(x, y) = (x + y)/3 on 0 < x < 2, 0 < y < 1.

Find f(x | y) and P(X > 1 | Y = 0.5).

**Solution.**

Marginal h(y) = ∫₀² (x + y)/3 dx = (1/3)·(2 + 2y) = (2 + 2y)/3 = (2/3)(1 + y).

f(x | y) = f(x, y)/h(y) = [(x + y)/3] / [(2/3)(1 + y)] = (x + y) / [2(1 + y)].

For y = 0.5: f(x | 0.5) = (x + 0.5) / 3.

P(X > 1 | Y = 0.5) = ∫₁² (x + 0.5)/3 dx = (1/3)·[x²/2 + 0.5x]₁² = (1/3)·[(2 + 1) − (0.5 + 0.5)] = (1/3)·2 = **2/3**.

---

### Q7.10 — Validate exponential-shape PDF (Generic)

f(x) = c·e^(−x) for x ≥ 0.

(a) c; (b) E[X]; (c) Var(X).

**Solution.**

(a) ∫₀^∞ c·e^(−x) dx = c·1 = 1 ⇒ **c = 1** (this is Exponential(1)).
(b) E[X] = ∫₀^∞ x·e^(−x) dx = **1** (by integration by parts).
(c) E[X²] = ∫₀^∞ x²·e^(−x) dx = 2. Var(X) = 2 − 1 = **1**.

---

## SECTION 8 — JOINT DISTRIBUTIONS

### Q8.1 — Joint PDF 6x²y (*Dec 2025 Q4*)

f(x, y) = 6x²y on [0,1]×[0,1], else 0.

(i) Verify; (ii) marginals; (iii) P(X < 0.5, Y > 0.5); (iv) independence; (v) E[X], E[Y], E[XY].

**Solution.**

(i) ∫₀¹ ∫₀¹ 6x²y dx dy = 6·(1/3)·(1/2) = 1 ✓.

(ii) fₓ(x) = 6x²·∫₀¹ y dy = 6x²·(1/2) = **3x²**.
   f_Y(y) = 2y (similarly).

(iii) P(X < 0.5, Y > 0.5) = ∫₀^{0.5} ∫_{0.5}^{1} 6x²y dy dx
   = ∫₀^{0.5} 6x²·(0.5 − 0.125) dx = ∫₀^{0.5} 6x²·0.375 dx = 2.25·[x³/3]₀^{0.5} = 2.25·0.04167 = **0.09375**.

(iv) f(x, y) = 6x²y; fₓ(x)·f_Y(y) = 3x²·2y = 6x²y ✓ → **independent**.

(v) E[X] = ∫₀¹ x·3x² dx = 3·(1/4) = **3/4**.
   E[Y] = ∫₀¹ y·2y dy = 2·(1/3) = **2/3**.
   E[XY] = E[X]·E[Y] = **1/2** (independence).

---

### Q8.2 — Joint PMF cybersecurity (*Jan 2026 Q4*)

f(x, y) = c·(x² + y + 1) for x ∈ {0, 1, 2, 3}, y ∈ {−1, 0, 1, 2}.

(a) c; (b) P(X = 1, Y ≥ 0); (c) Independence; (d) P(Y ≥ 0 | X ≥ 1).

**Solution.**

(a) Σ Σ (x² + y + 1):
For x = 0: y ∈ {−1, 0, 1, 2} → values 0, 1, 2, 3 → sum = 6.
x = 1: 0 + 1 + 1 = (1) at y=−1: 1+(−1)+1 = 1; y=0: 2; y=1: 3; y=2: 4. Sum = 10.
x = 2: y=−1: 4; y=0: 5; y=1: 6; y=2: 7. Sum = 22.
x = 3: y=−1: 9; y=0: 10; y=1: 11; y=2: 12. Sum = 42.

Grand total = 6 + 10 + 22 + 42 = 80. So c = **1/80**.

(b) P(X = 1, Y ≥ 0) = sum of (x² + y + 1)/80 for (1, 0), (1, 1), (1, 2)
   = (2 + 3 + 4)/80 = 9/80 = **0.1125**.

(c) P(X = 1) = 10/80 = 0.125. P(Y = 0) = (1 + 2 + 5 + 10)/80 = 18/80 = 0.225.
   P(X = 1)·P(Y = 0) = 0.0281; but actual joint = 2/80 = 0.025. Different ⇒ **dependent**.

(d) P(X ≥ 1) = (10 + 22 + 42)/80 = 74/80 = 0.925.
   P(X ≥ 1, Y ≥ 0) = (2+3+4) + (5+6+7) + (10+11+12) = 9 + 18 + 33 = 60. So 60/80 = 0.75.
   P(Y ≥ 0 | X ≥ 1) = 0.75/0.925 ≈ **0.811**.

---

### Q8.3 — Independent joint via product of margins (*2023 Regular Q6*)

X PMF: −1: 0.25, 0: 0.15, 1: 0.35, 2: 0.25.
Y PMF: 0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40.

(a) Joint distribution; (b) P(X < 1, Y < 2); (c) P(X < 1 | Y < 2).

**Solution.**

X and Y independent → joint = product of marginals.

| X\Y | 0 | 1 | 2 | 3 |
|-----|---|---|---|---|
| −1 | 0.025 | 0.05 | 0.075 | 0.10 |
| 0 | 0.015 | 0.03 | 0.045 | 0.06 |
| 1 | 0.035 | 0.07 | 0.105 | 0.14 |
| 2 | 0.025 | 0.05 | 0.075 | 0.10 |

(b) P(X < 1, Y < 2) = P(X ∈ {−1, 0}, Y ∈ {0, 1}) = (0.25 + 0.15)·(0.10 + 0.20) = 0.40·0.30 = **0.12**.

(c) P(Y < 2) = 0.30. P(X < 1 | Y < 2) = (0.40·0.30)/0.30 = **0.40**.

(Independence makes conditional = marginal.)

---

### Q8.4 — Joint PDF k(x + y)/8 on (0,2)×(0,2) (*2023 Makeup Q5*)

(a) k; (b) fₓ(x); (c) f_Y(y); (d) independence?

**Solution.**

∫₀² ∫₀² k(x + y)/8 dx dy = 1.
Inner: ∫₀² (x + y)/8 dx = [x²/16 + xy/8]₀² = 4/16 + 2y/8 = 0.25 + 0.25y.
Outer: ∫₀² k(0.25 + 0.25y) dy = k(0.5 + 0.5) = k = 1 ⇒ **k = 1**.

fₓ(x) = ∫₀² (x + y)/8 dy = (x·2 + 2)/8 = (x + 1)/4 on [0, 2].
f_Y(y) = (y + 1)/4 (symmetric).

f(x, y) = (x + y)/8; fₓ(x)·f_Y(y) = (x + 1)(y + 1)/16 ≠ (x + y)/8 in general → **not independent**.

---

### Q8.5 — Joint PMF table (*2024 Makeup-2 Q5*)

| X\Y | 0 | 1 | 2 | 3 |
|-----|---|---|---|---|
| 0 | 0.11 | 0.09 | 0.01 | 0.07 |
| 1 | 0.09 | 0.11 | 0.06 | 0.01 |
| 2 | 0.01 | 0.06 | 0.11 | 0.04 |
| 3 | 0.07 | 0.01 | 0.04 | 0.11 |

(i) Marginals; (ii) P(X = 1 | Y = 2); (iii) P(Y = 1 | X = 1); (iv) Mean and variance of marginals.

**Solution.**

Marginal X: row sums.
- P(X=0) = 0.11+0.09+0.01+0.07 = 0.28.
- P(X=1) = 0.09+0.11+0.06+0.01 = 0.27.
- P(X=2) = 0.01+0.06+0.11+0.04 = 0.22.
- P(X=3) = 0.07+0.01+0.04+0.11 = 0.23.

Marginal Y: column sums.
- P(Y=0) = 0.11+0.09+0.01+0.07 = 0.28.
- P(Y=1) = 0.09+0.11+0.06+0.01 = 0.27.
- P(Y=2) = 0.01+0.06+0.11+0.04 = 0.22.
- P(Y=3) = 0.07+0.01+0.04+0.11 = 0.23.

(ii) P(X = 1 | Y = 2) = 0.06/0.22 ≈ **0.273**.
(iii) P(Y = 1 | X = 1) = 0.11/0.27 ≈ **0.407**.

(iv) E[X] = 0(0.28) + 1(0.27) + 2(0.22) + 3(0.23) = 0 + 0.27 + 0.44 + 0.69 = **1.40**.
E[X²] = 0 + 0.27 + 0.88 + 2.07 = **3.22**.
Var(X) = 3.22 − 1.96 = **1.26**.

(Y has identical numbers by symmetry.)

---

### Q8.6 — Practice joint PMF (*Practice Set 2.4*)

| f(x, y) | y=0 | y=1 | y=2 |
|---------|-----|-----|-----|
| x=0 | 0.1 | 0.2 | 0.1 |
| x=1 | 0.3 | 0.1 | 0.2 |

(a) Marginals; (b) E[X], E[Y]; (c) Independence?

**Solution.**

g(0) = 0.4, g(1) = 0.6.
h(0) = 0.4, h(1) = 0.3, h(2) = 0.3.

E[X] = 0(0.4) + 1(0.6) = 0.6.
E[Y] = 0(0.4) + 1(0.3) + 2(0.3) = 0.9.

f(0, 0) = 0.1; g(0)·h(0) = 0.16. Different → **not independent**.

---

### Q8.7 — Joint PDF symmetric (Generic)

f(x, y) = (4xy) on (0,1)×(0,1).

Verify, find marginals, independence.

**Solution.**

∫₀¹ ∫₀¹ 4xy dx dy = 4·(1/2)·(1/2) = 1 ✓.

fₓ(x) = ∫₀¹ 4xy dy = 4x·(1/2) = 2x.
f_Y(y) = 2y.

f(x, y) = 4xy = 2x·2y = fₓ(x)·f_Y(y) → **independent**. E[XY] = E[X]·E[Y] = (2/3)·(2/3) = 4/9.

---

### Q8.8 — Three-variable concept check (Generic)

If P(A ∩ B ∩ C) = P(A)·P(B)·P(C), are A, B, C mutually independent?

**Answer.** Not necessarily. Mutual independence requires the product factorisation for **every** subset (pairs and triple). The triple-only condition is "pairwise independence" in disguise plus extra; the standard counterexample is two coin flips and the parity of their XOR.

For exam purposes: state both pairwise and triple conditions.

---

## SECTION 9 — BINOMIAL & POISSON

### Q9.1 — Restaurant water (*Companion S6 Ex 1*)

n = 10, p = 0.6.

(i) P(X = 6); (ii) P(X < 9); (iii) P(X = 0); (iv) P(X ≤ 2); (v) P(X ≥ 3).

**Solution.**

(i) C(10, 6)·(0.6)⁶·(0.4)⁴ = 210·0.046656·0.0256 ≈ **0.251**.
(ii) 1 − P(9) − P(10) = 1 − 0.0403 − 0.0060 = **0.954**.
(iii) (0.4)¹⁰ ≈ **0.0001**.
(iv) 0.0001 + 0.0016 + 0.0106 = **0.0123**.
(v) 1 − 0.0123 = **0.9877**.

---

### Q9.2 — Coin tosses, ≤ 3 heads in 10 (*Companion S6 Ex 2*)

n = 10, p = 0.5.

P(X ≤ 3)?

**Solution.**

P(X ≤ 3) = (C(10,0) + C(10,1) + C(10,2) + C(10,3))/2¹⁰ = (1 + 10 + 45 + 120)/1024 = **176/1024 ≈ 0.172**.

---

### Q9.3 — Dice product = 6 in 5 trials (*Companion S6 Ex 3*)

p = 4/36 = 1/9; q = 8/9; n = 5.

P(X ≥ 4)?

**Solution.**

P(X = 4) = C(5,4)·(1/9)⁴·(8/9) = 5·(1/6561)·(8/9) = 40/59049.
P(X = 5) = (1/9)⁵ = 1/59049.

P(X ≥ 4) = 41/59049 ≈ **0.000694**.

---

### Q9.4 — Guessing MCQ (*Companion S6 Ex 4*)

5 questions, 4 choices each, p = 1/4.

(a) Exactly 3 correct; (b) at most 3; (c) at least 1; (d) mean and variance.

**Solution.**

(a) C(5,3)·(1/4)³·(3/4)² = 10·(1/64)·(9/16) = 90/1024 = 45/512 ≈ **0.0879**.
(b) 1 − P(4) − P(5) = 1 − [C(5,4)(1/4)⁴(3/4) + (1/4)⁵] = 1 − (15+1)/1024 = 1008/1024 = **63/64 ≈ 0.984**.
(c) 1 − (3/4)⁵ = 1 − 243/1024 = **781/1024 ≈ 0.763**.
(d) np = 1.25; npq = 0.9375.

---

### Q9.5 — Reverse-engineering Bin (*Companion S6 P10*)

Mean 6, variance 4. Find n, p.

**Solution.** q = var/mean = 4/6 = 2/3. p = 1/3. n = mean/p = 18 → **Bin(18, 1/3)**.

---

### Q9.6 — Reverse-engineering Bin (*2023 Makeup Q2*)

μ = 4, σ = 2/√3 → σ² = 4/3 = npq.

q = (4/3)/4 = 1/3, p = 2/3, n = 4·(3/2) = 6 → **Bin(6, 2/3)**.

P(6 < X < 9) = 0 (X ≤ 6).

---

### Q9.7 — Bias coin Poisson mismatch (*2025 Regular Q2b*)

n = 40, p = 0.15, k = 5. Compute Bin and Poisson approximations.

**Solution.**

Bin: P(5) = C(40,5)·(0.15)⁵·(0.85)³⁵ ≈ 658008·7.59×10⁻⁵·0.00339 ≈ **0.1692**.

Poisson with λ = 6: P(5) = e⁻⁶·6⁵/120 = e⁻⁶·64.8 ≈ 0.00248·64.8 ≈ **0.1606**.

Difference ≈ 0.0086 — Poisson approximation is poor because p = 0.15 is too large; Poisson assumption breaks down.

---

### Q9.8 — Surgical blades (*Dec 2025 Q3b*)

0.2% defects, 10 per packet, 10000 packets. λ = 0.02 per packet.

P(0 defective)? P(1 defective)? Expected packets in each case?

**Solution.**

P(X = 0) = e^(−0.02) ≈ 0.9802 → ~9802 packets.
P(X = 1) = 0.02·e^(−0.02) ≈ 0.0196 → ~196 packets.

---

### Q9.9 — Cheque errors (*Jan 2026 Q6a*)

Rate 1.2 per 100 cheques; 250 cheques. Distribution? P(X ≤ 2)? P(X > 4)?

**Solution.**

λ = 250·0.012 = **3** → **X ~ Poisson(3)** (or Bin(250, 0.012) approximated by Poisson).

P(X ≤ 2) = e⁻³·(1 + 3 + 4.5) = 8.5·e⁻³ ≈ **0.4232**.
P(X > 4) = 1 − P(X ≤ 4) = 1 − e⁻³·(1 + 3 + 4.5 + 4.5 + 3.375) = 1 − 16.375·e⁻³ ≈ 1 − 0.8153 = **0.1847**.

---

### Q9.10 — Flaws per foot (*2024 Regular Q3a*)

Poisson, 0.2 per foot. (i) P(≥ 2 in 1 ft); (ii) P(5 ≤ X ≤ 8 in 50 ft).

**Solution.**

(i) λ = 0.2. P(X ≥ 2) = 1 − P(0) − P(1) = 1 − e^(−0.2)·(1 + 0.2) = 1 − 0.9824 = **0.0176**.
(ii) λ = 0.2·50 = 10. P(5 ≤ X ≤ 8) = e⁻¹⁰·(10⁵/120 + 10⁶/720 + 10⁷/5040 + 10⁸/40320)
   ≈ e⁻¹⁰·(833.3 + 1388.9 + 1984.1 + 2480.2) ≈ e⁻¹⁰·6686.5 ≈ 0.0000454·6686.5 ≈ **0.3037**.

---

## SECTION 10 — NORMAL DISTRIBUTION & APPROXIMATIONS

### Q10.1 — Washer diameters (*Dec 2025 Q5*)

X ~ N(0.500, 0.005²). Spec: 0.492–0.506. P(defective)?

**Solution.**

z₁ = −1.6, z₂ = 1.2.
P(in spec) = F(1.2) − F(−1.6) = 0.8849 − 0.0548 = 0.8301.
P(defective) = **0.1699 ≈ 17%**.

---

### Q10.2 — Exam scores claims (*Jan 2026 Q3*)

X ~ N(73, 8²).

(i) P(X < 91)? (ii) P(65 < X < 89)? (iii) Cut-off for top 5%?

**Solution.**

(i) z = 18/8 = 2.25 ⇒ F(2.25) = **0.9878**. → "Most" is justified.
(ii) z = −1, 2 ⇒ 0.9773 − 0.1587 = **0.8186**.
(iii) z₀.₉₅ = 1.645 ⇒ 73 + 1.645·8 = **86.16**.

---

### Q10.3 — Final exam grades (*2024 Makeup-2 Q2a*)

N(73, 64).

(i) P(X < 91); (ii) P(65 < X < 89).

**Solution.** Same numerics as Q10.2: **0.9878** and **0.8186**.

---

### Q10.4 — Resistor resistance (*2024 Regular Q6b*)

N(40, 4) (μ = 40, σ = 2).

(i) Percentage > 43; (ii) > 43 to nearest ohm (use 43.5).

**Solution.**

(i) z = 1.5; 1 − F(1.5) = 1 − 0.9332 = **6.68%**.
(ii) Nearest-ohm rule means resistor < 42.5 is "42 ohms", >43.5 is "44 ohms". So compute z = (43.5 − 40)/2 = 1.75; 1 − F(1.75) = 1 − 0.9599 = **4.01%**.

---

### Q10.5 — Salaries (*2023 Makeup Q6*)

500 employees, salary ~ N(20, 4) (lakhs; σ = 2).

(a) Above 22; (b) below 16; (c) between 16 and 22.

**Solution.**

(a) z = 1, 1 − F(1) = 1 − 0.8413 = 0.1587 → 79 employees.
(b) z = −2, F(−2) = 0.0228 → 11 employees.
(c) F(1) − F(−2) = 0.8413 − 0.0228 = 0.8185 → 409 employees.

---

### Q10.6 — Instrument life (*2025 Regular Q2a*)

N(12, 4) (μ = 12, σ = 2).

(i) P(X < 7); (ii) P(7 < X < 12).

**Solution.**

(i) z = −2.5 ⇒ F(−2.5) = **0.0062**.
(ii) F(0) − F(−2.5) = 0.5 − 0.0062 = **0.4938**.

---

### Q10.7 — Heights (*Practice Set 2.3*)

X ~ N(175, 49) (σ = 7).

(a) P(X > 185); (b) P(170 < X < 180); (c) 90th percentile.

**Solution.**

(a) z = 10/7 ≈ 1.43; 1 − F(1.43) = 1 − 0.9236 = **0.0764**.
(b) z₁ = −0.71, z₂ = 0.71; F(0.71) − F(−0.71) = 0.7611 − 0.2389 = **0.5222**.
(c) z₀.₉ ≈ 1.28; X = 175 + 1.28·7 = **183.96 cm**.

---

### Q10.8 — Bin(150, 0.2) Normal approx (*2023 Regular Q7*)

P(50 < X < 80)?

**Solution.**

μ = 30, σ = √24 ≈ 4.899.

Continuity: P(50.5 < X < 79.5).
z₁ = 4.19, z₂ = 10.10. F(10.10) ≈ 1, F(4.19) ≈ 1 ⇒ probability ≈ **0**.

---

### Q10.9 — College plans (*Companion S6 Ex 15*)

n = 50, p = 0.31. P(X < 14)?

**Solution.**

np = 15.5 ✓, nq = 34.5 ✓.
μ = 15.5, σ = √(50·0.31·0.69) ≈ 3.27.

Continuity: P(X < 13.5). z = (13.5 − 15.5)/3.27 = −0.61 ⇒ F(−0.61) = **0.2709**.

---

### Q10.10 — Cloud seeding (*Companion S6 Ex 14*)

n = 40, p = 0.62. P(X ≤ 20)?

**Solution.**

μ = 24.8, σ = √(9.424) ≈ 3.07.
Continuity: P(X ≤ 20.5). z = (20.5 − 24.8)/3.07 = −1.40 ⇒ F(−1.40) = **0.0808**.

---

## SECTION 11 — SAMPLING, CLT, SAMPLE SIZE

### Q11.1 — Workers in supermarket (*2023 Regular Q8*)

μ = 40, σ = 5, n = 50. P(35 < x̄ < 45)?

**Solution.**

SE = 5/√50 ≈ 0.7071.
z₁ = (35 − 40)/0.7071 ≈ −7.07; z₂ ≈ 7.07.

F(7.07) ≈ 1, F(−7.07) ≈ 0 → **P ≈ 1**.

(Sample mean of 50 is far more concentrated than individual values.)

---

### Q11.2 — Adult heights, n = 49 (*2024 Makeup Q1b*)

μ = 70, σ = 3. P(69 < x̄ < 71)?

**Solution.**

SE = 3/7 ≈ 0.4286.
z = ±2.33 ⇒ F(2.33) − F(−2.33) = 0.9901 − 0.0099 = **0.9802**.

---

### Q11.3 — Bounced cheques (*Companion S7 Practice 4*)

n = 23, x̄ = 23.0, s = 4.60. 95% CI for μ.

**Solution.**

t₂₂, 0.025 = 2.074. SE = 4.60/√23 ≈ 0.959.
ME = 2.074·0.959 ≈ 1.99.

CI = 23 ± 1.99 → **[$21.01, $24.99]**.

---

### Q11.4 — Sample size for canteen (*2024 Makeup Q3a*)

σ = 14, ME = 2.5, 99% (z = 2.576). n?

**Solution.**

n = (2.576·14/2.5)² = (14.4256)² = 208.1 → **n = 209**.

---

### Q11.5 — Sample size for medical expenses (*Companion S7 Practice 5*)

σ = 400, ME = $50, 95% (z = 1.96). n? With ME = $25, n?

**Solution.**

(a) n = (1.96·400/50)² = (15.68)² ≈ 245.9 → **n = 246**.
(b) n = (31.36)² = 983.4 → **n = 984** (4× the first → halving error quadruples sample size).

---

### Q11.6 — CLT and sampling distribution shape (*2023 Makeup Q7*)

(a) Population Bin(n, p), sample n=45, x̄ = 10, s = 2. Does sampling dist of mean follow Bin?
(b) Same with n = 25.
(c) Population Normal, n = 9.

**Solution.**

(a) Sample mean would suggest p = 10/45 = 2/9; predicted SD = √(45·2/9·7/9) ≈ 2.79 ≠ 2 actual SD → **not consistent with Bin**.
(b) p = 10/25 = 0.4; predicted SD = √(25·0.4·0.6) = √6 ≈ 2.45 ≠ 2 → **not consistent**.
(c) By CLT, normal population implies sampling distribution of mean is **normal for any n** → x̄ ~ N(μ, σ²/9).

---

## Final Tips for Using This Bank

1. **First pass:** read each question, attempt mentally, then check.
2. **Second pass:** time yourself — 8 minutes per 5-mark question. If you're slow on a topic, drill that section.
3. **Third pass:** write each problem's solution from memory on paper. The exam is closed-book, so handwriting speed matters.
4. **Common substitutions:** the same problem reappears with different numbers (e.g., the rare-disease Bayes drill in 2024, 2025, Practice). Master the **structure**, the numbers will follow.
5. **Decimals vs fractions:** the answer keys accept both. Round only at the final step to 3–4 decimals.

**You now have every recurring pattern from 2023, 2024, 2025, Dec 2025, and Jan 2026 papers solved in detail.**

**Good luck!**




