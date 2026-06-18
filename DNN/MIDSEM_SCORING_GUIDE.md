# DNN MidSem — Final 10-Hour Study Guide

> **Course**: AIMLCZG511 Deep Neural Networks
> **Exam**: 21-06-2026 (AN), Closed Book, 2 hours, 30% weightage
> **Format** (per faculty note): **6 questions × 5 marks = 30 marks**
> **Syllabus**: Contact Sessions 1–8 (up to **CNN Part 1**)
> **Style**: Conceptual + interpretation of architectures/code + numerical (parameter count, GD forward/backward)
>
> This is your **single source of truth**. It is built from the official course handout, all 7 companion notes (Sessions 1–7), and **every** past paper (2022–2026 MidSem + most recent EndSem).
>
> **How to use this guide (10-hour plan):**
>
> | Slot | Hours | What to do | Where to go |
> |---|---|---|---|
> | 1 | 0:00–1:30 | Build foundation — perceptron, neuron, XOR, MLP | Module 1 |
> | 2 | 1:30–3:00 | Linear NNs — regression, GD, logistic, softmax, metrics | Module 2 |
> | 3 | 3:00–5:00 | DFNN — forward/backward, activations, parameter count | Module 3 |
> | 4 | 5:00–6:00 | Optimization & regularization | Module 4 |
> | 5 | 6:00–7:30 | CNN Part 1 — convolution, pooling, parameter math | Module 5 |
> | 6 | 7:30–9:00 | Solve every numerical template in Module 6 (high-yield) | Module 6 |
> | 7 | 9:00–10:00 | Rapid-fire revision (theory cheat sheet + bug spotting) | Module 7 |
>
> **Conventions in this guide:**
> - 💡 **Intuition** — plain-English idea
> - 🌱 **Everyday analogy** — real-life picture
> - 🧮 **Math** — the formula with each symbol explained
> - ✏️ **Worked example** — numerical, step-by-step
> - 📝 **Exam template** — exactly the variant past papers ask
> - ⚠️ **Watch out** — common mistakes and bugs

---

## Table of Contents

- [Module 0: Exam Strategy](#module-0-exam-strategy-read-first)
- [Module 1: Foundations — Perceptron, Neuron, XOR, MLP](#module-1-foundations)
- [Module 2: Linear Neural Networks — Regression and Classification](#module-2-linear-neural-networks)
- [Module 3: Deep Feed-Forward Neural Networks](#module-3-deep-feed-forward-neural-networks-dfnn)
- [Module 4: Optimization & Regularization](#module-4-optimization--regularization)
- [Module 5: CNN Part 1](#module-5-cnn-part-1)
- [Module 6: Numerical Templates](#module-6-numerical-templates-do-every-one)
- [Module 7: Rapid-Fire Revision Cheat Sheet](#module-7-rapid-fire-revision-cheat-sheet)
- [Module 8: Final Hour Checklist](#module-8-final-hour-checklist)

---

## Module 0: Exam Strategy (read first)

### What the faculty said
- Syllabus = Sessions 1–8 (up to CNN Part 1).
- 6 questions × 5 marks = **30 marks total**.
- Expect **conceptual** understanding + **validation/interpretation** of architectures/code + **numerical** aspects (parameter computation, gradient descent forward/backward).
- If anything looks ambiguous in the question, **state your assumption** and proceed; do **not** ask the invigilator about question content.

### Question-type probability (from 5 years of past papers)

| Type | Probability | Time | What it tests |
|---|---|---|---|
| Q1 — Perceptron one-epoch training table OR truth-table design OR linear-separability discussion | ~95% | 15 min | Math + concept |
| Q2 — MLP design / XOR for n inputs / depth-vs-width / decision boundary | ~80% | 15 min | Concept |
| Q3 — Forward + backward pass on a small DFNN OR computational graph | ~90% | 25 min | Pure numerical |
| Q4 — Loss derivation (∂BCE/∂Z or ∂CCE/∂Z) OR softmax+CCE numerical | ~90% | 15 min | Numerical + concept |
| Q5 — Code interpretation / fill-in-blanks / find errors | ~85% | 15 min | Code reading |
| Q6 — Mixed: parameter counting / activation comparison / learning-rate bounds / CNN | ~100% | 15 min | Concept + numerical |

### How to score every mark you can
1. **Always write the formula first**, then substitute, then evaluate. Examiners give credit for the formula even if the arithmetic is wrong.
2. **Tabular layouts** for perceptron iterations and confusion matrices — rubric awards per-row marks.
3. **Justify in 30–50 words.** Vague answers are explicitly penalised in 2024 papers.
4. **State your assumptions** at the top of each answer (e.g., "I will use bipolar inputs" or "I will use 1/N gradient convention").
5. **Each answer on a fresh page** (instruction repeated in 2024 papers).
6. **Never leave a numerical question unanswered** — even partial work on the formula gets 1–2 marks.

---

## Module 1: Foundations

### 1.1 What is Deep Learning, in one paragraph?

**Deep Learning = Machine Learning with many stacked layers, each layer transforming the input into a slightly more useful representation than the previous one.** "Deep" literally counts the layers — three or more is "deep". The whole field has just **four ingredients**:

1. **Data** D = {X, t}: N examples, each described by d features, with labels t.
2. **Model**: a parameterised function from inputs to predictions (knobs are called *weights*).
3. **Objective (Loss)**: a single number that measures how bad the predictions are. Lower = better.
4. **Learning algorithm**: gradient descent — read the slope of the loss, step opposite to it, repeat.

🌱 **Everyday picture**: A kitchen mixer is a *model*. Ingredients go in, paste comes out, and the speed dial is the *parameter*. Tasting tells you the *loss*. Adjusting the dial based on the taste is *gradient descent*.

> **AI ⊃ ML ⊃ DL**: AI is the goal (act smart). ML is one strategy (learn from data). DL is one tool inside ML (learn with deep networks).

### 1.2 The Artificial Neuron

A **biological neuron** has dendrites (inputs), a soma (combines them), an axon (output) and synapses (connection strengths). The **artificial neuron** copies the *structure*, not the chemistry.

🧮 **Math**:
z = Σᵢ₌₀^(d) wᵢ xᵢ = w₀ x₀ + w₁ x₁ + ⋯ + w_d x_d,        ŷ = f(z)

Where:
- x₀ = 1 always (the **bias trick** — folds the intercept into the weighted sum).
- w₀ is the **bias weight**; w₁, …, w_d are the feature weights.
- z is the **pre-activation** (the weighted sum / "vote tally").
- f(·) is the **activation function** (the decision).
- ŷ is the **prediction** (hat distinguishes from the true label y).

💡 **Intuition**: Each input is a vote, weighted by importance. The neuron sums the votes. The activation decides whether to fire.

⚠️ **Watch out**: Without the bias, the decision hyperplane is forced to pass through the origin — a severe restriction. **Always include the bias.**

### 1.3 The Perceptron (Rosenblatt, 1958)

The simplest possible neuron — uses a **step function** as activation:

ŷ =  { 1 (if  z ≥ 0) ; 0 (otherwise (binary representation)) }     or     ŷ = sign(z) =  { +1 (z ≥ 0) ; -1 (z < 0) } (bipolar)

💡 **Intuition**: A perceptron draws **one straight line** (or hyperplane in higher dimensions) through input space. Class 1 on one side, class 0 on the other. **One line, one decision** — that's the entire model.

🌱 **Everyday picture**: A bouncer with one rule "total score ≥ 7 = enter" weighted by dress code, ID, queue time. Cannot handle "in if tall OR well-dressed but NOT both" — that's the XOR problem.

### 1.4 Perceptron Learning Algorithm — the most important rule in Module 1

**Update rule** (memorise verbatim — it's the core of Q1 in every paper):

e = t - ŷ,        wᵢ^(new) = wᵢ^(old) + η · e · xᵢ     ∀ i

Where:
- t = true label, ŷ = perceptron output.
- e ∈ {-1, 0, +1} for binary; e ∈ {-2, 0, +2} for bipolar.
- η > 0 = **learning rate** (step size).
- The bias is updated with x₀ = 1, i.e. w₀^(new) = w₀^(old) + η · e.

💡 **Intuition**:
- Got it right? Do nothing.
- Predicted 0 but answer was 1? Push weights **up** (positive direction along x).
- Predicted 1 but answer was 0? Push weights **down**.

⚠️ **Critical sign rule**:
| Algorithm | Update direction |
|---|---|
| **Perceptron rule** | w ← w + η(t - ŷ)x — **ADD** |
| **Gradient descent** | w ← w - η ∇ J — **SUBTRACT** |

These look opposite but are consistent: in GD, the negative gradient already incorporates (t - ŷ) with a flipped sign.

📘 **Convergence theorem**: If the data is **linearly separable**, the perceptron algorithm converges to a separating hyperplane in **finite steps**. If the data is **not** linearly separable, the algorithm oscillates forever (this is the XOR problem).

✏️ **Worked example — One epoch on AND gate, binary {0, 1}**

Setup: w = [0, 0, 0] (bias, w₁, w₂), η = 0.5. Step activation: 1 if z ≥ 0 else 0. Inputs prepend x₀ = 1.

| # | x₁ | x₂ | t | z | ŷ | e=t-ŷ | Δ w₀ | Δ w₁ | Δ w₂ | New w |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0+0+0=0 | 1 | 0-1=-1 | 0.5(-1)(1)=-0.5 | 0.5(-1)(0)=0 | 0.5(-1)(0)=0 | [-0.5, 0, 0] |
| 2 | 0 | 1 | 0 | -0.5+0+0=-0.5 | 0 | 0 | 0 | 0 | 0 | [-0.5, 0, 0] |
| 3 | 1 | 0 | 0 | -0.5+0+0=-0.5 | 0 | 0 | 0 | 0 | 0 | [-0.5, 0, 0] |
| 4 | 1 | 1 | 1 | -0.5+0+0=-0.5 | 0 | 1-0=+1 | 0.5(1)(1)=0.5 | 0.5(1)(1)=0.5 | 0.5(1)(1)=0.5 | [0, 0.5, 0.5] |

End-of-epoch weights: [0, 0.5, 0.5]. Always **verify** by re-checking all 4 inputs match t.

### 1.5 The XOR Problem — the "AI winter" example

The XOR truth table:
| x₁ | x₂ | t |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plot the points: the two 1-outputs sit at **opposite corners** of the unit square. **No single straight line can separate them from the two 0-outputs.**

💡 **Why this matters**: A perceptron can ONLY draw straight lines. So a single perceptron literally cannot represent XOR — no matter how long you train. The data is **not linearly separable**.

📝 **Exam-ready answer (30–50 words):**
> "XOR is not linearly separable. The two class-1 points (0,1) and (1,0) sit at opposite corners of the unit square, with class-0 points (0,0) and (1,1) at the other two corners. Any straight line that separates the diagonals will misclassify at least one point. A perceptron implements only a hyperplane, so it cannot represent XOR. The fix is a Multi-Layer Perceptron (MLP): a hidden layer with non-linear activations gives multiple decision boundaries that together carve out non-linear regions."

### 1.6 Multi-Layer Perceptron (MLP) — the cure for XOR

An MLP stacks multiple perceptrons in **layers**:
- **Input layer** (just holds features — not counted as computational).
- **Hidden layer(s)** with non-linear activations (ReLU/sigmoid/tanh).
- **Output layer** matched to the task (sigmoid for binary, softmax for multi-class, linear for regression).

💡 **Why depth solves XOR**: Each hidden neuron draws one boundary. Two hidden neurons → two boundaries → can carve out the "exactly-one-input-is-1" region. Add an output neuron that AND-s/OR-s these → XOR.

📘 **Universal Approximation Theorem (UAT)** — Cybenko 1989, Hornik 1989:

> A feedforward network with **a single hidden layer** and **enough neurons**, using a non-polynomial non-linear activation, can approximate **any continuous function** on a compact set to arbitrary accuracy.

⚠️ **In theory** one hidden layer is enough. **In practice** deeper networks achieve the same accuracy with **exponentially fewer parameters** because they learn hierarchical / reusable features.

### 1.7 XOR for n inputs — Shallow vs Deep (high-yield exam question!)

This appears in **2023 Reg Q3, 2024 Reg Q2, 2024 Mkp Q1, 2024 EndSem Q1**.

| Approach | Hidden-layer count | Total perceptron count |
|---|---|---|
| **Shallow (single hidden layer)** | 1 hidden | 2ⁿ⁻¹ in hidden + 1 output = 2ⁿ⁻¹+1 (one per AND minterm) |
| **Deep (multiple hidden layers)** | ~ 2 log₂ n | 3(n-1) total |

✏️ **For n = 6 (2023 Regular Q3 verbatim):**
- **Shallow**: 2⁶⁻¹ = 2⁵ = 32 perceptrons in the hidden layer.
- **Deep**: 3(6-1) = 15 perceptrons; number of layers = 2⌈log₂ 6⌉ = 6.

📝 **Justification for any n (ready to drop into exam):**
> "A single hidden layer needs one perceptron per minterm (2ⁿ⁻¹ minterms for n-input XOR), giving exponential growth. A deep network can decompose the XOR hierarchically (XOR pairs of inputs, then XOR the results), producing only 3(n-1) perceptrons. This concretely demonstrates the parameter-efficiency advantage of depth."

### 1.8 Depth vs Width

- **Depth** = number of layers. Builds **hierarchical features** (edges → shapes → objects).
- **Width** = neurons per layer. Adds **parallel capacity** at one abstraction level.

📝 **Common exam phrasing:**
- *"Wider networks learn more diverse patterns at each layer; deeper networks learn more abstract compositions of those patterns. EfficientNet's compound scaling (d^α · w^β · r^γ = 2) scales both together for the best accuracy-per-FLOP."*

---

## Module 2: Linear Neural Networks

This module is one neuron with **three different activations**, giving three classic models.

| Model | Activation | Loss | Output type |
|---|---|---|---|
| Linear regression | Identity f(z)=z | MSE | Continuous number |
| Logistic regression | Sigmoid σ(z) | Binary Cross-Entropy (BCE) | Probability ∈ (0,1) |
| Softmax regression | Softmax softmax(z)ₖ | Categorical Cross-Entropy (CCE) | Probability distribution over K classes |

### 2.1 Linear Regression — predicting a number

**Model**: ŷ = w₀ + w₁ x₁ + ⋯ + w_d x_d = wᵀ x (with bias absorbed via x₀=1).

**For all N examples at once**: ŷ = X w where X is the **design matrix** of shape N × (d+1).

**Loss — Mean Squared Error (MSE)**:
J(w) = 1/NΣᵢ₌₁^(N) (y⁽ⁱ⁾ - ŷ⁽ⁱ⁾)² = 1/N ||y - Xw||²

(Some textbooks use (1)/(2N) to cancel the 2 from differentiation. Both are accepted — **state the convention you use**.)

**Why squared error?**
- **Differentiable everywhere** — gradient descent loves it.
- **Convex** for linear models — exactly one global minimum, no local traps.
- **Penalises large errors quadratically** — forces the model to fix the worst misses.
- **Statistical interpretation**: minimising MSE = maximum likelihood under Gaussian noise.

⚠️ MSE is **sensitive to outliers**. Use Mean Absolute Error (MAE) when outliers exist.

🧮 **Gradient (memorise!)**:
∇ J = 2/N Xᵀ (ŷ - y)     or with  t1 / 2N  convention:  ∇ J = 1/N Xᵀ (ŷ - y)

**Update**: w ← w - η ∇ J (subtract — gradient *descent*).

✏️ **Worked example — One step of batch GD on slide example**

Setup: 3 points (x₁,y₁)=(1,2), (x₂,y₂)=(2,4), (x₃,y₃)=(3,5). Initial w=[0,0] (bias and slope), η=0.1. Use 1/N MSE → ∇ J = 2/N Xᵀ (ŷ - y).

```
Step 1. ŷ = X·w = [0,0,0]ᵀ.
Step 2. e = ŷ - y = [-2,-4,-5]ᵀ.
Step 3. Loss J(0) = (1/3)(4+16+25) = 15.0.
Step 4. ∇J = (2/3) Xᵀ e
              = (2/3) [[1,1,1],[1,2,3]] · [-2,-4,-5]ᵀ
              = (2/3) [-11, -23]ᵀ
              = [-7.33, -15.33]ᵀ.
Step 5. w⁽¹⁾ = w - η∇J = [0,0] - 0.1·[-7.33,-15.33] = [0.733, 1.533].
Step 6. New ŷ = [2.27, 3.80, 5.33]; new e = [0.27, -0.20, 0.33];
        new loss = (1/3)(0.071+0.040+0.111) = 0.074.
```

The loss dropped from 15.0 to 0.074 in **one step**. That's gradient descent in action.

### 2.2 Why we need different losses for classification

If you used MSE on labels {0, 1}:
1. The model predicts any real number ∈ ℝ — but probabilities must lie in [0, 1].
2. MSE doesn't punish confident wrong predictions hard enough.

**Fix**: squash the output through a function that maps ℝ → (0,1) — the **sigmoid** — and use a probability-aware loss — **cross-entropy**.

### 2.3 The Sigmoid Function

σ(z) = (1) / (1 + e^(-z))

- **Range**: (0, 1) — interpretable as a probability.
- σ(0) = 0.5 — maximum uncertainty.
- σ(z) → 1 as z → +∞, σ(z) → 0 as z → -∞.

🧮 **Derivative (memorise — used in backprop)**:
σ'(z) = σ(z)(1 - σ(z))

**Maximum derivative = 0.25** (at z = 0). This is the source of the **vanishing gradient problem** in deep networks: stacking sigmoids multiplies these small numbers, killing the gradient at early layers.

🌱 **Everyday analogy**: A weather app says "80% chance of rain". Behind the scenes: a weighted sum z of features, then sigmoid → 0.80. Threshold 0.5 is the umbrella decision.

### 2.4 Logistic Regression — Binary Classification

**Model**:
z = wᵀ x,        ŷ = σ(z) = P(y=1 mid x; w)

**Decision rule**: predict class 1 if ŷ ≥ 0.5 (equivalently, z ≥ 0); else class 0.

**Decision boundary**: the hyperplane wᵀ x = 0 (linear in feature space — so logistic regression is still a *linear classifier*, just with a probabilistic output).

**Loss — Binary Cross-Entropy (BCE)**:
mathcal{L} = -[y log ŷ + (1-y) log(1-ŷ)]

When y=1: loss = -log ŷ. Goes to 0 when ŷ → 1, blows up when ŷ → 0.
When y=0: loss = -log(1-ŷ). Mirror image.

📘 **Why this loss**:
- **Probabilistic** — derived from maximum likelihood.
- **Confident wrong answers are punished by huge log losses** — sharper learning signal than MSE.
- **Convex** for logistic regression — single global minimum.
- **The gradient simplifies beautifully** (next subsection).

### 2.5 ⭐ Derivation of ∂BCE/∂z (full marks 5-mark question — appeared in 2023 Reg Q4 verbatim)

This is THE most common derivation question. Memorise it step by step.

**Goal**: show that for ŷ = σ(z) and L = -[y log ŷ + (1-y)log(1-ŷ)],

**∂ L / ∂ z = ŷ - y**

**Proof (write all 5 steps in the exam):**

**Step 1**: Differentiate L w.r.t. ŷ (treat ŷ as a variable):
(∂ L)/(∂ ŷ) = -y/ŷ + (1-y)/(1-ŷ) = (-y(1-ŷ) + (1-y)ŷ)/(ŷ(1-ŷ)) = (ŷ - y)/(ŷ(1-ŷ))

**Step 2**: Differentiate ŷ = σ(z):
(∂ ŷ)/(∂ z) = σ(z)(1-σ(z)) = ŷ(1-ŷ)

**Step 3**: Apply chain rule:
(∂ L)/(∂ z) = (∂ L)/(∂ ŷ) · (∂ ŷ)/(∂ z)

**Step 4**: Multiply (the ŷ(1-ŷ) terms cancel beautifully):
(∂ L)/(∂ z) = (ŷ - y)/(ŷ(1-ŷ)) · ŷ(1-ŷ) = ŷ - y

**Step 5**: For weights, chain again with z = wᵀ x:
∇_w L = (∂ L)/(∂ z) · (∂ z)/(∂ w) = (ŷ - y)   x

✓ The "ŷ - y" identity is the **single most useful identity in deep learning**. It works for **both** sigmoid+BCE AND softmax+CCE.

### 2.6 Softmax — Multi-class Classification

For K classes, we need K output neurons. The softmax turns K raw logits into a valid probability distribution:

ŷₖ = softmax(z)ₖ = e^(zₖ) / Σⱼ₌₁^(K) e^(zⱼ),     k=1,…,K

**Properties**:
- All outputs are positive (exponential of anything > 0).
- They sum to 1 (we divide by the total) → valid probability distribution.
- A "smooth argmax": the largest logit gets the largest share, but every class gets non-zero probability.

⚠️ **Numerical stability trick** (often asked in code questions):
```
softmax(z) = exp(z - max(z)) / sum(exp(z - max(z)))
```
Subtracting the max keeps exponentials manageable; mathematically identical because the constant cancels.

**Loss — Categorical Cross-Entropy (CCE)** with one-hot label y:
mathcal{L} = -Σₖ₌₁^(K) yₖ log ŷₖ

Since y is one-hot (only y_c = 1 for the correct class c), this reduces to mathcal{L} = -log ŷ_c — just the negative log of the correct class's predicted probability.

🧮 **Gradient identity (same elegant form!)**:
(∂ L)/(∂ zₖ) = ŷₖ - yₖ

### 2.7 ⭐ Derivation of ∂CCE/∂Z (5-mark — appeared in 2023 Mkp Q2 verbatim)

For one-hot y and softmax output ŷₖ:

**Step 1**: (∂ L)/(∂ ŷₖ) = -(yₖ)/(ŷₖ)

**Step 2**: (∂ ŷₖ)/(∂ zₖ) = ŷₖ (1 - ŷₖ) (the diagonal entry of the softmax Jacobian)

**Step 3**: Chain: (∂ L)/(∂ zₖ) = -(yₖ)/(ŷₖ) · ŷₖ(1-ŷₖ) = -yₖ(1-ŷₖ)

**Step 4**: For one-hot labels (where Σⱼ yⱼ = 1), this simplifies to ŷₖ - yₖ ✓

### 2.8 Worked numerical: Softmax + CCE on 3-class logits (Dec 2025 Q4a verbatim)

Logits z = [2.0, 1.0, 0.5], true class = 1 (one-hot y = [0, 1, 0]):

```
Step 1. Exponentiate:
  e².0 = 7.389,  e¹.0 = 2.718,  e⁰.5 = 1.649
Step 2. Sum = 7.389 + 2.718 + 1.649 = 11.756
Step 3. Softmax:
  ŷ₀ = 7.389/11.756 ≈ 0.628
  ŷ₁ = 2.718/11.756 ≈ 0.231
  ŷ₂ = 1.649/11.756 ≈ 0.140    (sum ≈ 1.0 ✓)
Step 4. CCE = -log(0.231) ≈ 1.466
Step 5. Predicted class = argmax = class 0 (probability 0.628)
Step 6. Prediction is INCORRECT (true label = class 1)
```

### 2.9 Classification Metrics — the Confusion Matrix family

| Predicted Pos | Predicted Neg |
|---|---|
| **Actual Pos** | True Positive (TP) | False Negative (FN) |
| **Actual Neg** | False Positive (FP) | True Negative (TN) |

**Four metrics from these four numbers**:

| Metric | Formula | "Question it answers" |
|---|---|---|
| **Accuracy** | (TP+TN) / N | Of all predictions, how many were correct? |
| **Precision** | TP / (TP+FP) | Of predicted positives, how many were real? |
| **Recall (Sensitivity)** | TP / (TP+FN) | Of actual positives, how many did we catch? |
| **F1-score** | 2 · Prec · Rec / (Prec+Rec) | Harmonic mean of P and R (low if either is low) |

📝 **Use cases (canned 30–50-word answers)**:
- **Recall matters more for**: medical diagnosis (a missed cancer kills), fraud detection (a missed fraud is expensive).
- **Precision matters more for**: spam filtering (false positives lose your real emails), product recommendations (wrong recs annoy users).
- **Accuracy is misleading** when classes are imbalanced. A "predict-all-healthy" classifier gets 95% accuracy on a 5% disease prevalence dataset, but 0% recall — useless. Always inspect P, R, and F1 for imbalanced problems.

### 2.10 Threshold-shifting trade-off (high-yield 30–50-word answer)

> "Increasing the classification threshold from 0.5 to a higher value (e.g., 0.7) makes the model **more conservative** about predicting positive. Effect: **fewer false positives** (precision rises), but **more false negatives** (recall falls). For life-threatening disease detection, **lower** the threshold to favor recall — the cost of a missed disease far exceeds the cost of an extra screening test."

### 2.11 SGD vs Mini-batch vs Batch GD — every paper asks this

| Variant | Examples per update | Pros | Cons |
|---|---|---|---|
| **Batch GD** | All N | Smooth, exact gradient | Slow, memory-heavy |
| **SGD** | 1 | Fast updates, can escape sharp minima (noise helps) | Very noisy curve |
| **Mini-batch GD** | B (typically 32–256) | **Best of both** — GPU-friendly, moderate noise | Need to tune B |

📝 **Canned exam answer**:
> "Batch GD uses all N examples per step — stable but slow. SGD uses one example — fast but noisy. Mini-batch GD (32–256 examples) is the industry standard: it balances stable gradients with fast iterations and exploits GPU vectorisation. The gradient noise in mini-batch GD also acts as a mild regulariser, helping escape sharp local minima."

---

## Module 3: Deep Feed-Forward Neural Networks (DFNN)

This is **the** highest-mark module: ~10 marks of any MidSem come from forward/backward pass + parameter counting.

### 3.1 DFNN Architecture & Forward Pass

A DFNN has:
- An **input layer** (just holds features — not counted as a computational layer).
- **L computational layers**: hidden layers 1, 2, …, L-1 + output layer L.
- Layer ell has n_ell neurons; weight matrix W⁽ℓ⁾ of shape n_{ell-1} × n_ell; bias b⁽ℓ⁾ of shape n_ell.

**Forward pass — repeat for ell = 1, …, L:**
z⁽ℓ⁾ = (W⁽ℓ⁾)ᵀ h^(ℓ-1) + b⁽ℓ⁾,        h⁽ℓ⁾ = f⁽ℓ⁾(z⁽ℓ⁾)

where h⁽⁰⁾ = x (input) and h^(L) = ŷ (final prediction).

🌱 **Everyday picture**: Each layer is one stage of an assembly line. Stage 1 finds edges, Stage 2 groups them into shapes, Stage 3 names parts, Stage 4 says "that's a cat". Each ← is one learned matrix multiply + activation.

### 3.2 ⭐ Activation Functions — full comparison table (memorise!)

| Activation | Formula | Range | Derivative | Use case | Pros | Cons |
|---|---|---|---|---|---|---|
| **Identity** | f(z) = z | ℝ | 1 | Regression output | No info loss; gradient flows perfectly | Cannot model non-linearity |
| **Sigmoid** | 1 / 1+e^(-z) | (0,1) | σ(1-σ), max = 0.25 | Binary output | Probabilistic interpretation | **Vanishing gradient** in deep nets; not zero-centred |
| **Tanh** | e^z-e^(-z) / e^z+e^(-z) | (-1,1) | 1-tanh²(z), max = 1 | RNN cells, shallow nets | Zero-centred, stronger gradient than sigmoid | Still saturates → vanishing gradient |
| **ReLU** | max(0, z) | [0, ∞) | 1 if z>0 else 0 | **Default for hidden layers** | No saturation on positive side; cheap; sparse activations | **Dying ReLU** for z < 0; not differentiable at 0 (rarely matters) |
| **Leaky ReLU** | max(α z, z), α=0.01 | ℝ | 1 if z>0 else α | Replaces ReLU when many neurons die | Fixes dying ReLU | One extra hyperparameter |
| **Softmax** | e^(zₖ) / Σ e^(zⱼ) | (0,1), sum=1 | ŷₖ(1-ŷₖ) on diag | Multi-class output | Probability distribution | Output coupling makes interpretability harder |

📝 **Activation rules-of-thumb (canned)**:
- **Hidden layers**: ReLU (default) → if dying, switch to Leaky ReLU.
- **Binary output**: sigmoid + BCE.
- **Multi-class output**: softmax + CCE.
- **Regression output**: identity + MSE.
- **NEVER use sigmoid/tanh in hidden layers of deep networks** — vanishing gradient.

### 3.3 The Vanishing Gradient Problem — high-yield concept

💡 **Why it happens**: Backprop multiplies derivatives layer by layer. Sigmoid's max derivative is 0.25. In an L-layer network, the gradient at the first layer scales like 0.25^L. For L=8: 0.25⁸ ≈ 1.5 × 10⁻⁵ — essentially zero. Early layers stop learning.

**Solutions** (every fix asked in 2024 papers):
1. **ReLU activation** — derivative is 1 for z>0, no shrinkage.
2. **He / Xavier initialization** — keeps the variance of activations stable across layers.
3. **Batch Normalization** — re-centers and re-scales activations every layer.
4. **Residual / skip connections** (ResNets) — let gradients bypass layers.
5. **LSTM / GRU gates** for sequence models — let gradients flow through cell states unmodified.

**Exploding gradients** (the mirror problem):
- Symptom: loss → NaN/Inf, weights blow up.
- Fix: **gradient clipping** (cap ||∇|| at a threshold), proper initialization, smaller learning rate.

### 3.4 ⭐ Backpropagation — the chain rule, applied backwards

💡 **Intuition**: A bad prediction comes out at layer L. We trace the "blame" back through the network using the chain rule.

**Define the error signal at layer ℓ**: δ⁽ℓ⁾ = ∂ J / ∂ z⁽ℓ⁾.

**Three core equations of backprop** (memorise — write these as headings in any backprop question):

**1. Output error** (depends on loss + activation pairing):
δ^(L) = ŷ - y     (for sigmoid+BCE or softmax+CCE)

**2. Backpropagate the error**:
δ⁽ℓ⁾ = ( W^(ℓ+1) δ^(ℓ+1) ) ⊙ f'⁽ℓ⁾(z⁽ℓ⁾)

where ⊙ is **element-wise** multiplication and f' is the activation's derivative.

**3. Parameter gradients**:
∂ J / ∂ W⁽ℓ⁾ = h^(ℓ-1) (δ⁽ℓ⁾)ᵀ,        ∂ J / ∂ b⁽ℓ⁾ = δ⁽ℓ⁾

**Update**: W ← W - η   ∂ J/∂ W, b ← b - η   ∂ J / ∂ b.

📘 **Why backprop made deep learning possible**: Naively computing gradients = one forward pass per parameter (O(P²)). Backprop = one forward + one backward pass total (O(P)). For P=10⁶, that's a million-fold speedup.

### 3.5 ⭐ Full numerical: Forward + Backward on a 2-3-2 network (Companion Session 6)

This is **the** template question — virtually every MidSem has a smaller version.

**Setup**:
- Input x = [1.0, 0.5]ᵀ, true label y = [1, 0]ᵀ (one-hot).
- Layer 1: 3 neurons with **ReLU**.
- Layer 2: 2 neurons with **sigmoid + BCE**.
- W^([1]) = [[0.5, 0.2], [-0.3, 0.8], [0.1, -0.4]],     b^([1]) = [0.1, -0.2, 0.3]ᵀ
- W^([2]) = [[0.4, -0.1, 0.6], [0.2, 0.7, -0.3]],     b^([2]) = [0.2, -0.1]ᵀ
- η = 0.1.

**FORWARD PASS**:

```
Step 1. z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
  z₁⁽¹⁾ = 0.5(1.0) + 0.2(0.5) + 0.1  = 0.50 + 0.10 + 0.10 = 0.70
  z₂⁽¹⁾ = -0.3(1.0) + 0.8(0.5) + (-0.2) = -0.30 + 0.40 - 0.20 = -0.10
  z₃⁽¹⁾ = 0.1(1.0) + (-0.4)(0.5) + 0.3 = 0.10 - 0.20 + 0.30 = 0.20
  → z⁽¹⁾ = [0.70, -0.10, 0.20]ᵀ

Step 2. a⁽¹⁾ = ReLU(z⁽¹⁾)
  a⁽¹⁾ = [0.70, 0.00, 0.20]ᵀ   ← neuron 2 is DEAD (z<0)

Step 3. z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
  z₁⁽²⁾ = 0.4(0.7) + (-0.1)(0) + 0.6(0.2) + 0.2 = 0.28 + 0 + 0.12 + 0.20 = 0.60
  z₂⁽²⁾ = 0.2(0.7) + 0.7(0) + (-0.3)(0.2) + (-0.1) = 0.14 + 0 - 0.06 - 0.10 = -0.02
  → z⁽²⁾ = [0.60, -0.02]ᵀ

Step 4. ŷ = σ(z⁽²⁾)
  ŷ₁ = 1/(1 + e⁻⁰·⁶⁰) = 1/(1 + 0.549) = 1/1.549 ≈ 0.646
  ŷ₂ = 1/(1 + e⁰·⁰²) = 1/(1 + 1.020) = 1/2.020 ≈ 0.495
  → ŷ = [0.646, 0.495]ᵀ

Step 5. BCE loss L = -½[log(0.646) + log(0.505)]
                   = -½[(-0.437) + (-0.683)] = -½(-1.120) = 0.560
```

**BACKWARD PASS**:

```
Step 6. δ⁽²⁾ = ŷ - y = [0.646 - 1, 0.495 - 0] = [-0.354, 0.495]ᵀ

Step 7. ∂L/∂W⁽²⁾ = δ⁽²⁾ · (a⁽¹⁾)ᵀ
        = [-0.354;  0.495] · [0.7, 0.0, 0.2]
        = [[-0.248, 0, -0.071],
           [ 0.347, 0,  0.099]]
        ∂L/∂b⁽²⁾ = δ⁽²⁾ = [-0.354, 0.495]ᵀ

Step 8. δ⁽¹⁾ = (W⁽²⁾ᵀ δ⁽²⁾) ⊙ ReLU'(z⁽¹⁾)
   W⁽²⁾ᵀ δ⁽²⁾ = [[0.4, 0.2], [-0.1, 0.7], [0.6, -0.3]] · [-0.354; 0.495]
              = [0.4(-0.354)+0.2(0.495); -0.1(-0.354)+0.7(0.495); 0.6(-0.354)+(-0.3)(0.495)]
              = [-0.043,  0.382, -0.361]ᵀ
   ReLU'(z⁽¹⁾) = [1, 0, 1]ᵀ      (because z₂⁽¹⁾ = -0.10 ≤ 0)
   δ⁽¹⁾ = [-0.043, 0.000, -0.361]ᵀ      ← dead neuron blocks the gradient

Step 9. ∂L/∂W⁽¹⁾ = δ⁽¹⁾ · xᵀ
        = [-0.043; 0; -0.361] · [1.0, 0.5]
        = [[-0.043, -0.022],
           [ 0.000,  0.000],
           [-0.361, -0.181]]
        ∂L/∂b⁽¹⁾ = [-0.043, 0, -0.361]ᵀ

Step 10. WEIGHT UPDATES (η = 0.1):
   W⁽²⁾_new = W⁽²⁾ - 0.1·∂L/∂W⁽²⁾
            = [[0.425, -0.100, 0.607],
               [0.165,  0.700, -0.310]]
   b⁽²⁾_new = [0.235, -0.150]ᵀ
   W⁽¹⁾_new = [[0.504, 0.202],
               [-0.300, 0.800],   ← row 2 unchanged (dead neuron)
               [ 0.136, -0.382]]
   b⁽¹⁾_new = [0.104, -0.200, 0.336]ᵀ
```

**Sanity check**: Row 2 of W^([1]) and b₂^([1]) did NOT change because hidden neuron 2 was dead. This is the **dying ReLU problem** in action.

### 3.6 ⭐ Parameter Counting — guaranteed exam question

**Fully connected layer with n_{ in} inputs and n_{ out} outputs**:
params = (n_{ in} × n_{ out}) + n_{ out}
(weights + biases)

**Total network parameters**:
P_{ total} = Σ_{ell=1}^(L) (n_{ell-1} n_ell + n_ell)

✏️ **Worked example — 2-3-2 architecture above**:
- Layer 1: 2 × 3 + 3 = 9
- Layer 2: 3 × 2 + 2 = 8
- **Total**: 17 parameters.

✏️ **Worked — 4-input network with hidden sizes [10, 8, 5] and 3 outputs (ImageNet-style toy)**:
- Layer 1: 4 × 10 + 10 = 50
- Layer 2: 10 × 8 + 8 = 88
- Layer 3: 8 × 5 + 5 = 45
- Output: 5 × 3 + 3 = 18
- **Total**: 201 parameters.

### 3.7 Computational Graphs (asked in 2023 Reg)

A computational graph represents a function as nodes (operations) and edges (variables). Three benefits:
1. **Visualises** dependencies clearly.
2. **Auto-differentiation**: chain rule applied along edges → backprop is mechanical.
3. **Modularity**: each node only needs to know how to compute its forward and its local derivative.

📝 **Exam-ready 50-word answer**:
> "Computational graphs represent expressions as DAGs of operations: each node is an op, each edge is a value. Forward pass = traverse left-to-right computing values. Backward pass = traverse right-to-left applying chain rule at each node. This makes auto-differentiation systematic and is the foundation of PyTorch/TensorFlow."

📝 **Common task**: draw the computational graph for f(x₁, x₂) = (x₁ + x₂)² + 3x₁. Nodes for +, ×, square; trace forward then backward to get ∂ f/∂ x₁ = 2(x₁+x₂) + 3.

---

## Module 4: Optimization & Regularization

### 4.1 Challenges in deep network optimization

The loss surface of a deep network is **non-convex**. Three obstacles:

| Obstacle | What it is | Symptom | Fix |
|---|---|---|---|
| **Local minima** | Loss is locally minimal but globally not | Loss plateaus high | Random restarts; momentum; SGD noise |
| **Saddle points** | Gradient is zero but it's a saddle, not a minimum | Loss plateaus, then maybe escapes | Momentum; second-order info |
| **Plateaus** | Long flat regions | Slow training | Adaptive learning rate (Adam); LR scheduling |

### 4.2 Optimizers — comparison table (memorise)

| Optimizer | Update rule | Key idea | When to use |
|---|---|---|---|
| **SGD** | w ← w - η ∇ J | Bare-bones | Small models, fine-tuning |
| **SGD + Momentum** | v ← β v + ∇ J; w ← w - η v | "Roll downhill" — accumulate velocity | Image classification with SGD |
| **Nesterov Momentum** | Look-ahead gradient before update | Reduces overshoot | Slightly better than vanilla momentum |
| **Adagrad** | Scale each parameter's LR by 1/√{Σ g²} | Adapts LR per parameter | Sparse features (NLP) |
| **RMSProp** | Adagrad with **exponential moving average** of g² | Doesn't shrink LR forever | RNNs |
| **Adam** | Momentum + RMSProp combined; beta₁=0.9, beta₂=0.999 | **Default everywhere** | Most modern DL |

🌱 **Everyday picture for momentum**: A heavy ball rolling downhill. Even if the ground levels off briefly (a saddle), the ball keeps moving. Plain SGD stops at every flat patch; momentum doesn't.

📝 **Adam in 50 words (canned)**:
> "Adam combines momentum (first moment of gradients, mₜ) with RMSProp's adaptive per-parameter learning rate (second moment, vₜ). Bias-correction terms m̂ₜ, v̂ₜ counter the zero-initialization. Update: wₜ₊₁ = wₜ - η m̂ₜ / (√{v̂ₜ} + ε). It is robust to learning-rate choice and works well on a wide range of problems."

### 4.3 Learning Rate — the most fragile dial

| η chosen | Symptom |
|---|---|
| Too small | Loss decreases imperceptibly slowly |
| Just right | Smooth, monotonic decrease |
| Slightly too large | Loss oscillates but trends down |
| Way too large | Loss rises, NaN/Inf — divergence |

**Learning-rate schedules**:
- **Step decay**: η = eta₀ · 0.5^(⌊ t / k ⌋) — halve every k epochs.
- **Exponential decay**: η = eta₀ · e^(-kt).
- **Cosine annealing**: smoothly anneal from eta₀ to eta_{ min}.
- **Warm restarts**: cosine, then jump back up — escapes shallow minima.

📝 **For convergence guarantee on a strongly convex objective with Lipschitz gradient L_{ Lip}**: η < 2/L_{ Lip} — outside this range, GD diverges.

### 4.4 Regularization — fighting overfitting

**Overfitting** = low training loss, high test loss. The model memorises rather than generalises.

**Six techniques** (all asked in past papers):

| Technique | What it does | Effect |
|---|---|---|
| **L2 (weight decay)** | Add λ/2 ||w||² to loss | Shrinks all weights; prefers smaller-magnitude solutions |
| **L1 (lasso)** | Add λ ||w||₁ to loss | Drives some weights exactly to 0 → sparse model |
| **Dropout** | Zero out fraction p of activations during training; scale at test time | Prevents co-adaptation; like ensembling many sub-networks |
| **Batch Normalization** | Normalize each layer's inputs to mean 0, variance 1; learnable γ, β | Stabilises gradients, allows higher LR, mild regularisation |
| **Early stopping** | Halt training when validation loss starts rising | Free regulariser; matches model complexity to data |
| **Data augmentation** | Synthesise new training data (flip, rotate, crop) | Effectively increases dataset size |

📝 **L2 vs L1 (50-word ready answer)**:
> "L2 regularisation adds λ ||w||² to the loss; the gradient of this term is λ w, shrinking weights proportionally — produces small but non-zero weights. L1 adds λ ||w||₁; its sub-gradient λ · sign(w) shrinks weights by a fixed amount, driving many to exactly zero, yielding a sparse, feature-selecting model."

📝 **Dropout (canned 50 words)**:
> "Dropout randomly zeroes a fraction p of neurons during each forward pass. The network can no longer rely on any single neuron, so it learns redundant representations — equivalent to training an exponential ensemble of sub-networks. At test time, all neurons are kept but their outputs are scaled by 1-p to match expected magnitudes."

📝 **Batch Normalization (canned)**:
> "BN normalises each mini-batch's activations to zero mean and unit variance, then applies learnable scale γ and shift β: x̂ = (x - mu_B)/√{sigma_B² + ε}, y = γ x̂ + β. Benefits: faster convergence, allows higher LR, mild regularisation, less sensitive to initialisation. Critical for training very deep networks."

📝 **Early stopping (canned)**:
> "Monitor validation loss every epoch. If it has not improved for a 'patience' window (e.g. 10 epochs), stop training and restore the best weights. This prevents overfitting at zero extra cost — the model is implicitly regularised by the limited training time."

### 4.5 Parameter Initialization — why zero init fails

🚫 **Zero init**: every neuron computes the same gradient → they learn identical features → the network has effective width 1. **Symmetry breaking** is required.

| Scheme | Variance | Best for |
|---|---|---|
| **Xavier (Glorot)** | 1/n_{ in} | Sigmoid, tanh |
| **He (Kaiming)** | 2/n_{ in} | ReLU |

**Biases**: initialise to 0 (or small positive value for ReLU to avoid dead neurons at the start).

### 4.6 Covariate Shift & Internal Covariate Shift

**Covariate shift**: train and test inputs have different distributions. Causes test-time performance to drop.

**Internal covariate shift**: in a deep network, each layer's input distribution changes as the previous layers update — the layer keeps having to "track" a moving target. **Batch Normalization** addresses this by normalising every layer's inputs.

📝 **Canned answer**: "Internal covariate shift is the phenomenon where a deep network's layer-wise input distributions shift during training, slowing convergence. Batch Normalization normalises layer inputs to a stable distribution, decoupling each layer's optimisation from updates upstream and stabilising training."

---

## Module 5: CNN Part 1

### 5.1 Why convolutions for images?

A fully-connected layer on a 224×224×3 image has 150,528 input neurons. A single hidden layer of 1000 units = **150 million weights** — infeasible.

**Three problems FC layers have on images**:
1. **Too many parameters** → impossible to train.
2. **No translation invariance** — a cat in the corner is treated as totally different from a cat in the centre.
3. **Ignores spatial structure** — pixels close to each other are related, but FC layers treat them independently.

**CNN's three answers** (high-yield exam topic):
1. **Sparse interactions / local connectivity** — each output looks at a small patch (the receptive field) instead of the whole image.
2. **Parameter sharing** — the same kernel is applied at every spatial location.
3. **Equivariance to translation** — shift the input, the feature map shifts the same way.

### 5.2 The Convolution Operation

🧮 **2D convolution formula**:
Y[i, j] = Σₘ Σₙ K[m, n] · X[i+m, j+n] + b

where K is the **kernel** (filter), X is the input, Y is the output **feature map**.

(In ML the operation is technically *cross-correlation*, but everyone calls it convolution. The kernel is **not flipped** before sliding.)

🌱 **Everyday picture**: A torch slides over a photograph. At each position the torch reveals a 3×3 patch; the kernel weights say how to combine those 9 pixels into one number. That number is one pixel of the output feature map.

✏️ **Worked: Convolve 4×4 input with 3×3 kernel, no padding, stride 1**

Input:
```
X = [[1, 2, 3, 0],
     [4, 5, 6, 1],
     [7, 8, 9, 2],
     [3, 4, 5, 3]]
```
Kernel:
```
K = [[1, 0, -1],
     [1, 0, -1],
     [1, 0, -1]]    (vertical edge detector)
```

Output is (4-3+1) × (4-3+1) = 2 × 2.

```
Y[0,0] = 1·1 + 0·2 + (-1)·3 + 1·4 + 0·5 + (-1)·6 + 1·7 + 0·8 + (-1)·9
       = 1 - 3 + 4 - 6 + 7 - 9 = -6
Y[0,1] = 1·2 + 0·3 + (-1)·0 + 1·5 + 0·6 + (-1)·1 + 1·8 + 0·9 + (-1)·2
       = 2 + 5 - 1 + 8 - 2 = 12        wait, check signs:
       = 2 - 0 + 5 - 1 + 8 - 2 = 12
Y[1,0] = 1·4 + 0·5 + (-1)·6 + 1·7 + 0·8 + (-1)·9 + 1·3 + 0·4 + (-1)·5
       = 4 - 6 + 7 - 9 + 3 - 5 = -6
Y[1,1] = 1·5 + 0·6 + (-1)·1 + 1·8 + 0·9 + (-1)·2 + 1·4 + 0·5 + (-1)·3
       = 5 - 1 + 8 - 2 + 4 - 3 = 11

Y = [[-6, 12],
     [-6, 11]]
```

### 5.3 ⭐ Output Size Formula (memorise — every CNN paper has it!)

For one spatial dimension:
**O = ⌊ (I + 2P - K) / S ⌋ + 1**

where:
- I = input size (height or width)
- K = kernel size
- P = padding (zeros added around input)
- S = stride (step size of the sliding kernel)

✏️ **Quick examples (memorise these 6 cases — they cover ~90% of exam variants):**

| Input | Kernel | Pad | Stride | Output | Reason |
|---|---|---|---|---|---|
| 32 | 3 | 0 | 1 | 30 | (32+0-3)/1 + 1 = 30 |
| 32 | 3 | 1 | 1 | **32** | "same" padding (P = (K-1)/2) |
| 32 | 3 | 0 | 2 | 15 | (32-3)/2 + 1 = 15.5 → 15 |
| 32 | 5 | 2 | 1 | **32** | "same" padding |
| 28 | 5 | 0 | 1 | 24 | (28-5)/1 + 1 = 24 |
| 224 | 7 | 3 | 2 | 112 | (224+6-7)/2 + 1 = 112 |

📘 **Padding types**:
- **Valid (no padding)**: P=0 — output shrinks.
- **Same padding**: P = (K-1)/2 for odd K — output equals input size.
- **Full padding**: P = K-1 — output grows.

### 5.4 ⭐ CNN Parameter Counting

For a Conv layer with kernel size K × K, C_{ in} input channels, C_{ out} output channels (=number of filters):

**params = (K × K × C_{ in**) × C_{ out} + C_{ out}**

(weights of all filters + one bias per filter)

📘 **Pooling layers** (max/avg) have **0 parameters**.
📘 **Activation layers** have **0 parameters**.
📘 **Batch Norm** has 2 · C_{ out} trainable parameters (γ, β).

✏️ **Worked example — full CNN**:

| Layer | Specs | Output shape | Parameters |
|---|---|---|---|
| Input | 32×32×3 image | 32×32×3 | 0 |
| Conv1 | 3×3 kernel, 32 filters, P=1, S=1 | 32×32×32 | (3 · 3 · 3) · 32 + 32 = 896 |
| Pool1 | 2×2 max-pool, S=2 | 16×16×32 | 0 |
| Conv2 | 3×3 kernel, 64 filters, P=1, S=1 | 16×16×64 | (3 · 3 · 32) · 64 + 64 = 18,496 |
| Pool2 | 2×2 max-pool, S=2 | 8×8×64 | 0 |
| Flatten | | 4096 | 0 |
| FC1 | 4096 → 128 | 128 | 4096 · 128 + 128 = 524,416 |
| Output | 128 → 10 | 10 | 128 · 10 + 10 = 1,290 |
| **Total** | | | **545,098** |

Notice: the FC layers dominate the parameter count, even though the Conv layers do most of the spatial work. This is why modern CNNs (ResNet, EfficientNet) use **global average pooling** before the classifier — to slash FC parameters.

### 5.5 Pooling Layers

**Pooling** = downsample by taking max or average over each window.

| Type | What it does | When |
|---|---|---|
| **Max pooling** | Takes the max of each window | Default; preserves strong activations (edges, textures) |
| **Average pooling** | Takes the mean | Smoother; used in classifier heads (Global Average Pool) |

✏️ **Worked example — 2×2 max-pool with stride 2**:
Input:
```
[[1, 3, 2, 4],
 [5, 6, 7, 8],
 [3, 2, 1, 0],
 [1, 2, 3, 4]]
```
Output (each 2×2 patch → max):
```
[[6, 8],
 [3, 4]]
```

📘 **Why pooling**:
1. **Reduces spatial dimensions** → fewer parameters in later layers, less compute.
2. **Translation invariance** — small shifts don't change the output.
3. **Larger receptive field** in subsequent layers without more parameters.

📘 **Pooling output size**: same formula as Conv but no padding typically.

### 5.6 Special Convolutions to know

| Type | What it does | Use case |
|---|---|---|
| **1×1 convolution** | Linear combination across channels at each spatial location | Channel-wise dimensionality reduction (Inception, ResNet bottleneck) |
| **Dilated (atrous)** | Skips r-1 pixels between kernel taps; effective kernel size K + (K-1)(r-1) | Larger receptive field without more params (semantic segmentation) |
| **Strided convolution** | Stride > 1 — downsamples while convolving | Replaces pooling in modern designs |
| **Depthwise separable** | Depthwise conv (per channel) + 1×1 pointwise conv | Far fewer params (MobileNet, EfficientNet) |
| **Transposed (deconv)** | Upsamples — used in segmentation/generative models | Decoders |

### 5.7 Receptive Field

**Receptive field** = the region of the input that one output unit "sees".

- After one 3times3 conv: RF = 3.
- Two 3times3 convs stacked: RF = 3 + 2 = 5 (each layer adds K-1).
- Three: RF = 7. Etc.

📝 **Why two 3×3 convs > one 5×5 conv**: same receptive field (5), but
- Fewer parameters: 2 · (3 · 3) = 18 vs 25.
- One extra non-linearity in between → **more expressive**.

### 5.8 CNN Design Patterns

A typical CNN looks like:
```
[CONV → ReLU → CONV → ReLU → POOL] × N → [FC → ReLU] × M → SOFTMAX
```

**Design rules of thumb**:
- Spatial dimensions **decrease** as you go deeper (via pooling or stride-2 conv).
- Number of channels **increases** as you go deeper (more abstract features).
- Use **small kernels** (3×3) and stack them.
- Add **batch normalization** after every conv.
- End with **global average pooling** + small FC + softmax.

---

## Module 6: Numerical Templates (do every one!)

These are the **exact patterns** that appeared in past papers, with worked solutions. If you can do all eight, you can answer any numerical question on the MidSem.

### Template 1: Perceptron — One Epoch (every paper, Q1)

**Pattern**: Given initial weights and an activation rule, run one epoch (4 inputs for AND/OR, 2 for NOT, etc.) and write the final weights.

**Cookbook**:
1. Add x₀ = 1 for the bias.
2. For each example: compute z = w · x, apply step (or sign for bipolar), get ŷ.
3. Compute error e = t - ŷ.
4. Update: wᵢ ← wᵢ + η · e · xᵢ.
5. Move to next example using updated w.
6. After all examples, **verify** by re-checking each input matches its target.

**Bipolar variant** (some 2024 papers): inputs ∈ {-1, +1}, targets ∈ {-1, +1}, activation = sign(z). Same update rule. The error e now takes values {-2, 0, +2}.

### Template 2: Linear Regression GD — One Step (2024 Reg, 2025 Reg)

**Pattern**: 3–4 data points, initial w, η. Compute one or two GD steps.

**Cookbook**:
1. Form design matrix X (prepend column of 1s for bias). Form y.
2. Compute ŷ = Xw.
3. Compute error e = ŷ - y.
4. Compute loss J = 1/N Σ e².
5. Compute gradient ∇ J = 2/N Xᵀ e.
6. Update w ← w - η ∇ J.
7. Repeat for next step using updated w.

### Template 3: Logistic Regression SGD Step (2025 Mkp, Companion 4)

**Pattern**: One example, sigmoid activation, BCE loss, η.

**Cookbook**:
1. Compute z = wᵀ x.
2. Compute ŷ = σ(z) = 1/(1+e^(-z)).
3. Compute loss L = -[y log ŷ + (1-y)log(1-ŷ)].
4. Gradient: ∇ L = (ŷ - y) x.
5. Update: w ← w - η ∇ L.

✏️ **Worked**: x = [1, 1, 2, 3]ᵀ (bias prepended), y = 1, w = [0,0,0,0], η = 0.1.
- z = 0, ŷ = 0.5.
- L = -log 0.5 ≈ 0.693.
- ∇ L = (0.5 - 1) · [1,1,2,3] = [-0.5, -0.5, -1.0, -1.5].
- w_{ new} = [0.05, 0.05, 0.10, 0.15].
- Re-evaluate: z_{ new} = 0.75, ŷ_{ new} = σ(0.75) ≈ 0.679. Better!

### Template 4: Softmax + CCE — Compute Loss (Dec 2025 Q4a)

**Pattern**: 3-class logits, true label, compute softmax probabilities + loss + predicted class.

**Cookbook** (from §2.8 above):
1. Exponentiate each logit.
2. Sum the exponentials.
3. Divide each exponential by the sum → probabilities (must sum to 1).
4. Loss = -log ŷ_c where c is the true class.
5. Predicted class = argmax of probabilities.
6. State whether prediction is correct.

### Template 5: ⭐ Forward + Backward on Tiny DFNN (every Reg Paper)

**Pattern**: 2-2-1 or 2-3-2 network, given weights and inputs, run forward then backward.

**Cookbook** (mirror §3.5 above):
1. **Forward** layer by layer: z⁽ℓ⁾ = W⁽ℓ⁾ h^(ℓ-1) + b⁽ℓ⁾, h⁽ℓ⁾ = f(z⁽ℓ⁾).
2. **Compute loss** at the output.
3. **Backward**:
   - Output: δ^(L) = ŷ - y.
   - Hidden: δ⁽ℓ⁾ = (W^(ℓ+1))ᵀ δ^(ℓ+1) ⊙ f'⁽ℓ⁾(z⁽ℓ⁾).
4. **Gradients**: ∂ L/∂ W⁽ℓ⁾ = δ⁽ℓ⁾ (h^(ℓ-1))ᵀ, ∂ L/∂ b⁽ℓ⁾ = δ⁽ℓ⁾.
5. **Update** all weights with η.
6. **State** any dead ReLU neurons explicitly — examiners look for this insight.

### Template 6: Parameter Counting (every paper)

**Pattern**: Given an architecture, count total trainable parameters.

**Cookbook**:
- For each layer:
  - **FC**: n_{ in} · n_{ out} + n_{ out}.
  - **Conv**: K · K · C_{ in} · C_{ out} + C_{ out}.
  - **Pool / ReLU / Flatten**: 0.
  - **Batch Norm**: 2 · C.
- Sum across all layers.
- Always **show your work in a table** — partial credit per layer.

### Template 7: CNN Output Size (every CNN paper)

**Pattern**: Given image size, kernel, padding, stride — find output dimensions.

**Cookbook**: O = ⌊ (I + 2P - K)/S ⌋ + 1 for height and width independently. Number of output channels = number of filters.

✏️ **Quick mental practice — fill these in your head before the exam**:

| In | K | P | S | Out | In | K | P | S | Out |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3 | 1 | 1 | **64** | 64 | 3 | 0 | 1 | **62** |
| 64 | 3 | 1 | 2 | **32** | 64 | 5 | 2 | 1 | **64** |
| 28 | 5 | 0 | 1 | **24** | 28 | 5 | 2 | 2 | **14** |
| 224 | 7 | 3 | 2 | **112** | 224 | 11 | 0 | 4 | **54** |

### Template 8: Counting parameters of a CNN (high yield!)

**Pattern**: Whole CNN architecture given — count Conv+FC parameters.

**Cookbook**: build a table column-by-column:

| Layer | Spec | Output H × W × C | Params |
|---|---|---|---|

For each row:
- Compute output size with the formula in Template 7.
- Compute parameters with the formula in Template 6.
- Sum at the end.

---

## Module 7: Rapid-Fire Revision Cheat Sheet

### One-line answers to common 5-mark concept questions

**Why is XOR not linearly separable?**
> The two class-1 points sit at opposite corners of the unit square; no line can separate them from the two class-0 points at the other corners.

**Why use ReLU instead of sigmoid in hidden layers?**
> Sigmoid's max derivative is 0.25 → multiplying through L layers shrinks gradients to ~0 (vanishing gradient). ReLU's derivative is 1 for positive inputs, so gradients flow without shrinkage.

**What is the dying ReLU problem?**
> If a neuron's pre-activation is always negative, ReLU outputs 0 and gradient is 0 → its weights never update → neuron is permanently dead. Fix: Leaky ReLU, smaller LR, better init.

**Why is depth more efficient than width?**
> Deep networks build hierarchical features (edges → parts → objects). Achieving the same approximation with a shallow network requires exponentially more neurons. Hierarchical features also generalise better.

**Why doesn't MSE work for classification?**
> 1) MSE outputs are unbounded but probabilities live in [0,1]. 2) MSE doesn't punish confident-wrong predictions enough — the loss landscape is nearly flat. Cross-entropy fixes both.

**Why do we use cross-entropy instead of MSE for classification?**
> Cross-entropy is the negative log-likelihood under a Bernoulli/categorical assumption — the "right" probabilistic loss. Its gradient pairs cleanly with sigmoid/softmax (yielding ŷ-y). Confident wrong predictions incur near-infinite loss, providing a strong learning signal.

**Why is parameter sharing important in CNNs?**
> The same edge/texture appears at any spatial location. Sharing one kernel across all positions exploits this translation-invariance, drastically cutting parameter count and improving generalisation.

**Why use small (3×3) kernels stacked rather than large (7×7) kernels?**
> Two 3×3 convs have the same receptive field as one 5×5 (=5), but use 2 · 9 = 18 vs 25 parameters AND introduce one extra non-linearity, making them more expressive.

**What does Batch Normalization do?**
> Normalises each layer's mini-batch activations to mean 0, variance 1, then re-scales with learnable γ, β. Reduces internal covariate shift, allows higher LRs, acts as a mild regulariser.

**What does Dropout do?**
> Randomly zeros a fraction p of neurons during training; forces the network to learn redundant representations; equivalent to averaging over many sub-networks (ensembling).

**What is early stopping?**
> Track validation loss every epoch. When it stops improving for k epochs (the patience), halt training and restore best weights. Free regulariser.

**What is the difference between L1 and L2 regularization?**
> L2 (λ ||w||²) shrinks weights smoothly toward 0 — small but non-zero. L1 (λ ||w||) drives many weights *exactly* to 0, producing a sparse, feature-selecting model.

**What is overfitting and how to detect it?**
> Train loss low, val loss high. Detect by tracking both losses over epochs — divergence indicates overfitting.

**When does GD converge?**
> For a strongly convex objective with Lipschitz-continuous gradient with constant L, GD converges if η < 2/L. For non-convex objectives, GD finds a local minimum (or saddle) but not necessarily the global minimum.

**Why does the gradient of cross-entropy + sigmoid simplify to ŷ - y?**
> The ŷ(1-ŷ) factor in the sigmoid derivative cancels the ŷ(1-ŷ) factor in the BCE derivative, leaving just ŷ - y. Same cancellation happens for softmax + CCE.

### Common code-bug-spotting patterns (Q5 in 2024 papers)

| Bug | Symptom | Fix |
|---|---|---|
| Used sigmoid in last layer of multi-class problem | Outputs don't sum to 1 | Use softmax |
| MSE loss with sigmoid | Slow learning, vanishing gradients | Use BCE |
| Forgot to zero gradients (`optimizer.zero_grad()` in PyTorch) | Gradients accumulate; loss explodes | Add `optimizer.zero_grad()` before each `loss.backward()` |
| Updated weights with `+= η * grad` | Loss increases | Should be `-= η * grad` (gradient descent subtracts) |
| Applied softmax + CrossEntropyLoss in PyTorch | Double-softmax, wrong gradients | PyTorch's CrossEntropyLoss already includes softmax — pass logits |
| Activation before BatchNorm | Less stable training | BN should come *before* activation typically |
| Dropout enabled at inference time | Random predictions | Use `model.eval()` mode at test |
| Learning rate too high | Loss diverges to NaN | Decrease η; add gradient clipping |
| Used bipolar inputs but binary targets | Constant high error | Match input/target encoding |
| Sigmoid in deep hidden layers | Loss plateaus quickly | Replace with ReLU |
| Initialised weights to zero | Network never learns | Use Xavier/He init |

### High-yield mnemonics

- **"BCE BSEE"** — **B**inary cross-entropy goes with **S**igmoid (binary). Symmetric: **C**ategorical cross-entropy goes with **S**oftmax (multi-class).
- **"4-32-64"** — Adam's hyperparameter sweet spot: beta₁ = 0.9, beta₂ = 0.999, ε = 10⁻⁸. (Not "4-32-64", just memorise these three numbers.)
- **"PMSI"** — order of layer types in modern CNN: **P**ad → **C**onv (yes, Pad+Conv combined is one block) → BN → **R**eLU → **P**ool. (Or just CBNRP.)
- **"hat minus y"** — the gradient identity ŷ - y for softmax+CCE and sigmoid+BCE.
- **"OPK to formula"** — for output size: ⌊ (I + 2P - K)/S ⌋ + 1.

### Activation derivatives at a glance

| f(z) | f'(z) | Use in hidden? |
|---|---|---|
| Identity z | 1 | Output only (regression) |
| Sigmoid σ | σ(1-σ) | NO — vanishing gradients |
| Tanh | 1 - tanh² | Sometimes (RNNs) |
| ReLU | 1 if z>0 else 0 | YES — default |
| Leaky ReLU | 1 if z>0 else α | YES — replaces ReLU when dying |

---

## Module 8: Final Hour Checklist

### 30 minutes before the exam
- [ ] **Read** Module 0 + Module 7 (cheat sheet) one more time.
- [ ] **Solve** Templates 1, 5, 6, 7 from Module 6 in your head.
- [ ] **Memorise** the boxed formulas:
  - Perceptron update: w ← w + η(t-ŷ)x
  - Sigmoid derivative: σ'(z) = σ(z)(1-σ(z)), max = 0.25
  - ∂ BCE/∂ z = ŷ - y
  - Conv output size: O = ⌊ (I + 2P - K)/S ⌋ + 1
  - FC params: n_{ in} · n_{ out} + n_{ out}
  - Conv params: K · K · C_{ in} · C_{ out} + C_{ out}

### Inside the exam (read carefully!)
1. **First 10 minutes**: skim all 6 questions, note which are numerical (do those first), which are conceptual (last).
2. **Numerical questions** (Q1, Q3, Q4 typically): write the formula → substitute → compute → box the answer. Show *every* arithmetic step — the rubric awards step-by-step marks.
3. **Conceptual questions** (Q2, Q5, Q6 typically): write 30–50 words. Use the canned answers from Module 7. Mention key terms ("vanishing gradient", "linearly separable", "translation invariance") — keywords trigger marks.
4. **Code interpretation (Q5)**: state what the code does in 1 line, then list specific bugs with line numbers. Use the bug table in Module 7.
5. **Time discipline**: 2 hours / 6 questions = 20 min each. If stuck, leave space and move on.
6. **State assumptions** at the top of every answer to handle ambiguity.

### What to bring (admission-card check)
- [ ] Admit card / hall ticket
- [ ] ID proof
- [ ] Working pen + spare
- [ ] Pencil + sharpener (for diagrams)
- [ ] Eraser
- [ ] Ruler (for computational graphs and CNN architecture diagrams)
- [ ] Watch (no phones)

### Mindset
- The faculty explicitly said the paper covers **Sessions 1–8**. Anything beyond CNN Part 1 (RNN, LSTM, attention, transformers) is **EndSem material** — skip for now.
- The paper has **6 questions × 5 marks = 30 marks**. Each question is worth **3.3% of your total grade**. Manage time accordingly.
- "If you don't understand the question — make a reasonable assumption and proceed." This is faculty-stated.

---

### One-line summary of the entire course (Sessions 1–8)

> Deep learning is many stacked layers of (linear transform → non-linear activation), trained by gradient descent on a probabilistic loss. A perceptron is one layer with a step function — limited to linearly separable problems (fails on XOR). MLPs add hidden layers with non-linear activations to solve any continuous function (UAT). Activations are picked to match the loss: identity+MSE (regression), sigmoid+BCE (binary), softmax+CCE (multi-class). Backprop computes all gradients in one backward pass via the chain rule. ReLU + He init + Adam + dropout + BN are the standard recipe to make deep training stable. CNNs replace dense layers with local + shared kernels for images: convolution + pooling, with output size ⌊ (I+2P-K)/S ⌋ + 1 and parameters K² C_{ in} C_{ out} + C_{ out} per filter.

**Now go get it. You've got this.**
