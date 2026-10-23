# Statistical Learning for Automation Systems – Complete Question Bank

Prof. Simone Formentin – A.Y. 2026-2027. Lectures 1–10 and past exam questions (Nov 2022 – Feb 2026).

**How to use this file**
- **Part I** has one section per lecture: a *key formulas* block, then numbered questions (L1.Q1, L2.Q3, …) with full exam-style answers.
- Under every question, the **Source** line says where the topic is in the slides (lecture and slide number). "(extra)" means it is not written on the slides but is useful at the exam. "(Derivation)" means you should practise it on paper.
- **Part II** collects the questions that appeared in past exams, grouped by topic, with complete solutions and the slides where the theory is.
- The **Appendix** lists exam topics not covered yet in the lectures.
- The **Blackboard sheet** (right below) has one board per lecture with the essentials to copy on a blackboard and rewrite from memory. In Part I, **★** marks the same must-know formulas (in the key formulas) and the must-know questions (at the end of the title): study those first.
- Study by covering the answer and replying out loud and in full. For derivations, write them on paper before reading.
- When a new lecture arrives, copy the template at the bottom and keep the same numbering and Source style.

**Slide numbering.** Lecture 4 numbers refer to the complete PDF (28 slides), the version shown in the app.

**Lectures covered:** 1 – The feasibility of learning; 2 – Linear regression; 3 – Linear classification; 4 – Logistic regression and numerical optimization; 5 – The bias-variance trade-off; 6 – Model selection (regularization and cross-validation); 7 – Nonlinear modeling (feature transforms and neural networks); 8 – Nonparametric classification (k-NN and classifier quality); 9 – Clustering (k-means and hierarchical clustering); 10 – Data pre-processing (centering, normalization, whitening, cleaning, PCA); 11 – Stochastic processes (mean, covariance, stationarity, white noise, spectral density); 12 – Learning with stochastic processes (sample estimators, periodogram, averaging, trend and seasonality).

---

# Blackboard sheet – the essentials to write at home

One board per lecture: copy each board on the blackboard, then cover it and rewrite it from memory. **(D)** = derivation you must be able to redo on paper. The same items are marked with **★** in Part I (key formulas and must-know questions), so you know which answers to study first.

## Board 0 – The whole course on one board

```text
f (unknown) → 𝒟_N = {(x⁽ⁱ⁾, y⁽ⁱ⁾)} → pre-processing z = Φ(x) → algorithm 𝒜 searches ℋ → ĥ ≈ f
GOAL: E_out(ĥ) small  =  fitting (E_in small)  +  generalization (E_out ≈ E_in)
𝔼_𝒟[E_out] = bias + variance (+ σ²)      complexity knobs (m, λ, k, hidden units) → CROSS-VALIDATION
```

| Method | Output | Error minimized | How |
|---|---|---|---|
| Linear regression (L2) | s = wᵀx (a number) | (y − s)² | formula ŵ = (XᵀX)⁻¹XᵀY |
| Perceptron (L3) | sign(s) (a class) | ⟦y ≠ sign(s)⟧ | PLA: w ← w + y·x |
| Logistic regression (L4) | h(s) (a probability) | ln(1 + e^(−y·s)) | GD / SGD / Newton |
| Ridge (L6) | s = wᵀx | E_in + (λ/N)‖w‖² | (XᵀX + λI)⁻¹XᵀY |
| Neural network (L7) | tanh layers → output | (y − h(x))² | SGD + backpropagation |
| k-NN (L8) | sign(Σ_k y) | none (no training) | store the data, vote |
| k-means (L9) | k clusters | Σ_j Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖² | Lloyd |
| PCA (L10) | Ẑ = XV_k | max vᵀAv | SVD X = UΓVᵀ |

## Board 1 – Feasibility of learning

```text
E_in = (1/N) Σ ⟦h(x⁽ⁱ⁾) ≠ y⁽ⁱ⁾⟧  (measurable)        E_out = ℙ[h(x) ≠ f(x)]  (not measurable)
Hoeffding, one h:   ℙ[|E_in − E_out| > ε] ≤ 2e^(−2ε²N)
M hypotheses:       ℙ[|E_in − E_out| > ε] ≤ 2M·e^(−2ε²N)      (union bound; M = ∞ for lines ⇒ useless)
PAC (D):  2e^(−2ε²N) ≤ δ  ⇒  N ≥ ln(2/δ)/(2ε²)                 ε = δ = 0.05 ⇒ N ≥ 738
• deterministic view: impossible   • probabilistic view: probably approximately correct
• big ℋ: E_in ↓ but M ↑ (weaker guarantee)  ⇒ fitting vs generalization
```

## Board 2 – Linear regression

```text
ŷ = wᵀx  (x₀ = 1, w₀ = b)          E_in(w) = (1/N)‖Y − Xw‖²
(D)  ∇E_in = (2/N)(XᵀXw − XᵀY) = 0  ⇒  XᵀXŵ = XᵀY  ⇒  ŵ = (XᵀX)⁻¹XᵀY
Hessian (2/N)XᵀX ⪰ 0  ⇒ convex bowl;  ≻ 0 ⇔ X full column rank (N ≥ m+1)
XᵀX singular: data not informative (constant input) or ill-posed model (x₂ = 2x₁, N < m+1)
noise y = Xθ + v  ⇒  θ̂ = θ + (XᵀX)⁻¹Xᵀv
E_out = E_in + O(m/N)
```

## Board 3 – Perceptron

```text
ŷ = sign(wᵀx),  y ∈ {−1, +1},  boundary wᵀx = 0 (hyperplane, w ⊥ boundary)
E_in = (1/N) Σ ⟦y⁽ⁱ⁾ ≠ sign(wᵀx⁽ⁱ⁾)⟧  → staircase, no gradient ⇒ no OLS-like formula
PLA: take a misclassified point, w(k+1) = w(k) + y·x
(D)  y·s(k+1) = y·s(k) + ‖x‖²   (the score moves toward the right sign)
separable ⇒ converges in a finite number of steps;  not separable ⇒ never stops
E_out = E_in + O(√((m/N)·ln N))
```

## Board 4 – Logistic regression and optimization

```text
h(s) = 1/(1 + e^(−s)),  (D) 1 − h(s) = h(−s)  ⇒  P(y|x) = h(y·wᵀx)
(D)  max Π h(y⁽ⁱ⁾wᵀx⁽ⁱ⁾)  ⇔  min E_in = (1/N) Σ ln(1 + e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾))   (cross-entropy)
(D)  ∇E_in = −(1/N) Σ h(−y⁽ⁱ⁾s⁽ⁱ⁾)·y⁽ⁱ⁾x⁽ⁱ⁾   → convex, no closed form
GD (D, Taylor): w(k+1) = w(k) − η∇E_in(w(k))     η small: slow; η large: oscillates
SGD: w(k+1) = w(k) − η∇e_i(w(k)), i random  → cheap, noisy, zig-zag near the minimum
Newton: w(k+1) = w(k) − [∇²E_in]⁻¹∇E_in      quasi-Newton: M ≈ Hessian
Stop: ‖∇E_in‖ < 10⁻⁶  or  max iterations  or  E_in below threshold / not improving
Decide: ŷ = +1 if h(wᵀx) ≥ threshold (0.5 ⇔ wᵀx ≥ 0)
```

## Board 5 – Bias-variance trade-off

```text
average model:  h̄(x) = 𝔼_𝒟[h^𝒟(x)]
(D)  𝔼_𝒟[E_out] = 𝔼ₓ[(f(x) − h̄(x))²] + 𝔼ₓ𝔼_𝒟[(h^𝒟(x) − h̄(x))²]   (+ σ² with noise)
                        bias                       variance
complex ℋ: bias ↓ variance ↑        simple ℋ: bias ↑ variance ↓
learning curves: N ↑ ⇒ E_in ↑, E_out ↓, both → bias (never 0 if f ∉ ℋ)
choose the complexity according to the DATA available, not the target
```

## Board 6 – Model selection

```text
min ‖w‖₀ (NP-hard) → relax: ‖w‖₂² ≤ C  ⇔  E_aug = E_in + (λ/N)‖w‖₂²
(D)  ridge: ŵ_reg = (XᵀX + λI)⁻¹XᵀY   (invertible for every λ > 0)
λ = 0 → OLS (overfitting);  λ → ∞ → w → 0 (underfitting)
validation: train on N − k, E_val on k:  𝔼[E_val] = E_out(ŵ⁻),  Var = O(1/k),  k ≈ N/5
LOO: E_cv = (1/N) Σ_j e_j  (N trainings, unbiased);  k-fold: K trainings
three-way split: training 70% (w) | validation 15% (λ, m) | test 15% (E_out, used once)
```

## Board 7 – Nonlinear models and neural networks

```text
z = Φ(x), ŷ = sign(wᵀΦ(x))  → linear in w, nonlinear in x
circle: Φ(x) = [r², x₁², x₂²], w = [1, −1, −1]
choose Φ BEFORE looking at the data (otherwise data snooping: the guarantee is lost)
OR = sign(x₁ + x₂ + 1)   AND = sign(x₁ + x₂ − 1)   XOR needs 2 layers
neuron h(x) = θ(wᵀx), θ = tanh (smooth); sign has no useful gradient
training: SGD + backpropagation on e = (y − h(x))²
linear hidden activations ⇒ the whole network is linear
Cybenko: 1 hidden layer approximates any continuous f;  deep: reuse, fewer neurons
```

## Board 8 – k-NN and classifier quality

```text
nonparametric: no h(x, w), predict directly from the data
k-NN: sort d(x*, x⁽ⁱ⁾),  ŷ* = sign(Σ_{i=1}^{k} y(x̄⁽ⁱ⁾))    (k odd)
d = ‖x − x*‖, weighted (x − x*)ᵀQ(x − x*), cosine  → SCALE the features
k = 1: E_in = 0, high variance (Voronoi);  k = N: majority class, high bias
consistency: N → ∞, k → ∞, k/N → 0;  k = ⌊√N⌋ or cross-validation
cost per prediction: O(N·m + N·log k)
confusion [a b; c d]: precision a/(a+b), recall a/(a+c), F1 = 2PR/(P+R)
accuracy (a+d)/N is misleading with unbalanced classes
```

## Board 9 – Clustering

```text
no labels: split into S₁ … S_k,  μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾
E_in = Σ_j Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖²   (exact minimum NP-hard)
Lloyd: (1) init μ  (2) each point → nearest μ_j  (3) μ_j = mean of S_j  (4) repeat
(D)  no step can increase E_in ⇒ stops in finitely many steps at a LOCAL minimum
init: farthest point (random point, then the point farthest from those chosen)
k: knee of E_in(k)   (E_in(k) always ↓, = 0 for k = N)
hierarchical: merge the closest clusters N → 1; dendrogram = every k at once
```

## Board 10 – Data pre-processing and PCA

```text
z = Φ(x) to STANDARDIZE; x̄, σ, A, V from the TRAINING data only
centering z = x − x̄    normalization z_j = x_j/σ_j    whitening z = A^(−1/2)x
A = (1/N) Σ x⁽ⁱ⁾x⁽ⁱ⁾ᵀ;  (D) A^(−1/2)·A·A^(−1/2) = I ⇒ uncorrelated, unit variance
outliers pull the least-squares line (squared error) → clean the data
PCA: max_{‖v‖=1} Var[vᵀx] = vᵀAv
(D)  X = UΓVᵀ ⇒ A = V(Γ²/N)Vᵀ ⇒ v₁ = 1st column of V, variance γ₁²/N
Ẑ = X·V_k (N×k);  kept Σ_{i≤k} γ_i² / Σ γ_i²;  lost Σ_{i>k} γ_i²
limits: unsupervised (ignores y), linear, needs scaling, sensitive to outliers
```

## Board 11 – Stochastic processes

```text
SP x(s,t): fix s → realization (deterministic signal);  fix t → random variable
wide sense: m(t) = E_s[x(s,t)],  γ(t1,t2) = E[(x(t1) − m(t1))(x(t2) − m(t2))],  γ(t,t) = variance
STATIONARY (SSP): m(t) = m ∀t  and  γ(t1,t2) = γ(τ), τ = t2 − t1
γ(τ): γ(0) ≥ 0,  |γ(τ)| ≤ γ(0),  γ(τ) = γ(−τ)
(D) x = α, α ~ N(1,3): m = 1, γ(τ) = 3 ⇒ SSP     x = αt − t: m = 0, γ = 3·t1·t2 ⇒ NOT SSP
WN(μ, λ²): mean μ, variance λ², γ(τ) = 0 ∀τ ≠ 0
Γ(ω) = Σ_{τ=−∞}^{+∞} γ(τ)e^(−jωτ):  real, ≥ 0, even, periodic 2π
(D) WN: Γ(ω) = γ(0) = λ² (flat: all frequencies equally present)
γ(τ) = (1/2π)∫_{−π}^{π} Γ(ω)e^(jωτ)dω  ⇒  γ(0) = variance = (1/2π)·area under Γ on [−π, π]
SSP described by (m, γ(τ)) in time  or  (m, Γ(ω)) in frequency
```
## Board 12 – Learning with stochastic processes

```text
data = ONE realization 𝒟_N = {x(s̄,1), …, x(s̄,N)};  Q* = 𝔼[Q̂_N]
E_out(Q̂_N) = 𝔼[(Q − Q̂_N)²] = (Q − Q*)² [bias] + 𝔼[(Q̂_N − Q*)²] [variance]
correct: 𝔼[Q̂_N] = Q (zero bias)     consistent: 𝔼[(Q* − Q̂_N)²] → 0 as N → ∞
m̂_N = (1/N)Σ x(t):  (D) correct;  consistent if γ(τ) → 0 as |τ| → ∞   (x = α: NOT)
γ̂_N(τ) = 1/(N−|τ|) Σ_{t=1}^{N−|τ|} x(t)x(t+|τ|):  correct, consistent if γ → 0;  use |τ| ≪ N
Γ̂_N(ω) = Σ_{|τ|<N} γ̂_N(τ)e^(−jωτ):  asymptotically correct, NOT consistent
periodogram (1/N weights): Γ̂_N(ω) = (1/N)|Σ_t x(t)e^(−jωt)|²  → FFT
averaging (Bartlett): K pieces of N/K, average the K periodograms ⇒ variance ≈ 1/K
nonstationary: x = x_SSP + v(t): estimate v̂ → x̂_SSP = x − v̂ → m̂, γ̂, Γ̂ → m(t) = m̂ + v̂(t)
trend v = kt + q: OLS [k̂; q̂] = [Σt² Σt; Σt N]⁻¹[Σ t·x(t); Σ x(t)]
season v(t) = v(t+T), T known: v̂(t) = (1/M)Σ_{k=0}^{M−1} x(t + kT), t = 1…T
```
---

# Part I – Questions from the slides

## Lecture 1 – The feasibility of learning

### Key formulas
- Setup: input x ∈ 𝒳 ⊆ ℝⁿ, output y ∈ 𝒴, unknown target f: 𝒳 → 𝒴, dataset 𝒟_N = {(x⁽¹⁾,y⁽¹⁾),…,(x⁽ᴺ⁾,y⁽ᴺ⁾)}, hypothesis set ℋ, algorithm 𝒜, final model f̂ ≈ f
- ★ Hoeffding: ℙ[|p̂ − p| > ε] ≤ 2e^(−2ε²N)
- In-sample error: E_in(f̂) = (1/N) Σᵢ ⟦f̂(x⁽ⁱ⁾) ≠ f(x⁽ⁱ⁾)⟧ (measurable)
- Out-of-sample error: E_out(f̂) = ℙ[f̂(x) ≠ f(x)] (not measurable)
- ★ Single fixed hypothesis: ℙ[|E_out − E_in| > ε] ≤ 2e^(−2ε²N)
- ★ M hypotheses: ℙ[|E_out − E_in| > ε] ≤ 2M·e^(−2ε²N)
- ★ Sample size (exam): N ≥ (1/(2ε²))·ln(2/δ)

### L1.Q1 – When is learning from data meaningful?
*Source: Lecture 1, slide 4*

Only if three conditions hold: (1) a pattern exists; (2) the pattern cannot be derived mathematically, e.g. from physics; (3) we have data. With no pattern there is nothing to learn. If physics gives the model, just use the physical model. With no data there is nothing to learn from.

### L1.Q2 – What is the target function f?
*Source: Lecture 1, slide 2*

The unknown "ideal law" f: 𝒳 → 𝒴 that maps inputs to the correct outputs (e.g. email features → spam/ham). It generates the data but is never known. Do not confuse it with f̂ (the learned model), ℋ (the set of candidate models) or 𝒟_N (the dataset).

### L1.Q3 – What is the objective of supervised learning?
*Source: Lecture 1, slide 3*

To find a model f̂: 𝒳 → 𝒴 **inside a predefined hypothesis set ℋ** such that f̂ ≈ f, i.e. y ≈ f̂(x). The search does not happen over all possible functions, only inside ℋ, which is a design choice (e.g. the set of LTI systems for a DC motor with x = voltage, y = position). Pipeline: f → 𝒟_N → learning algorithm 𝒜 (which also receives ℋ) → f̂.

### L1.Q4 – What does the marble example show? Why is learning infeasible from a deterministic point of view?
*Source: Lecture 1, slides 5–7*

Six sequences of three marbles are labelled ±1. Two rules fit **all** the training data perfectly: "y = +1 if the first marble is black" and "y = −1 if more than one marble is white". On the new point ● ○ ○ they give opposite answers. Since infinitely many functions agree with finite data and disagree outside it, data alone cannot determine f outside the sample. This is Hume's problem of induction (1748). The same happens with a continuous curve through a few points (slide 6).

### L1.Q5 – State Hoeffding's inequality and explain its terms. ★
*Source: Lecture 1, slides 8 and 10*

ℙ[|p̂ − p| > ε] ≤ 2e^(−2ε²N). In the bin experiment, p = probability of picking a blue marble (unknown), p̂ = fraction of blue marbles in a sample of N independent draws (known). The probability that p̂ is far from p by more than ε decays **exponentially in N**. ε is the accuracy (tolerance); the right-hand side sets the confidence.

### L1.Q6 – What is the trade-off in Hoeffding? What happens if ε → 0?
*Source: Lecture 1, slide 10*

Accuracy (small ε) vs confidence (small bound). A smaller ε makes the bound looser unless N grows. If ε → 0 the bound tends to 2, which is useless since any probability is ≤ 1. We can never guarantee p̂ = p exactly, only approximately.

### L1.Q7 – What does PAC mean?
*Source: Lecture 1, slides 9–10*

Probably Approximately Correct. "Approximately" refers to ε (p̂ within ε of p); "probably" refers to the probability bound (this holds with high probability). Deterministic view: learning is impossible. Probabilistic view: learning is possible in the PAC sense.

### L1.Q8 – Explain the analogy between the bin and the learning problem.
*Source: Lecture 1, slides 11–12*

Bin: unknown p, data = sample of marbles, estimator p̂. Learning: unknown f, data = 𝒟_N, estimator f̂. Each input point x is a marble: red ⇔ f̂(x) ≠ f(x) (model wrong), blue ⇔ f̂(x) = f(x) (model right). Drawing marbles at random = drawing the x's of the dataset at random. So E_in plays the role of p̂ (fraction observed in the sample) and E_out plays the role of p (fraction in the whole bin). Note: strictly, red = error, so E_in corresponds to the fraction of red marbles (1 − p̂); Hoeffding is symmetric so the result is the same.

### L1.Q9 – Define E_in and E_out. Why is E_out not measurable? ★
*Source: Lecture 1, slide 12*

E_in(f̂) = (1/N) Σᵢ ⟦f̂(x⁽ⁱ⁾) ≠ f(x⁽ⁱ⁾)⟧: fraction of errors on the training data; ⟦·⟧ = 1 if the statement is true, 0 otherwise. It is measurable because the labels are known. E_out(f̂) = ℙ[f̂(x) ≠ f(x)]: probability of error on any input, including unseen ones. It requires the unknown input distribution, so it cannot be measured.

### L1.Q10 – State Hoeffding applied to learning. What does the bound (not) depend on? ★
*Source: Lecture 1, slide 13*

ℙ[|E_out(f̂) − E_in(f̂)| > ε] ≤ 2e^(−2ε²N). It is a probabilistic verification of the generalization of a hypothesis. The bound depends only on N and ε: it does **not** depend on E_out or on f, so we need no knowledge of the unknown function. Implicit assumptions: data drawn i.i.d. from the same distribution as future inputs, and f̂ **fixed before looking at the data**.

### L1.Q11 – Why does the factor M appear in 2M·e^(−2ε²N)? ★
*Source: Lecture 1, slide 14*

In learning, f̂ is not given: the algorithm **chooses** it from ℋ after seeing the data. This is like having M bins and choosing the one whose sample looks best; with many bins some look good by chance. The guarantee must then hold for all M hypotheses at once (union bound), which multiplies the bound by M. A richer ℋ (larger M) gives a weaker guarantee; more data (larger N) a stronger one.

### L1.Q12 – Why is the 2M·e^(−2ε²N) bound useless for linear models, and how is this solved?
*Source: Lecture 1, slide 14 (resolved in Lecture 2, slide 14 and Lecture 3, slide 9)*

For models with real-valued parameters (linear regression, perceptron), w can take infinitely many values, so M = ∞ and the bound is trivial. The solution is to measure complexity by the number of parameters m instead of M: E_out = E_in + O(m/N) for linear regression, E_out = E_in + O(√((m/N)·ln N)) for the perceptron (VC-dimension idea).

### L1.Q13 – What are the two objectives of learning? Why do they conflict? ★
*Source: Lecture 1, slide 15*

(1) Low E_in(f̂): **fitting** (the model explains the data). (2) Small |E_in(f̂) − E_out(f̂)|: **generalization** (performance carries over to new data). A bigger ℋ makes fitting easier but the generalization bound looser (larger M or m). This is the seed of the complexity trade-off (overfitting) that runs through the whole course. Conclusion of the lecture: learning is feasible, but not easy.

### L1.Q14 – (Exam) Derive the number of samples needed for a PAC guarantee. ★
*Source: Lecture 1, slides 10 and 13; exam 21 Jan 2026, Ex. 2d*

Impose 2e^(−2Nε²) ≤ δ ⇒ e^(−2Nε²) ≤ δ/2 ⇒ −2Nε² ≤ ln(δ/2) ⇒ **N ≥ (1/(2ε²))·ln(2/δ)**. This guarantees ℙ(|E_in − E_out| ≤ ε) ≥ 1 − δ: with probability at least 1 − δ (probably) E_out is within ε of E_in (approximately correct). Example: ε = δ = 0.05 ⇒ N ≥ 200·ln 40 ≈ 738.

### L1.Q15 – (Exam) Define E_out for a general bounded loss and explain fitting vs generalization.
*Source: Lecture 1, slides 12–13; exams 21 Jan 2026, Ex. 2a and 2c; 13 Feb 2025, Ex. 1.2*

With a loss 0 ≤ ℓ ≤ 1: E_in(h) = (1/N) Σ ℓ(h(xᵢ), yᵢ) (average loss on training data) and E_out(h) = 𝔼₍ₓ,ᵧ₎∼P[ℓ(h(x), y)] (expected loss on new data from the same distribution). Fitting = making E_in small; generalizing = making E_out small. A hypothesis can have small E_in but large E_out if it captures noise or accidental regularities of the dataset. To apply Hoeffding set Zᵢ = ℓ(h(xᵢ), yᵢ): then (1/N)ΣZᵢ = E_in(h) and 𝔼[Zᵢ] = E_out(h).

---

## Lecture 2 – Linear regression

### Key formulas
- Model: y = b + Σᵢ wᵢxᵢ; with x₀ = 1, w₀ = b ⇒ y = Σᵢ₌₀ᵐ wᵢxᵢ = wᵀx
- Ideal (not achievable): w_opt = argmin 𝔼[(y − wᵀx)²]
- ★ Proxy: E_in(w) = (1/N) Σⱼ (y⁽ʲ⁾ − wᵀx⁽ʲ⁾)² = (1/N)‖Y − Xw‖²
- Expansion: E_in(w) = (1/N)(YᵀY + wᵀXᵀXw − 2wᵀXᵀY)
- Gradient: ∇E_in(w) = (1/N)(2XᵀXw − 2XᵀY)
- ★ Normal equations: XᵀXŵ = XᵀY; OLS: ŵ = (XᵀX)⁻¹XᵀY
- ★ Hessian: (2/N)XᵀX ⪰ 0
- ★ Generalization: E_out = E_in + O(m/N)
- (Exam) With noise: θ̂ = θ + (XᵀX)⁻¹Xᵀv (ridge regression: Lecture 6)

### L2.Q1 – What are the assumptions of linear regression?
*Source: Lecture 2, slide 2*

ℋ is assumed to be given, and ℋ is the set of **linear** models. The output y is a real number, and the objective is y ≈ f̂(x).

### L2.Q2 – What is a linear model? Meaning of b and wᵢ. How is the bias absorbed?
*Source: Lecture 2, slide 3*

y = b + w₁x₁ + … + wₘxₘ. b is the bias (offset: the output when all inputs are 0); wᵢ are gains, the sensitivity of y to input xᵢ (if xᵢ grows by 1, y changes by wᵢ). Defining x₀ = 1 and w₀ = b gives the compact form y = Σᵢ₌₀ᵐ wᵢxᵢ = wᵀx, with w, x of size (m+1)×1.

### L2.Q3 – What does "linear" mean in a linear model?
*Source: Lecture 2, slide 3 (extra, not on the slide)*

Linear **in the parameters w**, not necessarily in x. E.g. y = w₀ + w₁x + w₂x² is still a linear model if we treat x₁ = x, x₂ = x² as features. This is why a perceptron with features φ(x) = [1, x₁², x₂²] can produce an elliptic boundary (exam question).

### L2.Q4 – Formulate the ideal learning problem. Why is it not achievable?
*Source: Lecture 2, slide 4*

w_opt = argmin_w E_out(w), with E_out(w) = 𝔼[(y − Σ wᵢxᵢ)²] (mean squared error between measured outputs and model outputs over all possible data). It is a parametric optimization problem, but E_out requires the expectation over the unknown distribution, so it cannot be computed.

### L2.Q5 – What do we minimize instead? Explain the graph of the simplest case.
*Source: Lecture 2, slide 5*

The proxy E_in(w) = (1/N) Σⱼ (y⁽ʲ⁾ − Σᵢ wᵢxᵢ⁽ʲ⁾)², the **sample average** of the squared error instead of the statistical average. This is achievable. Graph (y = w₀ + w₁x): the data points, a candidate line, and the vertical segments between each point and the line (residuals y⁽ʲ⁾ − ŷ⁽ʲ⁾). E_in is the average of these segments squared; changing w₀ (intercept) and w₁ (slope) moves the line. Distances are vertical because x is assumed exact and only y is noisy.

### L2.Q6 – Why the squared error? Are all residuals equal at the optimum?
*Source: Lecture 2, slide 5 (extra, not on the slide)*

Positive and negative errors do not cancel; large errors are penalized more; the cost is quadratic in w (smooth, differentiable) ⇒ closed-form solution. Residuals are **not** equal: the line is the best compromise minimizing the total Σ(residual)². With a bias term the residuals sum to zero (the line passes "through the middle"). Because of the square, far points (outliers) pull the line strongly.

### L2.Q7 – Write E_in in matrix form. Structure of X and Y. ★
*Source: Lecture 2, slides 7–8*

Y = [y⁽¹⁾ … y⁽ᴺ⁾]ᵀ is N×1. X is N×(m+1): each **row** is a sample x⁽ʲ⁾ᵀ, each **column** a feature, the first column all ones. Predictions Ŷ = Xw (N×1). Since ‖z‖² = Σ zᵢ², E_in(w) = (1/N) Σ (y⁽ʲ⁾ − ŷ⁽ʲ⁾)² = (1/N)‖Y − Xw‖².

### L2.Q8 – State the optimality conditions for ŵ.
*Source: Lecture 2, slides 9–10*

ψ(w) = E_in(w) is a scalar function of a vector. Necessary and sufficient conditions: (1) gradient zero, ∂ψ/∂w |_{ŵ} = 0; (2) Hessian positive definite, ∂²ψ/∂w² |_{ŵ} ≻ 0. The gradient is a vector of size m+1; the Hessian is a symmetric (m+1)×(m+1) matrix with entries ∂²E_in/∂wₖ∂wₗ.

### L2.Q9 – Define a positive definite matrix and its geometric meaning.
*Source: Lecture 2, slide 11*

M ≻ 0 if zᵀMz > 0 for all z ≠ 0. The quadratic function is then a bowl curving upward in every direction, with a unique minimum.

### L2.Q10 – (Derivation) Expand E_in(w).
*Source: Lecture 2, slide 11*

‖Y − Xw‖² = (Y − Xw)ᵀ(Y − Xw) = YᵀY − YᵀXw − wᵀXᵀY + wᵀXᵀXw. YᵀXw and wᵀXᵀY are **scalars** and transposes of each other, hence equal. So E_in(w) = (1/N)(YᵀY + wᵀXᵀXw − 2wᵀXᵀY). All three terms are 1×1.

### L2.Q11 – (Derivation) Compute the gradient and obtain the OLS estimator. ★
*Source: Lecture 2, slide 12*

Rules: ∂(wᵀAw)/∂w = 2Aw (A symmetric, and XᵀX is symmetric); ∂(wᵀb)/∂w = b; YᵀY is constant. ∇E_in(ŵ) = (1/N)(2XᵀXŵ − 2XᵀY) = 0 ⇒ **normal equations** XᵀXŵ = XᵀY. If XᵀX ≻ 0 it is invertible ⇒ **ŵ = (XᵀX)⁻¹XᵀY** (ordinary least squares). It is the only case where the optimal parameters are found analytically as an explicit function of the data. (XᵀX)⁻¹Xᵀ is the pseudo-inverse; in MATLAB use w = X\Y.

### L2.Q12 – Compute the Hessian. Why is it always positive semidefinite? ★
*Source: Lecture 2, slide 13*

∂²E_in/∂w² = (2/N)XᵀX. For any z: zᵀXᵀXz = ‖Xz‖² ≥ 0 ⇒ XᵀX ⪰ 0 always. So E_in is always convex (bowl or valley) and has no local minima that are not global. Both optimality conditions reduce to the same requirement: XᵀX invertible (positive definite).

### L2.Q13 – When is XᵀX singular? Give the two causes and examples. ★
*Source: Lecture 2, slide 13*

When ‖Xz‖² = 0 for some z ≠ 0, i.e. the columns of X are linearly dependent (rank X < m+1). (1) **Data not informative** – experimental identifiability issue: the model is fine but the experiment does not excite it (all x⁽ʲ⁾ identical, a constant feature; in system identification, no persistent excitation, e.g. constant voltage to a DC motor). (2) **Ill-posed problem** – structural identifiability issue: the model is redundant whatever the data (x₂ = 2x₁, so only a combination of w₁, w₂ is determined; or N < m+1). XᵀX ≻ 0 ⇔ X full column rank ⇒ need N ≥ m+1.

### L2.Q14 – Describe the graph of E_in(w₀, w₁) in the regular and singular case.
*Source: Lecture 2, slide 13*

XᵀX ≻ 0: a **bowl** with a single bottom ⇒ unique ŵ. XᵀX singular: a **valley (trough)**, flat along one direction ⇒ a whole line of minima with the same E_in; the data cannot choose between them.

### L2.Q15 – State and interpret the generalization theorem for linear regression. ★
*Source: Lecture 2, slide 14*

E_out(w) = E_in(w) + O(m/N), with m = dimension of the parameter vector, N = dataset size, O(·) = order of magnitude. The gap grows with complexity (m) and shrinks with data (N); if N ≫ m (rule of thumb ~10×), E_in is a reliable estimate of E_out. Adding features can only lower E_in (the bigger model contains the smaller) but increases the O(m/N) term ⇒ better fitting, worse generalization (overfitting). Compared with Lecture 1, what counts is m, not M = ∞. Precise classical version (linear target, noise variance σ²): 𝔼[E_in] ≈ σ²(1 − (m+1)/N), 𝔼[E_out] ≈ σ²(1 + (m+1)/N).

### L2.Q16 – (Exam) How can uniqueness of the LS solution be promoted during data collection?
*Source: Lecture 2, slide 13; exam 11 Feb 2026, Ex. 1b*

Ensure X has full column rank: make the inputs vary enough and **independently** of each other, and collect N ≥ m+1 samples. Counter-example: if machinery hours always grow exactly with the number of loading operations, those two columns are nearly collinear and their effects cannot be separated.

### L2.Q17 – (Exam) How does the noise v affect the LS estimate? ★
*Source: Lecture 2, slide 12 (extra); exam 11 Feb 2026, Ex. 1d*

Model y = Xθ + v. Noiseless (v = 0): θ̂₀ = (XᵀX)⁻¹XᵀXθ = θ (exact). With noise: θ̂ = (XᵀX)⁻¹Xᵀ(Xθ + v) = θ + (XᵀX)⁻¹Xᵀv. The error θ̂ − θ = (XᵀX)⁻¹Xᵀv depends on the noise realization and on the conditioning of XᵀX. If 𝔼[v] = 0 the estimate is unbiased (𝔼[θ̂] = θ). If v is white with variance σ², Var(θ̂) = σ²(XᵀX)⁻¹: an ill-conditioned XᵀX amplifies the noise; more data makes XᵀX grow and the variance shrink.

---

## Lecture 3 – Linear classification

### Key formulas
- ★ Perceptron: ŷ = sign(Σᵢ wᵢxᵢ + b) = sign(wᵀx), sign(z) = +1 if z ≥ 0, −1 if z < 0; score s = wᵀx
- Classes: y ∈ {−1, +1}; decision boundary: hyperplane wᵀx = 0
- E_out(w) = ℙ[y ≠ sign(wᵀx)] (not measurable); E_in(w) = (1/N) Σ ⟦y⁽ʲ⁾ ≠ ŷ⁽ʲ⁾(w)⟧ (measurable)
- ★ PLA update: w(k+1) = w(k) + y⁽ⁱ⁾(k)·x⁽ⁱ⁾(k), with (x⁽ⁱ⁾, y⁽ⁱ⁾) misclassified
- ★ Why it works: y·s(k+1) = y·s(k) + ‖x‖²
- ★ Generalization: E_out = E_in + O(√((m/N)·ln N))

### L3.Q1 – What is the objective of binary classification? Assumptions?
*Source: Lecture 3, slide 2*

y is a class, not a number. Find a separating surface between two classes (e.g. cats vs dogs from weight and height). Assumptions: ℋ given a priori; the separating surface is linear (a hyperplane: line in 2D, plane in 3D).

### L3.Q2 – Describe the perceptron model. ★
*Source: Lecture 3, slide 3*

Classes encoded as y = ±1. Boundary: Σ wᵢxᵢ + b = 0 (a hyperplane). Classifier ŷ = sign(Σ wᵢxᵢ + b), where s = Σ wᵢxᵢ + b is the **score**. Block diagram: s → [sign] → ŷ, a relay / ideal comparator. It is the same score as linear regression, but only its sign is used.

### L3.Q3 – What is the geometric meaning of w and of the score?
*Source: Lecture 3, slide 3 (extra, not on the slide)*

The vector [w₁ … wₘ] is **normal** to the hyperplane and points toward the +1 side. The score is proportional to the **signed distance** of x from the hyperplane (distance = s/‖w‖): s > 0 on the +1 side, s < 0 on the −1 side, |s| large ⇒ far from the boundary, confident classification.

### L3.Q4 – What does "best" mean for a perceptron? Define E_in and E_out.
*Source: Lecture 3, slide 4*

Best = minimizing the number of misclassified points. E_out(w) = ℙ[y ≠ sign(wᵀx)] (not measurable); E_in(w) = (1/N) Σ ⟦y⁽ʲ⁾ ≠ ŷ⁽ʲ⁾(w)⟧ (measurable). These are the same errors as Lecture 1 (0/1 error).

### L3.Q5 – Why can't the perceptron be trained like OLS (gradient = 0)?
*Source: Lecture 3, slide 4 (extra, not on the slide)*

The 0/1 error E_in is **piecewise constant** in w: it only jumps when the boundary crosses a data point and is flat elsewhere. The gradient is zero almost everywhere and undefined at the jumps, so there is no closed-form solution. We need an iterative method: PLA.

### L3.Q6 – Describe the Perceptron Learning Algorithm and its key assumption. ★
*Source: Lecture 3, slides 5–7*

PLA is an empirical (iterative) solution to ŵ = argmin E_in(w). Compact notation w₀ = b, x₀ = 1 ⇒ ŷ = sign(wᵀx). Start from an initial guess ŵ(0) (e.g. zeros). At each iteration pick **one misclassified point** and update w(k+1) = w(k) + y⁽ⁱ⁾(k)x⁽ⁱ⁾(k) (Δw = y·x). Repeat until no point is misclassified. Key assumption: the data are **linearly separable** – there exists w* with sign(w*ᵀx⁽ʲ⁾) = y⁽ʲ⁾ for every point of 𝒟_N. Graphically each update rotates and shifts the boundary; adding y·x rotates the normal vector toward (y = +1) or away from (y = −1) the misclassified point.

### L3.Q7 – (Derivation) Why does each PLA update move in the right direction? ★
*Source: Lecture 3, slide 8*

A point is misclassified ⇔ y ≠ sign(wᵀ(k)x). E.g. y = +1, ŷ = −1 ⇒ wᵀ(k)x < 0. Multiply the update by x: wᵀ(k+1)x = wᵀ(k)x + y·xᵀx, i.e. new score = old score + y‖x‖². Since ‖x‖² > 0 (≥ 1 because x₀ = 1), the score increases if y = +1 and decreases if y = −1. Compact form: **y·s(k+1) = y·s(k) + ‖x‖²**.

### L3.Q8 – State the PLA convergence theorem. ★
*Source: Lecture 3, slide 7*

If the data are linearly separable, PLA converges to perfect classification (E_in = 0) in a **finite** number of iterations (Novikoff). The number of iterations is bounded by (R/γ)², R = largest ‖x‖, γ = margin of the best separator: well-separated data ⇒ fast convergence.

### L3.Q9 – What are the common traps about PLA?
*Source: Lecture 3, slide 8 (extra)*

(1) A single update does not guarantee that the point becomes correct; it only improves its score (it may need several updates). (2) An update can **break** points that were correct. (3) E_in does **not** decrease at every step: it can go up and down (e.g. 2/3 → 1/3 → 2/3 → … → 0). Convergence comes from the theorem, not from step-by-step descent.

### L3.Q10 – What happens if the data are not linearly separable? ★
*Source: Lecture 3, slide 5*

PLA never stops: there is always at least one misclassified point, so it keeps updating forever. The convergence theorem requires separability.

### L3.Q11 – Does PLA solve the classification problem completely?
*Source: Lecture 3, slide 9*

No: it solves it only in terms of **fitting** (E_in = 0) and says nothing about E_out. PLA stops at the **first** separating line it finds, which may pass very close to some points (e.g. touching a point with score exactly 0), so slightly different new data can be misclassified.

### L3.Q12 – Discuss generalization for the perceptron.
*Source: Lecture 3, slide 9*

Hoeffding with M hypotheses gives 2M·e^(−2ε²N), but for the perceptron M → ∞ (w ∈ ℝ^(m+1)), so it is useless. Generalization theorem: E_out(w) = E_in(w) + O(√((m/N)·ln N)). As in regression, what matters is m relative to N (effective complexity m+1, VC idea). Compared with regression's O(m/N), the gap shrinks more slowly (square root and log) ⇒ classification needs more data for the same guarantee.

### L3.Q13 – (Exam) Why can't a perceptron represent "STOP inside an ellipse, GO outside"? How do features fix it?
*Source: Exam 21 Jan 2026, Ex. 3a–3b (theory: Lecture 3, slide 3 and Lecture 2, slide 3)*

A perceptron in the input space has a straight-line boundary splitting the plane into two **half-planes**, which are unbounded; the inside of an ellipse is bounded, so no line can separate inside from outside. With φ(x) = [1, x₁², x₂²]ᵀ the perceptron computes g(x) = w₀ + w₁x₁² + w₂x₂²; with w₀ > 0, w₁, w₂ < 0 the boundary g(x) = 0 is an ellipse. Explicitly w₀ = 1, w₁ = −1/a², w₂ = −1/b²: g(x) ≥ 0 ⇔ x₁²/a² + x₂²/b² ≤ 1 ⇒ ŷ = +1 (STOP) inside. Linear in the feature space, nonlinear in the original space.

---

## Lecture 4 – Logistic regression and numerical optimization

### Key formulas
- ★ Logistic function: h(s) = eˢ/(1+eˢ) = 1/(1+e⁻ˢ) (exam notation: σ(z)); h(0) = ½
- ★ Property: 1 − h(s) = h(−s)
- ★ Model: P(y = +1|x) = h(wᵀx), P(y = −1|x) = 1 − h(wᵀx) ⇒ P_w(y|x) = h(y·wᵀx)
- Maximum likelihood: max_w Πᵢ P_w(y⁽ⁱ⁾|x⁽ⁱ⁾) ⇔ min_w −(1/N) ln Πᵢ P_w(y⁽ⁱ⁾|x⁽ⁱ⁾)
- ★ Cross-entropy: E_in(w) = (1/N) Σᵢ ln(1 + e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾)); pointwise err = ln(1 + e^(−y·s))
- ★ Gradient: ∇E_in = (1/N) Σᵢ [e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾)/(1 + e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾))]·(−y⁽ⁱ⁾x⁽ⁱ⁾) = −(1/N) Σᵢ h(−y⁽ⁱ⁾s⁽ⁱ⁾)·y⁽ⁱ⁾x⁽ⁱ⁾
- Gradient descent: w(k+1) = w(k) + η·v; Taylor: E_in(w(k+1)) ≈ E_in(w(k)) + ∇E_inᵀ(w(k+1) − w(k))
- Optimal direction: v̂ = −∇E_in/‖∇E_in‖ ⇒ ΔE_in ≈ −η‖∇E_in‖
- Basic GD: w(k+1) = w(k) − η·∇E_in/‖∇E_in‖
- ★ Variable step: η(k) = η·‖∇E_in(w(k))‖ ⇒ GD: w(k+1) = w(k) − η·∇E_in(w(k))
- Pointwise gradient: δE_in⁽ⁱ⁾(w) = ∂/∂w err(y⁽ⁱ⁾, h(x⁽ⁱ⁾)); for logistic regression −h(−y⁽ⁱ⁾s⁽ⁱ⁾)·y⁽ⁱ⁾x⁽ⁱ⁾
- ★ SGD: w(k+1) = w(k) − η·δE_in⁽ⁱ⁾(w(k)), i random in 𝒟_N
- ★ Newton: w(k+1) = w(k) − [∂²E_in/∂w²]⁻¹·∇E_in(w(k)); quasi-Newton: w(k+1) = w(k) − M⁻¹·∇E_in(w(k)), M ≈ Hessian

### L4.Q1 – Why estimate a probability instead of building a classifier?
*Source: Lecture 4, slides 2–3*

When the target is **noisy**: the same x can produce different y because of unmeasured factors (e.g. blood pressure, age, weight → heart attack). Any hard classifier has a large misclassification error, and "the risk is 73%" is more informative than yes/no. The problem formulation changes: x ∼ P(x), y ∼ P(y|x) instead of a deterministic f, and the algorithm outputs an estimate of P(y|x) instead of f̂.

### L4.Q2 – Why are the linear classifier and linear regression outputs unsuitable?
*Source: Lecture 4, slide 4*

Both compute s = wᵀx. The perceptron outputs sign(s) ∈ {−1, +1}: only yes/no. Linear regression outputs s ∈ ℝ: it can be −5 or 1000. A probability must lie in [0, 1], so we need a different output transformation.

### L4.Q3 – Describe the logistic model and the logistic function. ★
*Source: Lecture 4, slide 5*

x → [wᵀx] → s → [h(·)] → ŷ ∈ [0, 1], with h(s) = eˢ/(1+eˢ), a sigmoid: h → 0 as s → −∞, h(0) = ½, h → 1 as s → +∞. Values: h(−2) ≈ 0.12, h(1) ≈ 0.73, h(2) ≈ 0.88, h(4) ≈ 0.98. The score is interpreted as a measure of **risk**. Compared with sign (hard threshold) and s (no threshold), h is a **soft threshold**. Predicting +1 when h > 0.5 ⇔ s > 0, so the decision boundary is still a hyperplane: logistic regression is a linear classifier that also gives a confidence.

### L4.Q4 – (Derivation) Prove that 1 − h(s) = h(−s) and write P(y|x). ★
*Source: Lecture 4, slide 6*

1 − h(s) = 1 − eˢ/(1+eˢ) = (1+eˢ−eˢ)/(1+eˢ) = 1/(1+eˢ). h(−s) = e⁻ˢ/(1+e⁻ˢ); multiplying numerator and denominator by eˢ gives 1/(1+eˢ). They are equal (the sigmoid is symmetric around (0, ½)). Therefore P(y|x) = h(wᵀx) if y = +1 and 1 − h(wᵀx) = h(−wᵀx) if y = −1.

### L4.Q5 – Write the compact form of the model and verify it.
*Source: Lecture 4, slide 6*

P_w(y|x) = h(y·wᵀx). For y = +1: h(wᵀx) ✓. For y = −1: h(−wᵀx) = 1 − h(wᵀx) ✓. The ±1 encoding makes one formula cover both classes.

### L4.Q6 – (Exam) Interpret p(y|x;θ) and the sign of a parameter (e.g. vibration θ_V).
*Source: Lecture 4, slides 5–6; exam 21 Jan 2026, Ex. 1a*

p(yᵢ|xᵢ;θ) = σ(yᵢθᵀxᵢ) is the probability the model assigns to the **observed** label yᵢ given the features (for y = +1 the probability of failure, for y = −1 of no failure). θᵀx is the risk score: θ_V > 0 ⇒ more vibration raises the score and the failure probability (physically sensible); θ_V < 0 would mean more vibration lowers the risk, which is implausible and suggests model misspecification. Extra: ln[P(+1)/P(−1)] = θᵀx (log-odds), so a unit increase in V multiplies the odds of failure by e^(θ_V).

### L4.Q7 – Why can't we fit logistic regression by comparing y and h(x)?
*Source: Lecture 4, slide 7*

y is a label (±1) and h(x) is a probability. We never observe the true probability P(y|x), only outcomes, so a squared error between y and h makes no sense. We need a different criterion: maximum likelihood.

### L4.Q8 – Describe the maximum likelihood approach.
*Source: Lecture 4, slide 8*

Choose the w under which **the observed labels are the most probable**: max_w Πᵢ P_w(y⁽ⁱ⁾|x⁽ⁱ⁾). The product comes from the i.i.d. assumption (independent ⇒ joint probability is the product; identically distributed ⇒ same model for every sample). Example: three points with the same x, labels +1, +1, −1: L(w) = h(w)²(1 − h(w)) is maximized when h = 2/3 (w = ln 2), i.e. the model learns the observed frequency.

### L4.Q9 – Why can we replace max Π with min −(1/N) ln Π?
*Source: Lecture 4, slide 9*

The change of sign turns max into min; 1/N is only a scaling factor; ln is monotonic, so it does not change the position of the optimal w. The log also turns the product into a sum (easier to differentiate, and it avoids numerical underflow of a product of many numbers < 1).

### L4.Q10 – (Derivation) Derive the cross-entropy error. ★
*Source: Lecture 4, slide 10*

−(1/N) ln Π P = −(1/N) Σ ln P(y⁽ⁱ⁾|x⁽ⁱ⁾) = (1/N) Σ ln(1/P(y⁽ⁱ⁾|x⁽ⁱ⁾)) = (1/N) Σ ln(1/h(y⁽ⁱ⁾wᵀx⁽ⁱ⁾)). Since 1/h(s) = (1+eˢ)/eˢ = 1 + e⁻ˢ: **E_in(w) = (1/N) Σ ln(1 + e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾))**. Each term err(y⁽ⁱ⁾, h(x⁽ⁱ⁾)) is the cross-entropy, the pointwise error of logistic regression. Exam notation: with σ(z) = 1/(1+e⁻ᶻ), −log σ(z) = log(1+e⁻ᶻ) gives the same J(θ) = Σ log(1 + e^(−yᵢθᵀxᵢ)).

### L4.Q11 – How does the pointwise error ln(1 + e^(−y·s)) behave?
*Source: Lecture 4, slide 10 (extra)*

It depends only on y·s (the same quantity as in the PLA proof). y·s = +3 (correct and confident) → 0.05; y·s = 0 (on the boundary) → ln 2 ≈ 0.69; y·s = −3 (wrong and confident) → 3.05, growing roughly linearly. Unlike the 0/1 error it is smooth (differentiable) and still penalizes correct points that are close to the boundary.

### L4.Q12 – Compare the pointwise errors seen so far.
*Source: Lecture 4, slide 10 (extra, compares Lectures 1–3)*

Regression (L2): (y − s)², squared error. Classification (L1, L3): ⟦y ≠ sign(s)⟧, 0/1 error. Probability (L4): ln(1 + e^(−y·s)), cross-entropy. Every learning problem is "E_in = average of a pointwise error"; what changes is the error.

### L4.Q13 – (Derivation) Compute the gradient of the cross-entropy. Why is there no closed-form solution? ★
*Source: Lecture 4, slides 11–12*

With u = −y·wᵀx: d/du ln(1+eᵘ) = eᵘ/(1+eᵘ) and ∂u/∂w = −y·x. Chain rule: ∇E_in = (1/N) Σ [e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾)/(1 + e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾))]·(−y⁽ⁱ⁾x⁽ⁱ⁾). w enters the cost nonlinearly (exponential inside a log), so ∇E_in(ŵ) = 0 is a transcendental equation with no analytical solution ⇒ we need a numerical (nonlinear optimization) solution.

### L4.Q14 – Interpret −∇E_in and link it to PLA.
*Source: Lecture 4, slide 12 (extra)*

e^(−ys)/(1+e^(−ys)) = h(−y·s) is the probability the model assigns to being **wrong** on that point, so −∇E_in = (1/N) Σ h(−y⁽ⁱ⁾s⁽ⁱ⁾)·y⁽ⁱ⁾x⁽ⁱ⁾. Moving against the gradient adds y·x to w, the same direction as the PLA update. Difference: PLA uses one misclassified point with weight 1; logistic regression uses all points, each weighted by how wrong it is (badly classified points push hard, well-classified ones almost nothing).

### L4.Q15 – What are the two main problems of numerical optimization?
*Source: Lecture 4, slide 13*

Objective: start from an initial (possibly random) guess and move toward the global optimum. (1) How to "roll down" the surface? (gradient descent). (2) How not to get stuck in local minima? (initialization).

### L4.Q16 – Discuss the initialization issue. Is it a problem for logistic regression?
*Source: Lecture 4, slide 14*

For a non-convex E_in, different starting points roll into different minima (some local, one global). Heuristic: try several initial conditions and keep the result with the lowest E_in. Good news: the logistic cost is **convex** ⇒ only one minimum, reached from any starting point. Why: Hessian = (1/N) Σ h(s)(1−h(s))·x·xᵀ ⪰ 0 (as XᵀX in regression). Caveat: if the data are linearly separable, E_in keeps decreasing as ‖w‖ → ∞ (the model becomes ever more confident), so no finite minimum exists – convex guarantees no local minima, not that a minimum is reached. Regularization fixes this.

### L4.Q17 – What is the key idea of gradient descent?
*Source: Lecture 4, slides 15–16*

Build a sequence w(0) → w(1) → … → ŵ by taking a **small step** in the direction in which E_in decreases: w(k+1) = w(k) + η·v, with η > 0 the step size (how far) and v a unit vector (which way).

### L4.Q18 – (Derivation) Derive the optimal direction v. ★
*Source: Lecture 4, slides 17–20*

We want ΔE_in = E_in(w(k+1)) − E_in(w(k)) negative and as large as possible in magnitude. First-order Taylor around w(k): E_in(w(k+1)) ≈ E_in(w(k)) + ∇E_inᵀ(w(k))·(w(k+1) − w(k)) (valid for small steps). Substituting w(k+1) − w(k) = η·v: ΔE_in ≈ η·∇E_inᵀv. By Cauchy–Schwarz (‖v‖ = 1): |ΔE_in| ≤ η‖∇E_in‖. Since ∇ᵀv = ‖∇‖cos θ, the most negative value is at θ = 180°: **v̂ = −∇E_in/‖∇E_in‖**. Check: E_in(w(k+1)) ≈ E_in(w(k)) − η‖∇E_in‖²/‖∇E_in‖ = E_in(w(k)) − η‖∇E_in‖, the maximum decrease (steepest descent).

### L4.Q19 – Write the basic GD update and explain its terms.
*Source: Lecture 4, slide 21*

w(k+1) = w(k) − η·∇E_in(w)/‖∇E_in(w)‖. η is a scalar (size of the jump); ∇E_in/‖∇E_in‖ is a unit vector (direction of the jump). It works, but has problems if implemented like this: the step length is always η, even next to the minimum. The graph on this slide shows why it is called gradient descent: to the right of the minimum the slope is positive, so −∇ points left; to the left it points right. Either way, w moves downhill.

### L4.Q20 – Discuss the choice of the step size η. ★
*Source: Lecture 4, slide 22; exam 7 Nov 2022, Ex. 1c*

η small: many tiny steps, it converges but very slowly. η large: it jumps over the minimum and bounces from side to side, possibly never settling. η variable: large steps far from the minimum and small steps close to it ⇒ fast and precise. Example: with the normalized update and fixed η = 0.5, w oscillates forever between 0.5 and 1.0 around the optimum 0.693; with w(k+1) = w(k) − η∇E_in the step η‖∇E_in‖ shrinks automatically as the gradient vanishes and w converges smoothly.

### L4.Q21 – (Exam) Decision rule and effect of the threshold. ★
*Source: Exam 21 Jan 2026, Ex. 1c(ii) (theory: Lecture 4, slide 5)*

ŷ = +1 if θᵀx ≥ γ, −1 otherwise (often γ = 0 ⇔ probability ≥ 0.5). Increasing γ makes the classifier more **conservative**: fewer false positives (false alarms), more false negatives (missed events). In a failure-prediction context, missing a real failure is costly, so a lower threshold may be preferable.

### L4.Q22 – (Derivation) How can η be varied? Derive the standard gradient descent update. ★
*Source: Lecture 4, slides 22–23*

Slide 22 asks how to obtain the "η variable" behaviour: large steps far from the minimum, small steps close to it. The heuristic on slide 23 is **η(k) = η·‖∇E_in(w(k))‖** (η(k) is called the learning rate). The gradient norm is an indicator of how far w(k) is from the minimum: it is zero at the minimum and large on steep slopes. For a quadratic E = ½a(w − w*)² this is exact, since ‖∇E‖ = a·|w − w*|.

Substitute η(k) into the basic update w(k+1) = w(k) − η(k)·∇E_in/‖∇E_in‖. The two norms cancel:

**w(k+1) = w(k) − η·∇E_in(w(k))**

The effective step length η‖∇E_in‖ shrinks automatically as the minimum is approached. Notes: in most books the fixed η in this formula is the "learning rate". It is still a heuristic: on a plateau far from the minimum the gradient is small too, so progress is slow.

### L4.Q23 – (Extra) When does gradient descent with a fixed η converge?
*Source: Lecture 4, slides 22–23 (extra, not on the slides)*

Take a quadratic E(w) = ½a(w − w*)² + c, with a > 0. Then ∇E = a(w − w*) and the update gives **w(k+1) − w* = (1 − ηa)(w(k) − w*)**: the error is multiplied by (1 − ηa) at every step.
- 0 < η < 1/a: monotone convergence (very small η ⇒ very slow);
- 1/a < η < 2/a: convergence while bouncing from side to side;
- η > 2/a: divergence.

So GD converges only if 0 < η < 2/a, and η = 1/a reaches the minimum in one step (this is what Newton's method does automatically). Examples: on a bowl with a = 1, η = 1.9 bounces but converges, η = 2.1 diverges. In the 2-of-3 logistic example the curvature at the minimum is h(1 − h) = 2/9, so η must be below 9 (η = 12 keeps bouncing).

### L4.Q24 – Write the gradient descent algorithm.
*Source: Lecture 4, slide 24*

GD rule: w(k+1) = w(k) − η·∇E_in(w(k)).
1. **Initialize** the weights w(0): zeros or random. For logistic regression the cost is convex, so any start works; for non-convex costs try several starts (slide 14).
2. **For k = 0, 1, 2, …:**
   - compute the gradient ∇E_in(w(k)) (for logistic regression: the slide 12 formula, a sum over all N points);
   - set the direction v̂ = −∇E_in(w(k)) (no longer normalized: the norm was absorbed into η(k));
   - update the weights w(k+1) = w(k) + η·v̂;
   - iterate until it is time to stop (slide 25).

### L4.Q25 – List and discuss the stopping criteria. ★
*Source: Lecture 4, slide 25*

In theory we stop when ∇E_in = 0, but numerically we never land exactly there.
1. **Threshold on ‖∇E_in(w(k))‖**, e.g. 10⁻⁶: stop at a point practically at the minimum (top graph: ∇E_in = 0 at w*, stop where it is 10⁻⁶). The most natural criterion.
2. **Upper bound on the number of iterations**: a safety net when η is too large (bouncing, divergence), too small (slow), or when the minimum is at infinity (logistic regression on separable data).
3. **Threshold on E_in** (bottom graph: E_in vs iterations, stop when it crosses the threshold). Weakness: you must know which value is reachable; with noisy data the minimum of E_in is above 0, so a too low threshold is never reached. Variant: stop when E_in stops improving.
4. **Combination of the above**: in practice, stop as soon as any of them is met (e.g. gradient < 10⁻⁶ or k > 10⁴).

### L4.Q26 – Why can full gradient descent be expensive? Define the pointwise gradient.
*Source: Lecture 4, slide 26*

E_in(w) = (1/N) Σᵢ err(y⁽ⁱ⁾, h(x⁽ⁱ⁾)), so ∇E_in(w) = (1/N) Σᵢ ∂/∂w err(y⁽ⁱ⁾, h(x⁽ⁱ⁾)). Every single GD step requires going through **all N samples**: with very large datasets (e.g. 10⁶ sensor samples) each iteration is expensive, and many iterations are needed. The **pointwise gradient** is the gradient of the error on one point only: **δE_in⁽ⁱ⁾(w) = ∂/∂w err(y⁽ⁱ⁾, h(x⁽ⁱ⁾))**, so ∇E_in = (1/N) Σᵢ δE_in⁽ⁱ⁾. For logistic regression: δE_in⁽ⁱ⁾ = −h(−y⁽ⁱ⁾s⁽ⁱ⁾)·y⁽ⁱ⁾x⁽ⁱ⁾.

### L4.Q27 – Describe stochastic gradient descent and compare it with GD. ★
*Source: Lecture 4, slide 27*

**SGD: w(k+1) = w(k) − η·δE_in⁽ⁱ⁾(w(k))**, with i any point of 𝒟_N, usually picked at random (in practice the data are often shuffled and used in passes called epochs). Start from an initial w(0), not from a data point, and use a different data point at every step.

Why it works: if i is uniform, the average of δE_in⁽ⁱ⁾ over i equals ∇E_in, so each step points in the right direction **on average** (unbiased estimate of the gradient).

Graph (level curves of E_in in the (w₁, w₂) plane): GD moves smoothly, perpendicular to the level curves; SGD follows a zig-zag but drifts toward the centre. "It can be proven that w(k) converges to the minimum asymptotically" (precisely: with a constant η it ends up jittering near the minimum; a slowly decreasing η makes it converge).

Pros: each step costs 1 point instead of N, so many more steps for the same computation; it can learn online as data arrive; its noise can help escape shallow local minima. Cons: noisy path, never settles exactly with constant η, needs care with η. (Mini-batch SGD uses a small random group of points per step, a compromise.)

### L4.Q28 – (Extra) Why does the SGD path zig-zag, and why doesn't it stop at the minimum?
*Source: Lecture 4, slide 27 (extra)*

The level curves on the slide belong to the **total** E_in. Each single point has its own error surface with its own minimum (in linear regression, one point is fitted perfectly by a whole line of w's), so its gradient points toward *its* preferred w, not toward the centre. SGD follows one of these pulls at a time, and different points pull in different directions, hence the zig-zag. Far from the minimum most points agree, so the path is fairly straight; near the minimum they disagree. At the minimum the pulls cancel **only on average** (∇E_in = 0), but each single pull is non-zero, so SGD keeps being kicked around. Example (2-of-3 data, minimum w = ln 2): pulls −1/3, −1/3, +2/3, average 0; SGD with η = 1 moves by +1/3 or −2/3 at every step.

### L4.Q29 – (Extra) What is the link between logistic SGD and PLA?
*Source: Lecture 4, slide 27 and Lecture 3, slide 7 (extra)*

Logistic SGD step: w(k+1) = w(k) + η·h(−y⁽ⁱ⁾s⁽ⁱ⁾)·y⁽ⁱ⁾x⁽ⁱ⁾. PLA step: w(k+1) = w(k) + y⁽ⁱ⁾x⁽ⁱ⁾. Both use one point at a time and move w along y·x. PLA uses only misclassified points with weight 1; logistic SGD uses any point, weighted by the probability h(−y·s) of being wrong. Logistic SGD is a "soft PLA".

### L4.Q30 – (Derivation) Derive Newton's method and discuss pros and cons. ★
*Source: Lecture 4, slide 28*

GD uses a first-order Taylor expansion (a tangent: direction, but no natural step length). Newton uses the **second-order** expansion: E_in(w + Δ) ≈ E_in(w) + ∇E_inᵀΔ + ½ΔᵀHΔ, with H = ∂²E_in/∂w² the Hessian. This is a paraboloid; setting its gradient in Δ to zero gives ∇E_in + HΔ = 0 ⇒ Δ = −H⁻¹∇E_in:

**w(k+1) = w(k) − [∂²E_in/∂w² |_{w(k)}]⁻¹ ∇E_in(w(k))** (classical Newton)

Interpretation: jump to the bottom of the local quadratic approximation. The Hessian (curvature) sets the step automatically: small steps in strongly curved directions, large steps in flat ones; no η to tune. In 1D the step is −E'(w)/E''(w).

Pros: very few iterations near the minimum. Cons: H must be invertible and positive definite (otherwise the step can go uphill or toward a saddle); each step needs the (m+1)×(m+1) Hessian and its inverse, which is very expensive with many parameters. For logistic regression H = (1/N)Σ h(s)(1 − h(s))·x·xᵀ ⪰ 0.

### L4.Q31 – (Extra) What happens if one Newton step is applied to linear regression?
*Source: Lecture 4, slide 28 and Lecture 2, slide 12 (extra)*

For linear regression ∇E_in = (2/N)(XᵀXw − XᵀY) and H = (2/N)XᵀX. From any w: w − H⁻¹∇E_in = w − (XᵀX)⁻¹(XᵀXw − XᵀY) = (XᵀX)⁻¹XᵀY, the OLS solution, in **one step**. Reason: E_in is exactly quadratic, so its second-order approximation is exact. The OLS formula is one Newton step.

### L4.Q32 – What are quasi-Newton methods? Give a unified view of GD, SGD and Newton. ★
*Source: Lecture 4, slide 28*

Quasi-Newton: **w(k+1) = w(k) − M⁻¹∇E_in(w(k))**, where M is an **approximation of the Hessian**, built step by step from how the gradient changes (best-known: BFGS; MATLAB's fminunc uses a quasi-Newton method by default). It keeps most of Newton's speed without computing and inverting the true Hessian.

All methods have the form w(k+1) = w(k) − M⁻¹·(gradient):
- GD: M = (1/η)I, full gradient. Cheap steps, many iterations.
- SGD: M = (1/η)I, gradient of one point. Very cheap steps, very many noisy iterations.
- Newton: M = true Hessian. Expensive steps, very few iterations.
- Quasi-Newton: M ≈ Hessian. Medium cost, few iterations.

### L4.Q33 – (Extra) Compare GD, Newton and SGD on the 2-of-3 example.
*Source: Lecture 4, slides 23, 27, 28 (extra)*

Three points with x = 1 (no bias), labels +1, +1, −1. E'(w) = h(w) − 2/3, E''(w) = h(w)(1 − h(w)), minimum at w = ln 2 ≈ 0.693.
- **GD** (w − 3E'), from 0: 0 → 0.500 → 0.633 → 0.673 → 0.687 → 0.691 → 0.692 → 0.693 (about 7 steps).
- **Newton** (w − E'/E''), from 0: E' = −0.167, E'' = 0.25 ⇒ 0.667; then 0.6930; then 0.69315 (3 steps).
- **Basic GD with fixed length** (η = 0.5): 0 → 0.5 → 1.0 → 0.5 → 1.0 … bounces forever.
- **SGD** (η = 1, points +, −, +, +, −, +): 0 → 0.500 → −0.122 → 0.408 → 0.807 → 0.116 → 0.587: cheap steps, but it jitters around 0.693.

---

## Lecture 5 – The bias-variance trade-off

### Key formulas
- Model learned from dataset 𝒟: h^𝒟(x) = w^𝒟ᵀx (it is random, because 𝒟 is random)
- Quantity analysed: 𝔼_𝒟[E_out(w^𝒟)] = 𝔼_𝒟{𝔼ₓ[(f(x) − h^𝒟(x))²]} (noiseless target y = f(x))
- ★ Average model: h̄(x) ≜ 𝔼_𝒟[h^𝒟(x)]
- Variance identity: 𝔼_𝒟[(h^𝒟(x) − h̄(x))²] = 𝔼_𝒟[h^𝒟(x)²] − h̄(x)²
- ★ Decomposition: **𝔼_𝒟[E_out(w^𝒟)] = 𝔼ₓ[(f(x) − h̄(x))²] + 𝔼ₓ{𝔼_𝒟[(h^𝒟(x) − h̄(x))²]}** = bias + variance
- With noise y = f(x) + ε, Var(ε) = σ²: 𝔼_𝒟[E_out] = bias + variance + σ² (extra)

### L5.Q1 – What does the dartboard picture on the title slide represent?
*Source: Lecture 5, slide 1*

Imagine repeating learning many times, each time with a different dataset. Each dart is the model learned from one dataset, the bullseye is the target f. **Bias** is how far the centre of the darts (the average model) is from the bullseye: a systematic error. **Variance** is how spread out the darts are: how much the result depends on which dataset you happened to get. Low bias + low variance: tight group on the bullseye (ideal). Low bias + high variance: centred but scattered. High bias + low variance: tight group off-centre. High bias + high variance: scattered and off-centre (worst).

### L5.Q2 – Why do we need a deeper analysis of E_out?
*Source: Lecture 5, slide 2*

So far we have mostly talked about **fitting** (making E_in small with OLS, PLA, GD). For generalization we only had rough results: E_out = E_in + O(m/N) (linear regression) and E_out = E_in + O(√((m/N) log N)) (perceptron). They say that the gap grows with complexity and shrinks with data, but not much more. To do **learning** and not just fitting we need a deeper analysis of what happens to E_out: the bias-variance trade-off is one perspective on the generalization problem.

### L5.Q3 – What is the setting of the analysis, and why do we study 𝔼_𝒟[E_out(w^𝒟)]?
*Source: Lecture 5, slide 3*

For simplicity (with no loss of generality) take linear regression: ŷ = ŵᵀx, ŵ = argmin E_in(w) on 𝒟_N, with **N fixed and finite**. The ideal objective is E_out(w^𝒟) = 𝔼ₓ[(y − w^𝒟ᵀx)²]. The superscript 𝒟 stresses that the learned weights **depend on the dataset**: another set of N samples gives another ŵ, so w^𝒟 is a random variable. We would be "even happier" minimizing **𝔼_𝒟[E_out(w^𝒟)]**, the out-of-sample error averaged over all possible datasets of size N. It evaluates the **learning procedure** (ℋ + algorithm + N), not one particular model, which is what matters when choosing ℋ. Two averages appear: 𝔼ₓ over the points where the model is used, 𝔼_𝒟 over the datasets it could have been trained on.

### L5.Q4 – Define the average model and explain its meaning. ★
*Source: Lecture 5, slide 4*

**h̄(x) ≜ 𝔼_𝒟[h^𝒟(x)]**: at each x, the average of the predictions of all the models learned from all possible datasets of size N. In the dartboard, it is the centre of the darts. It is a theoretical object (with one dataset you cannot compute it). It does not necessarily belong to ℋ (for linear models the average of lines is a line, but not in general).

### L5.Q5 – (Derivation) Derive the bias-variance decomposition. ★
*Source: Lecture 5, slides 4–6*

1. Noiseless target (y = f(x)): 𝔼_𝒟[E_out] = 𝔼_𝒟{𝔼ₓ[(f(x) − h^𝒟(x))²]}.
2. Swap the expectations (independent sources of randomness) and expand the square:
   𝔼ₓ{𝔼_𝒟[f²]} + 𝔼ₓ{𝔼_𝒟[(h^𝒟)²]} − 2𝔼ₓ{𝔼_𝒟[f·h^𝒟]}.
3. f does not depend on 𝒟: 𝔼_𝒟[f²] = f² and 𝔼_𝒟[f·h^𝒟] = f·𝔼_𝒟[h^𝒟] = f·h̄. So:
   𝔼ₓ{f² + 𝔼_𝒟[(h^𝒟)²] − 2f·h̄}.
4. Add and subtract h̄² to complete the square: f² − 2f·h̄ + h̄² = (f − h̄)². So:
   𝔼ₓ[(f − h̄)²] + 𝔼ₓ{𝔼_𝒟[(h^𝒟)²]} − 𝔼ₓ{h̄²}  (the last two terms are the "?" on slide 5).
5. Slide 6: 𝔼_𝒟[(h^𝒟 − h̄)²] = 𝔼_𝒟[(h^𝒟)²] + h̄² − 2h̄·𝔼_𝒟[h^𝒟] = 𝔼_𝒟[(h^𝒟)²] − h̄² (since h̄ does not depend on 𝒟 and 𝔼_𝒟[h^𝒟] = h̄). This is Var = 𝔼[Z²] − (𝔼Z)².
6. Therefore 𝔼_𝒟[E_out] = 𝔼ₓ[(f − h̄)²] + 𝔼ₓ{𝔼_𝒟[(h^𝒟 − h̄)²]}.

### L5.Q6 – State the decomposition and explain bias and variance. ★
*Source: Lecture 5, slide 7*

**𝔼_𝒟[E_out(w^𝒟)] = 𝔼ₓ[(f(x) − h̄(x))²] + 𝔼ₓ{𝔼_𝒟[(h^𝒟(x) − h̄(x))²]}** = bias error + variance error.
- **Bias**: constant with respect to 𝒟. It represents how well the model class ℋ can express a model "close" to the target function. It compares f with the *average* model: it is the systematic error that remains even averaging over infinitely many datasets. It is a property of ℋ (e.g. lines cannot follow a sine).
- **Variance**: independent of f(x) (f does not appear in it). It represents the deviation of h^𝒟(x) from the average h̄(x): how much the learned model changes when the dataset changes, i.e. its sensitivity to the particular N samples.

### L5.Q7 – (Extra) How does the decomposition change with noisy data?
*Source: Lecture 5, slide 7 (extra, not on the slide)*

If y = f(x) + ε with zero-mean noise of variance σ² independent of x and 𝒟, the same derivation gives **𝔼_𝒟[E_out] = bias + variance + σ²**. σ² is the **irreducible error**: no model and no amount of data can predict pure noise.

### L5.Q8 – Discuss the two extreme cases of hypothesis set.
*Source: Lecture 5, slide 8*

**Case A, very small ℋ = {h⁰}**: whatever the dataset, the algorithm returns h⁰, so h^𝒟 = h⁰ = h̄ for every 𝒟 ⇒ **variance = 0**; but h⁰ may be far from f ⇒ **only bias**. **Case B, very large ℋ** (containing models close to f): each dataset gives a different h^𝒟, scattered around f(x), so their average h̄ ≈ f ⇒ **bias ≈ 0**; but each single model can be far from f ⇒ **only variance**. Enlarging ℋ moves from A toward B: bias decreases, variance increases. That is the trade-off.

### L5.Q9 – Describe how the expected errors change with the complexity of ℋ (N fixed). ★
*Source: Lecture 5, slide 9*

- 𝔼_𝒟[E_in] **decreases** monotonically with complexity: a richer model always fits the training data better.
- 𝔼_𝒟[E_out] is **U-shaped**. Left: **underfitting** (too simple, high bias, both errors high). Right: **overfitting** (too complex, E_in keeps dropping but E_out rises because variance explodes: the model fits noise and accidental patterns). The minimum is the **"right complexity"**, the best bias/variance balance.
- The gap between the two curves (generalization error) grows with complexity, consistent with O(m/N).
- Complexity of ℋ means how flexible the model class is (roughly the number of free parameters: constant < line < parabola < high-degree polynomial < neural network). It is not the amount of data, which is N.

### L5.Q10 – Describe the learning curves (complexity fixed, N varying). ★
*Source: Lecture 5, slide 10*

- 𝔼_𝒟[E_out] **decreases** with N: more data ⇒ each dataset is more representative ⇒ models from different datasets agree more ⇒ **variance decreases**.
- 𝔼_𝒟[E_in] **increases** with N: with few points it is easy to fit them perfectly (a line through 2 points has E_in = 0); with many points the model cannot pass through all of them.
- Both converge to the **bias** level (dashed line); with noise the asymptote is bias + σ². E_in is optimistic: on average it stays below E_out, and the gap closes as N grows.

### L5.Q11 – (Extra) Why doesn't E_out go to zero when N → ∞? Will it ever reach zero?
*Source: Lecture 5, slide 10 (extra)*

As N → ∞ the variance goes to zero: every dataset gives the same model, the best one available in ℋ. The bias remains, because it is a property of ℋ: if f ∉ ℋ, even the best model in ℋ is wrong somewhere. Example: the best line for sin(πx) on [−1, 1] has slope 3/π and intercept 0, and still leaves an error 0.5 − 3/π² ≈ 0.20. E_out reaches 0 only if f ∈ ℋ **and** there is no noise; in practice σ² > 0, so the floor is bias + σ². The goal is not zero error but an E_out small enough for the application. Lowering the floor requires a more complex ℋ, which brings back more variance: that is where the trade-off appears.

### L5.Q12 – How should the model complexity be selected?
*Source: Lecture 5, slide 11*

Simple ℋ: learning curves converge **quickly** to a **high** bias level. Complex ℋ: they converge **slowly** to a **low** bias level. With **few data** the simple model has lower E_out (its variance is already small) ⇒ choose the simple model. With **many data** the complex model has lower E_out (its variance has died out, its low bias wins) ⇒ choose a complex model. **The right model selection must depend on the resources we have (the amount of data), not on the complexity of f(x).** Even if the true system is very complex, with few data a simple model predicts better.

### L5.Q13 – (Extra) Worked example: constants vs lines on a sine target.
*Source: Lecture 5, slides 8–11 (extra)*

Target f(x) = sin(πx), x uniform on [−1, 1], noiseless. H₀ = constants, H₁ = lines, fitted by least squares.
- Hand check for H₀, N = 2: h^𝒟 = (sin πx₁ + sin πx₂)/2; h̄ = 0 by symmetry; bias = 𝔼[sin²πx] = 0.5; variance = Var(sin πx)/2 = 0.25; 𝔼[E_out] = 0.75.
- Results (20,000 simulated datasets):

| N | Model | Bias | Variance | 𝔼[E_out] | 𝔼[E_in] |
|---|---|---|---|---|---|
| 2 | constant | 0.50 | 0.25 | **0.75** | 0.25 |
| 2 | line | 0.21 | 1.68 | 1.89 | 0.00 |
| 5 | constant | 0.50 | 0.10 | 0.60 | 0.40 |
| 5 | line | 0.20 | 0.21 | **0.41** | 0.10 |
| 20 | constant | 0.50 | 0.03 | 0.52 | 0.48 |
| 20 | line | 0.20 | 0.03 | **0.22** | 0.17 |

With N = 2 the constant wins even though the sine is not constant (slide 11); from N = 5 the line wins. For the line, E_in rises (0 → 0.10 → 0.17) and E_out falls (1.89 → 0.41 → 0.22), both toward the bias ≈ 0.20 (slide 10). With N = 2 the line has E_in = 0 and the worst E_out: overfitting in its purest form.

---

## Lecture 6 – Model selection (regularization and cross-validation)

### Key formulas
- Family of hypothesis sets: ℋ_m = {h | h(x) = wᵀx, w ∈ ℝ^m}, m = complexity index
- ★ ‖w‖₀ = number of non-zero elements of w (minimizing it is NP-hard)
- ‖w‖₂² = w₀² + w₁² + … + w²_{m−1}
- Constrained form: ŵ_reg = argmin_w E_in(w) s.t. ‖w‖₂² ≤ C (C = budget)
- ★ Penalized form: ŵ_reg = argmin_w E_aug(w), **E_aug(w) = E_in(w) + (λ/N)‖w‖₂²** (λ ↔ C: large C ↔ small λ)
- ∇E_aug(w) = ∇E_in(w) + (2λ/N)w
- ★ Regularized least squares (ridge): **ŵ_reg = (XᵀX + λI)⁻¹XᵀY**
- Validation split: 𝒟_train (N − k points) and 𝒟_val (k points); ŵ⁻ = model trained on 𝒟_train
- Validation error: E_val(ŵ⁻) = (1/k) Σ_{i=N−k+1}^{N} (y⁽ⁱ⁾ − ŵ⁻ᵀx⁽ⁱ⁾)²
- ★ Theorem: 𝔼_{𝒟val}[E_val] = E_out(ŵ⁻) (unbiased); Var[E_val − E_out] = 𝔼[(E_val − E_out)²] = O(1/k)
- Rule of thumb: k ≈ N/5
- ★ Leave-one-out: **E_cv = (1/N) Σ_{j=1}^{N} E_val⁽ʲ⁾(ŵ⁻⁽ʲ⁾)**, 𝔼_𝒟[E_cv] = 𝔼[E_out(ŵ⁻)] ≈ E_out(ŵ) for large N
- ★ Three-way split: training ≈ 70% (ŵ), validation ≈ 15% (λ, m, …), test ≈ 15% (E_out)

### L6.Q1 – What problem does model selection solve?
*Source: Lecture 6, slides 1–2*

How to choose the "just right" complexity. A too simple model (e.g. a line on curved data) **underfits**; a too complex model (a wiggly curve through every point, like the green boundary on slide 1) **overfits**; somewhere in between is "just right", and slide 2 ends with a "?": how do we find it in practice? Lecture 5 explained the trade-off in theory (𝔼[E_out] is U-shaped in the complexity), but E_out cannot be measured. This lecture gives two practical tools.

### L6.Q2 – Which two approaches are discussed, and in which setting?
*Source: Lecture 6, slide 3*

(1) **Regularization**, (2) **cross-validation**. Without loss of generality, linear regression with the family ℋ_m = {h | h(x) = wᵀx, w ∈ ℝ^m}, where **m is the complexity index** (number of parameters). Extra: the sets are nested, ℋ₁ ⊂ ℋ₂ ⊂ …, so a larger m can always reach a lower E_in, which is exactly why E_in cannot be used to choose m.

### L6.Q3 – What is the key idea of regularization? Explain the example ℋ*_m.
*Source: Lecture 6, slide 4*

m should be large (to make E_in small) but not too large (generalization, E_out = E_in + O(m/N)). Key idea: keep the model structure but **force some weights to be zero**. Example: ℋ*_m = {h | h(x) = wᵀx, w ∈ ℝ^m, w₁ = w₃ = 0}: two weights are fixed to zero, so the **equivalent number of parameters is m − 2**. Regularization is called a **"necessary evil"**: it deliberately worsens the fit (E_in increases) to improve generalization (E_out decreases).

### L6.Q4 – What is ‖w‖₀ and why can't we simply minimize it? ★
*Source: Lecture 6, slide 4*

‖w‖₀ := number of non-zero elements of w (not a true norm, despite the name). Ideally we would minimize E_in while keeping ‖w‖₀ small, i.e. choose which weights to switch off. But this is **NP-hard**: it is a combinatorial problem, since with m weights there are 2^m possible subsets of non-zero weights to try (extra: with m = 30 that is about 10⁹ least-squares problems).

### L6.Q5 – How is the problem relaxed? Write the constrained and the penalized forms. ★
*Source: Lecture 6, slide 5*

Instead of ‖w‖₀, use the squared Euclidean norm ‖w‖₂² = w₀² + w₁² + … + w²_{m−1}: instead of forcing weights to be exactly zero, we keep them **small**. This "relaxation" turns an NP-hard problem into an easy one.
- Constrained form: **ŵ_reg = argmin_w E_in(w) s.t. ‖w‖₂² ≤ C**, with C the **budget** on the size of the weights.
- Equivalent penalized form: **ŵ_reg = argmin_w E_in(w) + (λ/N)‖w‖₂²**. The objective is called **E_aug(w)** (augmented error). For each budget C there is a corresponding λ (λ ↔ C), with an inverse relation: tight budget (small C) ↔ strong penalty (large λ).

### L6.Q6 – (Extra) Why are the constrained and the penalized forms equivalent?
*Source: Lecture 6, slide 5 (extra)*

Geometric view in the (w₀, w₁) plane: the level curves of E_in are ellipses centred at ŵ_OLS; the constraint ‖w‖² ≤ C is a disk centred at the origin. If ŵ_OLS is inside the disk, the constraint is inactive and ŵ_reg = ŵ_OLS (λ = 0). Otherwise the solution lies on the circle, at the point where the smallest reachable ellipse touches it. There the two curves are tangent, so ∇E_in(ŵ_reg) is parallel and opposite to the normal of the circle, which is w itself: ∇E_in(ŵ_reg) = −(2λ/N)ŵ_reg for some λ ≥ 0. This is exactly ∇E_aug(ŵ_reg) = 0. λ is the Lagrange multiplier of the constraint (KKT conditions); smaller C ⇒ larger λ.

### L6.Q7 – What is E_aug and why is it a "proxy of E_out"? ★
*Source: Lecture 6, slide 5*

E_aug(w) = E_in(w) + (λ/N)‖w‖₂². Remember E_out = E_in + O(m/N) (Lecture 2, slide 14): E_out is E_in plus a complexity term we cannot compute. E_aug replaces that unknown term with a computable penalty that grows with the "size" of the model and is divided by N, like O(m/N). Minimizing E_aug is therefore closer to minimizing E_out than minimizing E_in alone. Extra, why large weights mean complexity: by Cauchy–Schwarz |h(x) − h(x′)| = |wᵀ(x − x′)| ≤ ‖w‖·‖x − x′‖, so a small ‖w‖ forbids the model from changing too fast; with polynomial features, large coefficients of alternating sign produce the wild oscillations of the overfitting curve on slide 2. The dependence on 1/N also means that with more data less regularization is needed (Lecture 5: more data ⇒ less variance).

### L6.Q8 – (Derivation) Derive the regularized least-squares solution. ★
*Source: Lecture 6, slide 6*

1. OLS minimizes E_in(w) = (1/N)‖Y − Xw‖² and gives ŵ = (XᵀX)⁻¹XᵀY (Lecture 2).
2. Now E_aug(w) = (1/N)‖Y − Xw‖² + (λ/N)wᵀw.
3. ∇E_aug(w) = ∇E_in(w) + (2λ/N)w = (2/N)(XᵀXw − XᵀY) + (2λ/N)w.
4. Set it to zero at ŵ_reg and multiply by N/2: XᵀXŵ_reg − XᵀY + λŵ_reg = 0 ⇒ (XᵀX + λI)ŵ_reg = XᵀY.
5. **ŵ_reg = (XᵀX + λI)⁻¹XᵀY** (regularized least squares, also called ridge regression).
6. Hessian: (2/N)(XᵀX + λI) ≻ 0 for λ > 0 ⇒ unique global minimum.

Notes: on the slide the first term of the third line is written with a minus sign in front, −(2/N)(XᵀXŵ_reg − XᵀY); the correct sign is plus (∇E_in = (2/N)(XᵀXw − XᵀY)). The boxed result is correct. The slide says "we assume (XᵀX + λI)⁻¹ exists": for λ > 0 it **always** exists, because zᵀ(XᵀX + λI)z = ‖Xz‖² + λ‖z‖² > 0 for every z ≠ 0. It is still a closed-form solution, the only regularized problem in the course with a formula.

### L6.Q9 – What is the effect of λ? Give a numerical example. ★
*Source: Lecture 6, slides 5–6; Lecture 5, slides 7–9*

- λ = 0: ŵ_reg = ŵ_OLS (no regularization).
- λ → ∞: ŵ_reg → 0 (the model predicts ≈ 0 everywhere: underfitting).
- In between: the weights are **shrunk** toward zero; ‖ŵ_reg‖ decreases and E_in increases as λ grows.

Bias-variance (Lecture 5): a larger λ means lower effective complexity ⇒ variance decreases, bias increases. 𝔼[E_out] as a function of λ is U-shaped (a mirror image of the complexity axis: small λ = complex, large λ = simple). Link with Lecture 2: Var(θ̂) = σ²(XᵀX)⁻¹ explodes when XᵀX is ill-conditioned; adding λI raises every eigenvalue of XᵀX by λ, so the inverse can no longer blow up.

Example: points (0, 1), (1, 2), (2, 4), model y = w₀ + w₁x. XᵀX = [3 3; 3 5], XᵀY = [7; 10].
- λ = 0: ŵ = [0.833; 1.5].
- λ = 1: XᵀX + I = [4 3; 3 6], det = 15, inverse (1/15)[6 −3; −3 4] ⇒ ŵ = (1/15)[42 − 30; −21 + 40] = [0.8; 1.267].
- λ = 3: ŵ = [0.667; 1.0].

Extra: the slides penalize all weights, w₀ included. In practice the intercept is often left unpenalized (or the data are centred), otherwise the predictions are pulled toward 0 instead of toward the mean of y.

### L6.Q10 – (Extra) How does regularization fix a singular XᵀX? Give an example.
*Source: Lecture 6, slide 6; Lecture 2, slide 13 (extra)*

Two points with the same input x = 1: (1, 1) and (1, 3). X = [1 1; 1 1], XᵀX = [2 2; 2 2] (det = 0): every w with w₀ + w₁ = 2 is an OLS solution (flat valley). With λ = 1: XᵀX + I = [3 2; 2 3], det = 5, inverse (1/5)[3 −2; −2 3], XᵀY = [4; 4] ⇒ ŵ_reg = [0.8; 0.8]. The penalty adds a bowl λ‖w‖² to the valley, so a unique minimum appears (the solution of the valley closest to the origin, shrunk). As λ → 0⁺ it tends to [1; 1], the minimum-norm OLS solution.

### L6.Q11 – (Extra) How is regularization used with logistic regression and gradient descent?
*Source: Lecture 6, slide 5; Lecture 4, slides 14 and 23 (extra)*

E_aug(w) = (1/N) Σ ln(1 + e^(−y⁽ⁱ⁾wᵀx⁽ⁱ⁾)) + (λ/N)‖w‖². There is no closed form, so use GD: w(k+1) = w(k) − η[∇E_in(w(k)) + (2λ/N)w(k)] = (1 − 2ηλ/N)·w(k) − η∇E_in(w(k)). At every step the weights first shrink by the factor (1 − 2ηλ/N): this is why ℓ₂ regularization is also called **weight decay**. It also fixes the caveat of Lecture 4, slide 14: with separable data E_in → 0 only as ‖w‖ → ∞, but the penalty grows with ‖w‖, so E_aug has a finite, unique minimum (its Hessian is the logistic Hessian plus (2λ/N)I ≻ 0).

### L6.Q12 – What is the idea of cross-validation? How does it differ from regularization?
*Source: Lecture 6, slide 7*

Regularization changes the cost and uses the same data to mimic E_out. Cross-validation does the opposite: **it uses the same cost and changes the way the data are used**. Key idea: partition 𝒟_N into two datasets: (1) a **training** set 𝒟_train of size N − k and (2) a **validation** set 𝒟_val of size k. The validation points are never used to fit the model, so for the model they behave like future, unseen data.

### L6.Q13 – Describe the validation procedure. ★
*Source: Lecture 6, slide 8*

1. **Optimization:** learn a model h⁻(x) on the training set only: E_in(w) = (1/(N − k)) Σ_{i=1}^{N−k} (y⁽ⁱ⁾ − wᵀx⁽ⁱ⁾)² ⇒ ŵ⁻. The superscript "−" means "trained on fewer points" (N − k instead of N).
2. **Evaluation:** compute the error of ŵ⁻ on the k validation points: (1/k) Σ_{i=N−k+1}^{N} (y⁽ⁱ⁾ − ŵ⁻ᵀx⁽ⁱ⁾)² ⇒ Ê_out(ŵ⁻), a "proxy" of the out-of-sample performance. The slide writes it as E_in(ŵ⁻) computed on the validation points; slide 10 calls it **E_val(ŵ⁻)**.

Extra: shuffle the data before splitting, so that the validation set is representative (e.g. not only the last days of a time series). Open question at the end of the slide: how to select k?

### L6.Q14 – Discuss the trade-off in the selection of k (size of the validation set). ★
*Source: Lecture 6, slide 9*

- **Small k:** large N − k ⇒ a better ŵ⁻ (good), but few validation points ⇒ a bad (noisy) estimate of E_out(ŵ⁻).
- **Large k:** a good estimate of E_out(ŵ⁻), but small N − k ⇒ a bad ŵ⁻.

The chain we want is E_val(ŵ⁻) ≈ E_out(ŵ⁻) ≈ E_out(ŵ): the first ≈ needs **k large**, the second ≈ (the model trained on N − k points is close to the final model trained on N points) needs **k small**. The two requirements conflict. **Rule of thumb: k ≈ N/5** (20% of the data for validation). Extra: after validation, retrain on all N points to get the final ŵ; E_val(ŵ⁻) is then a slightly pessimistic estimate of E_out(ŵ), because more data means lower E_out (learning curves, Lecture 5, slide 10).

### L6.Q15 – State the theorem on the validation error and explain why it holds. ★
*Source: Lecture 6, slide 10*

With E_val(ŵ⁻) = (1/k) Σ_{i=N−k+1}^{N} (y⁽ⁱ⁾ − ŵ⁻ᵀx⁽ⁱ⁾)²:
- **𝔼_{𝒟val}[E_val] = E_out(ŵ⁻)**: the validation error is **unbiased**.
- **Var[E_val − E_out] = 𝔼[(E_val − E_out)²] = O(1/k)**.

Why: ŵ⁻ does not depend on the validation points, so each validation point is a fresh i.i.d. sample and its error has expected value E_out(ŵ⁻); the average of k such errors is unbiased. The k errors are independent, so the variance of their average is σ²_e/k = O(1/k), and the typical estimation error is O(1/√k). Link with Lecture 1: this is the bin experiment with **one fixed hypothesis** (ŵ⁻ is chosen before looking at 𝒟_val), so for a bounded error Hoeffding applies with M = 1: ℙ(|E_val − E_out| > ε) ≤ 2e^(−2ε²k).

### L6.Q16 – Describe leave-one-out (LOO) cross-validation. ★
*Source: Lecture 6, slide 11*

A useful tool when N is not large (we cannot afford to set aside N/5 points). Key observation: there are N ways to split 𝒟_N into a training set of N − 1 points and a validation set of 1 point. For j = 1, …, N: train on all points except j ⇒ ŵ⁻⁽ʲ⁾; compute its error on point j ⇒ E_val⁽ʲ⁾(ŵ⁻⁽ʲ⁾). This gives N models and N validation errors, and

**E_cv = (1/N) Σ_{j=1}^{N} E_val⁽ʲ⁾(ŵ⁻⁽ʲ⁾)**.

It gets the best of both sides of slide 9: every model is trained on N − 1 points (like k small), and the estimate uses all N points for validation (like k large). Price: N trainings.

### L6.Q17 – Why is LOO still unbiased? What is k-fold cross-validation? ★
*Source: Lecture 6, slide 12*

Example on the slide: 3 points; each line is fitted on 2 of them and its error is measured on the third (Eval⁽¹⁾, Eval⁽²⁾, Eval⁽³⁾).

𝔼_𝒟[E_cv] = 𝔼_𝒟[(1/N) Σ_j E_val⁽ʲ⁾(ŵ⁻⁽ʲ⁾)] = (1/N) Σ_j 𝔼_𝒟[E_val⁽ʲ⁾(ŵ⁻⁽ʲ⁾)] = (N/N)·𝔼[E_out(ŵ⁻)]: still unbiased, as an estimate of the expected E_out of a model trained on N − 1 points. If N is large, E_out(ŵ⁻⁽ʲ⁾) ≈ E_out(ŵ). Extra: the N terms are not independent (the models share N − 2 training points), so the variance is not simply O(1/N), but it is much smaller than with a single validation point.

**k-fold cross-validation** is the extension of this idea using k points instead of 1: split 𝒟_N into folds, train on all folds but one, validate on the remaining one, rotate and average. LOO is the case with folds of one point. Typical choices are 5 or 10 folds (5 folds ⇔ 20% validation ⇔ rule of thumb N/5). Notation warning: on slides 7–10 k is the **size of the validation set**; in "k-fold", k usually denotes the **number of folds**.

### L6.Q18 – How is cross-validation used in practice?
*Source: Lecture 6, slide 13*

Given 𝒟_N (N large):
1. Find a model ŵ using **all** the data points ⇒ the **product** (what you deliver).
2. Perform LOO CV and obtain E_cv(ŵ⁻).
3. Since N is large, 𝔼_𝒟[E_cv(ŵ⁻)] ≈ E_out(ŵ) ⇒ E_cv is a **certificate of performance** of the product.
4. Use E_cv as a **criterion for model selection**: compute E_cv for each candidate ℋ (linear, quadratic, …, m-th order polynomial) and choose the minimum (quadratic in the figure).

The E_cv-vs-complexity curve is a measured estimate of the U-shaped 𝔼[E_out] curve of Lecture 5, slide 9, which in Lecture 5 was only theoretical.

### L6.Q19 – How can cross-validation select λ? What is a hyperparameter?
*Source: Lecture 6, slide 14*

In E_aug(w) = E_in(w) + (λ/N)‖w‖₂², λ is a **hyperparameter**: a quantity fixed before training that determines the problem to be solved (like m, or the step size η). The **parameters** w are learned by minimizing E_aug on the training data; λ cannot be learned that way (minimizing over λ as well would always give λ = 0, the smallest E_in). So compute E_cv(λ) for a grid of values and choose **λ_opt** at the minimum. E_cv(λ) is U-shaped: λ too small ⇒ overfitting (variance), λ too large ⇒ underfitting (bias).

### L6.Q20 – Why is the validation estimate optimistic after selecting hyperparameters? What is the three-way split? ★
*Source: Lecture 6, slide 14; Lecture 1, slide 14*

If 𝒟_val is used to select one or more hyperparameters, the estimate of E_out made with 𝒟_val is **optimistic**: λ was chosen precisely to minimize the error on that 𝒟_val, so the chosen value also benefits from the lucky noise of that particular set. It is the same mechanism as the factor M in Lecture 1, slide 14: choosing the best of M candidates on the same data breaks the "fixed hypothesis" assumption. Solution: split 𝒟_N in three:
- **training** (≈ 70%): fit ŵ for each candidate;
- **validation** (≈ 15%): select λ, m, …;
- **test** (≈ 15%): estimate E_out once, at the end. It must never be used for any decision, so it is again a fresh sample for one fixed hypothesis (Hoeffding with M = 1).

### L6.Q21 – (Extra) Worked example: choose between a constant and a line with LOO.
*Source: Lecture 6, slides 11–13 (extra)*

Points (0, 1), (1, 2), (2, 4).
- **Constant** (prediction = mean of the training y's): leave out (0, 1) ⇒ mean of 2 and 4 = 3 ⇒ error (1 − 3)² = 4; leave out (1, 2) ⇒ 2.5 ⇒ 0.25; leave out (2, 4) ⇒ 1.5 ⇒ 6.25. **E_cv = 10.5/3 = 3.5**.
- **Line** (through the two training points): leave out (0, 1) ⇒ y = 2x ⇒ predicts 0 ⇒ error 1; leave out (1, 2) ⇒ y = 1 + 1.5x ⇒ 2.5 ⇒ 0.25; leave out (2, 4) ⇒ y = 1 + x ⇒ 3 ⇒ 1. **E_cv = 2.25/3 = 0.75**.

The line is selected. Compare with the training errors: E_in(line) = 0.056, E_in(constant) = 1.556. E_in is much more optimistic than E_cv (0.056 vs 0.75), because each LOO model is evaluated on a point it has never seen.

### L6.Q22 – (Extra) Compare regularization and cross-validation.
*Source: Lecture 6, slides 3–14 (extra)*

| | Regularization | Cross-validation |
|---|---|---|
| Idea | change the cost, same data | same cost, change how the data are used |
| How E_out is mimicked | E_aug = E_in + (λ/N)‖w‖² | error on held-out data |
| Result | a model with controlled complexity | an estimate of E_out, used as selection criterion |
| Needs | a value of λ | k (or the number of folds), extra trainings |
| Together | λ is chosen by cross-validation (slide 14) | |

They are complementary: regularization gives a continuous knob (λ) to control complexity, cross-validation tells you where to set it.

---

## Lecture 7 – Nonlinear modeling (feature transforms and neural networks)

### Key formulas
- Nonlinear boundary example: x₁² + x₂² = r², classifier ŷ = sign(r² − x₁² − x₂²)
- ★ Feature transform: z = Φ(x) = [r², x₁², x₂²]ᵀ ⇒ ŷ = h(Φ(x)) = sign(z₀ − z₁ − z₂) = sign(wᵀz), w = [1, −1, −1]ᵀ (linear in z)
- Classifier in the original space: ŷ = sign(wᵀΦ(x))
- Logic with perceptrons (inputs ±1, x₀ = 1): OR(x₁, x₂) = sign(x₁ + x₂ + 1), w = [1, 1, 1]; AND(x₁, x₂) = sign(x₁ + x₂ − 1), w = [−1, 1, 1]; NOT: multiply by −1; AND of K inputs: sign(z₁ + … + z_K − (K − 1)) (extra)
- XOR-type target: f(x) = [h₁ AND h̄₂] OR [h̄₁ AND h₂]
- ★ Neuron: h(x) = θ(wᵀx); activation θ = sign (perceptron) or θ(s) = tanh(s)
- tanh(s) = (eˢ − e⁻ˢ)/(eˢ + e⁻ˢ) = 2·σ(2s) − 1 ∈ (−1, 1); derivative 1 − tanh²(s) (extra)
- ★ Training: SGD w(t+1) = w(t) − η∇e(w(t)), e(w) = (y⁽ⁱ⁾ − h(x⁽ⁱ⁾; w))², i random point; ∇e computed by backpropagation
- ★ Universal approximation (Cybenko): ∀ε > 0 ∃ a FF-NN h with 1 hidden layer such that |f(x) − h(x)| < ε ∀x ∈ X, if f is continuous on X

### L7.Q1 – Why are linear models not always sufficient? Give the circle example.
*Source: Lecture 7, slide 2*

In many applications, linear models are not sufficient. Example: class +1 inside a circle of radius r, class −1 outside. No hyperplane (line) separates them, but a separating surface that is **not** a hyperplane exists: the circle x₁² + x₂² = r². A good classifier is ŷ = sign(r² − x₁² − x₂²), which is a **nonlinear** function of x₁, x₂ (the sad face on the slide: our learning tools so far, PLA, OLS, logistic regression, only handle wᵀx).

### L7.Q2 – How does a change of coordinates make the problem linear? ★
*Source: Lecture 7, slide 3*

Define z₀ ≜ r², z₁ ≜ x₁², z₂ ≜ x₂², i.e. z = Φ(x) = [r², x₁², x₂²]ᵀ. Then ŷ = h(Φ(x)) = h(z) = sign(z₀ − z₁ − z₂) = sign(wᵀz) with w = [1, −1, −1]ᵀ: the classifier is **linear in z**. The nonlinearity is moved into the features; the model stays linear in the parameters (the same idea as "linear means linear in w", Lecture 2, slide 3). Example (r = 1): x = (0.5, 0.5) ⇒ z = [1, 0.25, 0.25] ⇒ wᵀz = 0.5 > 0 ⇒ +1 (inside); x = (1, 1) ⇒ z = [1, 1, 1] ⇒ wᵀz = −1 ⇒ −1 (outside).

### L7.Q3 – Describe the full procedure of the feature-transform approach.
*Source: Lecture 7, slide 4*

1. Apply the nonlinear transform z = Φ(x): in the (z₁, z₂) = (x₁², x₂²) plane the inner points are close to the origin and the outer ones far away, so the data become **linearly separable**.
2. Run a linear method in z-space (e.g. **PLA**, Lecture 3): it finds the line w₀z₀ + w₁z₁ + w₂z₂ = 0.
3. Go back to x-space (x = Φ⁻¹(z) on the slide): the line becomes the circle, and the classifier is **ŷ = sign(wᵀΦ(x))**.

Note: Φ is not really invertible (x₁² loses the sign of x₁). "Going back" means that the boundary {x : wᵀΦ(x) = 0} is drawn in the original space; the classifier never needs Φ⁻¹, because it only computes wᵀΦ(x).

### L7.Q4 – What is the key issue of feature transforms? Why must Φ not be selected by looking at the data? ★
*Source: Lecture 7, slide 5*

The key issue is the **knowledge of Φ(x)**:
- Φ(x) **must not be selected by looking at the data** (generalization problem). If you look at the data and choose the transform that separates them, the search over transforms has been done in your head: the effective hypothesis set is much larger than the final ℋ, so the generalization guarantees (the M factor of Lecture 1, O(m/N) of Lecture 2) no longer hold, and the measured E_in is optimistic (extra: "data snooping", the same mechanism as the optimistic validation error of Lecture 6, slide 14).
- Φ(x) **could be available from prior knowledge of the physics** (e.g. the obstacle shape in the Jan 2026 exam is known: then Φ = [1, x₁², x₂²] is legitimate).
- If there is no prior knowledge, Φ(x) **must be learnt from the data**, analogously to λ and the other hyperparameters (Lecture 6) ⇒ **neural networks do the job**.

Extra: the price of hand-made features is complexity: polynomial features of degree d in n inputs give many parameters, so the gap O(m/N) grows (Lecture 2) and so does the variance (Lecture 5).

### L7.Q5 – What are neural networks, and what is their relation with the human brain?
*Source: Lecture 7, slide 6*

A biological neuron has dendrites (inputs), a cell body (where signals are combined), an axon and synaptic terminals (output to other neurons). The human brain is the **inspiration** behind these modeling tools, **but the human brain ≠ neural networks**. Analogy: airplanes "mimic" birds, but they fly in a very different way. An artificial neuron is simply a weighted sum followed by an activation function.

### L7.Q6 – How can a complex classifier be built from two perceptrons?
*Source: Lecture 7, slide 7*

Two lines split the plane into four regions A, B, C, D with target f = +1, −1, +1, −1 (an XOR-like pattern: not linearly separable). Two perceptrons h₁, h₂ each realize one line. Then **f(x) = [h₁(x) AND h̄₂(x)] OR [h̄₁(x) AND h₂(x)]**, i.e. f = +1 when exactly one of h₁, h₂ is +1. Truth table:

| region | f | h₁ | h₂ | h₁ AND h̄₂ | h̄₁ AND h₂ | OR |
|---|---|---|---|---|---|---|
| A | +1 | +1 | −1 | +1 | −1 | +1 ✓ |
| B | −1 | +1 | +1 | −1 | −1 | −1 ✓ |
| C | +1 | −1 | +1 | −1 | +1 | +1 ✓ |
| D | −1 | −1 | −1 | −1 | −1 | −1 ✓ |

Extra (why one perceptron cannot do it): with ±1 inputs, XOR needs w₀ + w₁ + w₂ < 0 and w₀ − w₁ − w₂ < 0 (equal inputs ⇒ −1), and w₀ + w₁ − w₂ > 0 and w₀ − w₁ + w₂ > 0 (different inputs ⇒ +1). Summing the first two gives w₀ < 0, summing the last two gives w₀ > 0: contradiction.

### L7.Q7 – How are logical operations implemented with perceptrons? ★
*Source: Lecture 7, slides 8–9*

"Good news": perceptrons can be combined with logical operations. "Very very good news": logical operations can themselves be implemented by perceptrons (inputs ±1, x₀ = 1):
- **OR(x₁, x₂) = sign(x₁ + x₂ + 1)**, w = [1, 1, 1]: inputs (−1, −1) ⇒ −1; (−1, +1) ⇒ +1; (+1, −1) ⇒ +1; (+1, +1) ⇒ 3 ⇒ +1.
- **AND(x₁, x₂) = sign(x₁ + x₂ − 1)**, w = [−1, 1, 1]: (−1, −1) ⇒ −3; (−1, +1) ⇒ −1; (+1, −1) ⇒ −1; (+1, +1) ⇒ +1.
- **NOT**: multiply the input by −1 (a weight −1).

The bias decides how many +1 inputs are needed: OR needs at least one, AND needs both. Extra: AND of K inputs = sign(z₁ + … + z_K − (K − 1)).

### L7.Q8 – What are the computational graph and the graph representation of a neuron?
*Source: Lecture 7, slide 9*

Computational graph: inputs 1, x₁, x₂ multiplied by weights w₀, w₁, w₂, summed into the score s = wᵀx, then passed through the activation (sign): h(x) = ŷ. Graph representation: the same thing drawn as one node, the **"neuron"**, with weighted incoming edges and the activation drawn inside. Every perceptron of Lecture 3 is one neuron.

### L7.Q9 – Describe the multilayer perceptron that implements the XOR-type classifier. ★
*Source: Lecture 7, slide 10*

- **Input layer:** 1, x₁, x₂.
- **First hidden layer:** two neurons h₁, h₂ with weights w₀₁, w₁₁, w₂₁ and w₀₂, w₁₂, w₂₂ (w_ij = weight from input i to neuron j): the two lines.
- **Second hidden layer:** h̄₁ AND h₂ = sign(−1 − h₁ + h₂) and h₁ AND h̄₂ = sign(−1 + h₁ − h₂).
- **Output layer:** OR = sign(1 + z₁ + z₂) ⇒ h(x).

This is a **multilayer perceptron (MLP)**: input layer, hidden layer(s), output layer. Numerical check with h₁ = sign(x₁ + x₂), h₂ = sign(x₁ − x₂): x = (0, 2) ⇒ h₁ = +1, h₂ = −1 ⇒ z₁ = sign(−3) = −1, z₂ = sign(1) = +1 ⇒ OR = sign(1) = +1; x = (2, 0) ⇒ h₁ = h₂ = +1 ⇒ z₁ = z₂ = −1 ⇒ OR = sign(−1) = −1.

### L7.Q10 – How complex can the decision regions of a network of perceptrons be?
*Source: Lecture 7, slide 11*

By increasing the number of perceptrons we can obtain a surface of **arbitrary complexity**: two lines give the X-shaped regions; four lines combined with AND give a rectangle (intersection of four half-planes); many lines combined with AND approximate a circle with a polygon; OR of several such regions gives unions of shapes. More perceptrons ⇒ more flexible ℋ (lower bias, higher variance, Lecture 5).

### L7.Q11 – What is a feed-forward neural network? What is an activation function?
*Source: Lecture 7, slide 12*

A feed-forward neural network (FF-NN) has an input layer (1, x₁, …, x_m), one or more hidden layers of neurons, and an output layer producing h(x). Each neuron computes θ(weighted sum of the previous layer + bias); **θ is the activation function**. "Feed-forward": information flows only from input to output, with no loops (extra: networks with loops are called recurrent).

### L7.Q12 – Which activation is usually used, and why not sign? ★
*Source: Lecture 7, slide 13*

Usually **θ(s) = tanh(s)**: a "sigmoidal" function like the logistic function, going from −1 to +1. Major advantage over sign(s): **tanh is differentiable**, so the network can be trained with (stochastic) gradient descent. The sign function has zero derivative almost everywhere (the "staircase" of Lecture 3), so gradients give no information. Extra: tanh(s) = 2σ(2s) − 1 (a rescaled logistic function, range (−1, 1) matching ±1 labels); tanh′(s) = 1 − tanh²(s); for large |s| it approximates sign(s).

### L7.Q13 – How is a neural network trained? What is backpropagation? ★
*Source: Lecture 7, slide 14; Lecture 4, slides 26–27*

With SGD: **w(t+1) = w(t) − η∇e(w(t))**, where for mean-squared-error minimization E_in(w) = (1/N) Σ (y⁽ⁱ⁾ − h(x⁽ⁱ⁾, w))² and e(w(t)) = (y⁽ⁱ⁾ − h(x⁽ⁱ⁾, w(t)))² is the error on a random point i. The gradient ∇e(w(t)) is computed by **backpropagation**. The plot shows E_in versus t, measured in **epochs** (one epoch = one pass over all N points): noisy but decreasing.

Backpropagation (extra, the idea): the chain rule applied layer by layer. Forward pass: compute all scores s and outputs θ(s) from input to output. Backward pass: compute δ = ∂e/∂s for the output neuron, then propagate it backwards: each hidden neuron gets δ_j = θ′(s_j)·Σ_k w_jk δ_k; the gradient of every weight is ∂e/∂w_ij = (input of the weight)·δ_j. All gradients cost about as much as one forward pass. E_in of a network is **not convex**: local minima exist, so initialization matters (Lecture 4, slide 14).

### L7.Q14 – (Extra) Work one SGD/backpropagation step on a tiny network.
*Source: Lecture 7, slides 13–14 (extra)*

Network: â = tanh(w₁x), ŷ = w₂â (one tanh hidden neuron, linear output), e = (y − ŷ)². Data point x = 1, y = 1; w₁ = 0.5, w₂ = 1, η = 0.1.
- Forward: s = 0.5, â = tanh(0.5) = 0.462, ŷ = 0.462, e = (0.538)² = 0.289.
- Backward: ∂e/∂ŷ = −2(y − ŷ) = −1.076; ∂e/∂w₂ = −1.076·â = −0.497; ∂e/∂s = −1.076·w₂·(1 − â²) = −0.846; ∂e/∂w₁ = −0.846·x = −0.846.
- Update: w₂ = 1 + 0.1·0.497 = 1.050; w₁ = 0.5 + 0.1·0.846 = 0.585.
- Check: new ŷ = 1.050·tanh(0.585) = 0.552, e = 0.201 < 0.289.

### L7.Q15 – State the universal approximation theorem and discuss its limits. ★
*Source: Lecture 7, slide 15*

Theorem (Cybenko): **for every ε > 0 there exists a FF-NN h(x) with one hidden layer such that |f(x) − h(x)| < ε for all x ∈ X, if f(x) is continuous in X** (extra: X bounded and closed, sigmoidal activation). A network with one hidden layer can approximate any continuous function as accurately as desired. Limits (extra, important at the oral): it is an **existence** result; it does not say how many neurons are needed (possibly very many), how to find the weights (training is non-convex), or how much data are needed. It is a statement about **bias** (ℋ can get arbitrarily close to f), not about generalization: a huge network has low bias but high variance with finite N (Lecture 5, slide 8, case B).

### L7.Q16 – Shallow vs deep networks: why use more than one layer if one is enough?
*Source: Lecture 7, slide 16*

A network with one (hidden) layer is a **shallow NN**; with more than one layer it is a **deep NN**; the science of learning deep NNs is **deep learning**. Why more than one layer given Cybenko's theorem? Example, image classification: the 1st layer detects edges, the 2nd combines edges into shapes, further layers combine shapes into objects, and the output answers "is there a cat in the image?". For a different object (e.g. "dog") a part of this network can be **re-used**: deep NNs are sometimes better for **transfer learning** and for **interpretation** of the results. Extra: deep networks can represent some functions with far fewer neurons than a shallow one, which means fewer parameters to learn.

### L7.Q17 – (Extra) Why do hidden layers need a nonlinear activation? ★
*Source: Lecture 7, slides 12–13 (extra); exams Nov 2022 Ex. 2, Sep 2023 Ex. 2*

If all activations are linear, each layer computes a linear function of its inputs, and a composition of linear maps is linear: W₂(W₁x) = (W₂W₁)x. The whole network collapses to a single linear model (one layer), so it cannot represent XOR, circles or any nonlinear boundary. The nonlinear activation (tanh) is what gives depth its power.

### L7.Q18 – (Extra) How does Lecture 7 connect to the previous lectures?
*Source: Lecture 7 (extra)*

- Lecture 2: "linear in the parameters": feature transforms keep the model linear in w.
- Lecture 3: each neuron with sign activation is a perceptron; PLA is used in z-space; XOR is not linearly separable.
- Lecture 4: the output neuron with logistic activation is logistic regression on learned features; training uses SGD; tanh is a rescaled logistic function; non-convexity makes initialization matter.
- Lecture 5: more neurons/layers ⇒ lower bias, higher variance; universal approximation is a bias statement.
- Lecture 6: the architecture (layers, neurons), λ (weight decay) and η are hyperparameters chosen by cross-validation; Φ must not be chosen by looking at the data (same reason as the test set).

---

## Lecture 8 – Nonparametric classification (k-NN and classifier quality)

### Key formulas
- Parametric: h(x) = h(x, w), ŵ = argmin_w (1/N) Σ (y⁽ⁱ⁾ − wᵀx⁽ⁱ⁾)²; nonparametric: no structure assumed for h(x), only ŷ* is produced
- ★ Distances: Euclidean d(x, x*) = ‖x − x*‖; weighted Euclidean d(x, x*) = (x − x*)ᵀQ(x − x*), e.g. Q = diag(1/1000, 1) ⇒ d = (1/1000)(x₁ − x₁*)² + (x₂ − x₂*)²; cosine similarity cos(x, x*) = xᵀx*/(‖x‖·‖x*‖)
- Sorting: d(x*, x̄⁽¹⁾) ≤ d(x*, x̄⁽²⁾) ≤ … ≤ d(x*, x̄⁽ᴺ⁾)
- Nearest neighbor (1-NN): ŷ* = sign(y(x̄⁽¹⁾))
- ★ k-NN: **ŷ* = sign(Σ_{i=1}^{k} y(x̄⁽ⁱ⁾))**
- ★ Consistency: N → ∞, k → ∞, k/N → 0 ⇒ E_in(h) → E_out(h) and E_out(h) → minimum achievable error
- ★ Rule of thumb: k = ⌊√N⌋
- Cost: memory O(N·m); computation O(N·m + N·log k) per prediction
- Confusion matrix (rows = predicted +1/−1, columns = real +1/−1): C = [a b; c d], a = true positives, b = false positives, c = false negatives, d = true negatives
- ★ Precision PPV = a/(a + b); recall TPR = a/(a + c); F1 = 2·PPV·TPR/(PPV + TPR); accuracy = (a + d)/(a + b + c + d) (extra)

### L8.Q1 – What does the title picture show?
*Source: Lecture 8, slide 1*

A new point "?" (green) among blue squares and red triangles. With the 3 nearest points (solid circle: 2 triangles, 1 square) it is classified as a triangle; with the 5 nearest (dashed circle: 3 squares, 2 triangles) it is classified as a square. The prediction depends only on the neighbours, and it changes with k: the choice of k is the critical issue of the lecture.

### L8.Q2 – Which three solutions exist for non-linearly separable data?
*Source: Lecture 8, slide 2*

1. **Nonlinear transforms + perceptron:** z = Φ(x), ŷ = sign(wᵀz) = sign(wᵀΦ(x)); but Φ is unknown (Lecture 7, slides 3–5).
2. **Neural networks:** learn Φ(x) and w together (Lecture 7).
3. **Nonparametric prediction:** do not build a function at all; predict directly from the data points (this lecture).

### L8.Q3 – What is the difference between parametric and nonparametric modeling? ★
*Source: Lecture 8, slide 3*

**Parametric:** we assume h(x) is fully defined by a set of parameters w, i.e. h(x) = h(x, w). Learning a function, ĥ = argmin_{h∈ℋ} (1/N) Σ (y⁽ⁱ⁾ − h(x⁽ⁱ⁾))², becomes learning a vector, ŵ = argmin_w (1/N) Σ (y⁽ⁱ⁾ − wᵀx⁽ⁱ⁾)², w = [w₀, …, w_m]. **Nonparametric:** we do **not** assume any structure for h(x); we look for classification rules based on other criteria (e.g. geometrical, like distances). It is a good choice if we are only interested in the prediction ŷ = f(x) and do not look for a model h of f(x). Extra: "nonparametric" does not mean "no numbers to store"; it means that the number of quantities the predictor uses is not fixed in advance and grows with N (k-NN uses the whole dataset).

### L8.Q4 – Compare the two approaches with block diagrams.
*Source: Lecture 8, slide 4*

Parametric: 𝒟_N → learning machine → ĥ(x) → ŷ* = ĥ(x*); the outputs are a model ĥ and the predictions ŷ. Nonparametric: 𝒟_N (with the new point x*) → learning machine → ŷ*; the only output is the prediction, there is no ĥ to inspect or reuse.

### L8.Q5 – What is the nearest-neighbor rule?
*Source: Lecture 8, slide 5*

Binary classification with 𝒟_N = {(x⁽¹⁾, y⁽¹⁾), …, (x⁽ᴺ⁾, y⁽ᴺ⁾)}. **Objective:** find y* as the label of a new point x* (y* = f(x*)). **Key idea:** assign the label of the closest training point. Warning on the slide: it is sometimes abbreviated "NN", but NN usually means neural networks.

### L8.Q6 – Which notions of distance can be used? What is the role of Q? ★
*Source: Lecture 8, slide 6*

- **Euclidean distance:** d(x, x*) = ‖x − x*‖ (the length of the difference vector).
- **Weighted Euclidean distance:** d(x, x*) = (x − x*)ᵀQ(x − x*), with Q positive definite (Lecture 2 definition). Example Q = diag(1/1000, 1): d = (1/1000)(x₁ − x₁*)² + (x₂ − x₂*)², i.e. a difference in x₁ counts 1000 times less than the same difference in x₂.
- **Cosine similarity:** cos(x, x*) = xᵀx*/(‖x‖·‖x*‖), the cosine of the angle between the two vectors.

Notes (extra): the weighted form is a squared distance, which gives the same ranking of neighbours. Cosine is a **similarity**: larger means closer, so the nearest neighbours are those with the largest cosine (or use 1 − cos as a distance); it ignores the length of the vectors. Q is how **feature scaling** is done: with features in different units (grams vs a 0–1 intensity), the plain Euclidean distance is dominated by the feature with the largest range; choosing Q = diag(1/σ₁², 1/σ₂², …) (standardization) gives each feature a comparable weight.

### L8.Q7 – Describe the nearest-neighbor algorithm.
*Source: Lecture 8, slide 7*

Inputs: (𝒟_N, d(·,·), x*).
1. Sort the training points according to their distance from x*: from the original numbering x⁽¹⁾, …, x⁽ᴺ⁾ to x̄⁽¹⁾, …, x̄⁽ᴺ⁾ with d(x*, x̄⁽¹⁾) ≤ d(x*, x̄⁽²⁾) ≤ … ≤ d(x*, x̄⁽ᴺ⁾).
2. Predict ŷ = sign(y(x̄⁽¹⁾)), the label associated with the closest point x̄⁽¹⁾.

### L8.Q8 – What is the Voronoi tessellation and what does it say about 1-NN?
*Source: Lecture 8, slide 8*

"We are intrinsically modeling the data": every training point owns a **cell**, the set of all points of the plane closer to it than to any other training point. The cells form the **Voronoi tessellation** of the input space. 1-NN assigns to each cell the label of its point, so the decision boundary is made of the cell edges between points of different classes: piecewise linear, as irregular as the data. The model is the dataset itself. Extra: 1-NN always has E_in = 0 (each training point is its own nearest neighbour), so the training error says nothing about its quality.

### L8.Q9 – Describe k-nearest neighbors. ★
*Source: Lecture 8, slide 9*

Generalization of the nearest-neighbor algorithm (also called 1-NN). **Key idea:** assign the label of the **majority** of the k closest points: **ŷ* = sign(Σ_{i=1}^{k} y(x̄⁽ⁱ⁾))**, where y(x̄⁽ⁱ⁾) = ±1 is the label of the i-th closest point; the sum counts (+1 votes) − (−1 votes). Example on the slide, k = 3: two "o" and one "×" ⇒ "o". Extra: in binary classification use **odd k** to avoid ties (sum = 0).

### L8.Q10 – What happens in the limit cases k = 1 and k = N? ★
*Source: Lecture 8, slide 10*

- **k = 1:** k-NN returns the same class as the classical nearest neighbor (Voronoi boundary, very irregular).
- **k = N:** every new point x* gets the class of the **majority of the whole dataset**, independently of its position in the input space: a constant classifier.

Extra, bias-variance (Lecture 5): small k ⇒ flexible, jagged boundary, low bias, high variance (a single noisy point changes the prediction); large k ⇒ smooth boundary, low variance, high bias; k = N is the extreme "too simple" model. The complexity of k-NN **decreases** as k grows (roughly N/k effective parameters).

### L8.Q11 – State the consistency theorem and the rule of thumb for k. ★
*Source: Lecture 8, slide 11*

"Solution of the k-selection problem is the most critical task!" **Theorem:** as N → +∞, if k → +∞ but k/N → 0, then (a) E_in(h) → E_out(h) and (b) E_out(h) is the **minimum achievable error**. **Rule of thumb:** k = k(N) = ⌊√N⌋ (⌊·⌋ = largest integer below the argument). √N satisfies both conditions: k → ∞ and k/N = 1/√N → 0.

Meaning (extra): k → ∞ means that each prediction averages more and more neighbours (variance → 0); k/N → 0 means that the neighbours occupy a smaller and smaller region around x* (they stay local, bias → 0). The "minimum achievable error" is the error of the best possible classifier for noisy labels (Bayes error, Lecture 4 noisy target): with y ~ P(y|x), even the ideal rule errs with probability min(P(+1|x), P(−1|x)). In practice k is a hyperparameter chosen by cross-validation (Lecture 6); LOO is cheap for k-NN because there is no training.

### L8.Q12 – What are the computational drawbacks of k-NN?
*Source: Lecture 8, slide 12*

- **No training:** there is no h(x) to be learned (all the work is postponed to prediction time).
- **High inference cost:**
  - **memory O(N·m):** the whole dataset must be stored, N points of m numbers each (m = size of a stored sample);
  - **computation O(N·m + N·log k)** per prediction: N distances in m dimensions (O(N·m)), plus keeping the k smallest among N (O(N·log k), e.g. with a heap; extra).

A parametric model, once trained, stores only w and predicts with one product wᵀx (O(m)).

### L8.Q13 – Summarize the classification rules seen in the course.
*Source: Lecture 8, slide 13*

- **Perceptron** (possibly with nonlinear transforms): ŷ = sign(wᵀx) or sign(wᵀΦ(x)).
- **Logistic regression:** s = wᵀx → h(s) = logistic → threshold: ŷ = +1 if h(s) > ε, −1 otherwise (ε = 0.5 by default).
- **Neural networks:** layers of tanh neurons, sigmoid/sign output.
- **k-NN:** majority vote of the k nearest neighbours.

The first three are parametric; k-NN is nonparametric.

### L8.Q14 – How can the quality of a classifier be assessed? Define the confusion matrix. ★
*Source: Lecture 8, slide 14*

Assessment is needed e.g. to choose the best classifier for a problem.
1. **Misclassification error:** E_in on the training set, or an estimate of E_out on a validation set (Lecture 6).
2. **Confusion matrix:** C = [a b; c d], rows = **predicted** class (+1, −1), columns = **real** class (+1, −1): a = number of true positives, b = false positives (predicted +1, really −1), c = false negatives (predicted −1, really +1), d = true negatives. The misclassification error is (b + c)/(a + b + c + d); the confusion matrix also says **which kind** of error is made.

### L8.Q15 – Define precision, recall and F1-score. When are they needed instead of accuracy? ★
*Source: Lecture 8, slide 15*

- **Precision** (positive predicted value): **PPV = a/(a + b)**: among the points predicted positive, the fraction that really is positive.
- **Recall** (true positive rate): **TPR = a/(a + c)**: among the real positives, the fraction that is detected.
- **F1-score:** **F1 = 2·PPV·TPR/(PPV + TPR)**, the harmonic mean of the two: it is high only if both are high, and 0 if either is 0.

They are needed when the classes are **unbalanced** or when the two kinds of error have **different costs**. Example: 1000 people, 10 infected; a test that answers "negative" for everyone has a = 0, b = 0, c = 10, d = 990: accuracy 99%, recall 0 (useless). A more realistic test with a = 8, b = 40, c = 2, d = 950: accuracy 95.8%, precision 8/48 = 0.17, recall 8/10 = 0.8, F1 = 0.28. Trade-off (link with Lecture 4, threshold): raising the threshold ε predicts +1 less often ⇒ fewer false positives (precision ↑) but more false negatives (recall ↓).

### L8.Q16 – (Extra) Pros and cons of nonparametric vs parametric classification.
*Source: Lecture 8, slides 3–4 and 8–12 (extra)*

| | Parametric (perceptron, logistic, NN) | Nonparametric (k-NN) |
|---|---|---|
| Assumptions | a structure h(x, w) (may be wrong ⇒ bias) | none: flexible boundaries of any shape |
| Training | optimization needed | none |
| Prediction cost | low (store w, compute wᵀx) | high: O(N·m) memory and time |
| Model | explicit ĥ, interpretable, reusable | no model: only predictions |
| Data needed | fewer, if the structure is right | many (consistency only as N → ∞) |
| Sensitive to | the choice of ℋ | k, the distance metric, feature scaling, irrelevant features, high dimension |

Extra, curse of dimensionality: in many dimensions all points are far from each other, so the "nearest" neighbours are no longer near and the data needed grow exponentially with the dimension.

### L8.Q17 – (Extra) How does Lecture 8 connect to the previous lectures?
*Source: Lecture 8 (extra)*

- Lecture 1: k-NN still relies on the probabilistic view (data i.i.d., noisy targets); E_in vs E_out again (slide 11).
- Lecture 3: the perceptron is the parametric linear classifier k-NN is compared with; sign(sum of ±1 votes) is a majority vote.
- Lecture 4: the logistic threshold ε controls the precision/recall trade-off; the "minimum achievable error" is the error left by noisy labels.
- Lecture 5: k controls bias and variance (k = 1 high variance, k = N maximum bias).
- Lecture 6: k is a hyperparameter chosen by cross-validation (E_in is useless: 1-NN gives E_in = 0); the misclassification error on a validation set estimates E_out.
- Lecture 7: k-NN is the third answer to non-linearly separable data, after feature transforms and neural networks.

---

## Lecture 9 – Clustering (k-means and hierarchical clustering)

### Key formulas
- Data without labels: x⁽¹⁾, …, x⁽ᴺ⁾ ∈ ℝᵈ; k clusters S₁, …, S_k (a partition of the N points), N_j = number of points in S_j
- ★ Centroid of a cluster: **μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾**
- Error of one cluster: **E_j = Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖²** (a cluster is "good" if E_j is small)
- ★ k-means objective: **E_in = Σ_{j=1}^{k} E_j = Σ_{j=1}^{k} Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖²**, to be minimized over the partition and the centroids: NP-hard
- ★ Lloyd's algorithm (k given): (1) initialize μ₁, …, μ_k; (2) assignment: S_j = all points closest to μ_j; (3) update: μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾; (4) repeat (2)–(3) until E_in stops decreasing. Steps (2) and (3) cannot increase E_in
- Why the mean (extra): ∇_μ Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ‖² = −2 Σ_{i∈S_j} (x⁽ⁱ⁾ − μ) = 0 ⇒ μ = (1/N_j) Σ x⁽ⁱ⁾; equivalently Σ‖x⁽ⁱ⁾ − μ‖² = Σ‖x⁽ⁱ⁾ − μ_j‖² + N_j‖μ_j − μ‖²
- Initialization (farthest point): μ₁⁽⁰⁾ = a random data point; μ₂⁽⁰⁾ = the data point farthest from μ₁⁽⁰⁾; and so on (each new centroid = the point farthest from the centroids already chosen)
- ★ Choice of k: the optimal E_in(k) decreases with k and is 0 for k = N ⇒ choose k at the **knee** of the E_in(k) curve
- ★ Hierarchical clustering: start from N clusters, merge the two closest clusters at each step, until 1 cluster; the **dendrogram** shows E_in for every k (0 at k = N, maximum at k = 1)
- Extra (Ward merge cost): merging clusters A and B increases E_in by N_A·N_B/(N_A + N_B)·‖μ_A − μ_B‖²
- Extra (cost): one Lloyd iteration O(N·k·d); hierarchical clustering needs all pairwise distances, O(N²) memory and at least O(N²) time

### L9.Q1 – What does the title picture show?
*Source: Lecture 9, slide 1*

A cloud of points in 3 dimensions, coloured by group: the points are not labelled by a teacher, but they clearly form separate "clouds" (green, yellow, orange, red). Clustering is the task of discovering these groups automatically from the inputs alone.

### L9.Q2 – What is the difference between classification and clustering? ★
*Source: Lecture 9, slide 2*

**Classification** (supervised): the data are pairs (x⁽ⁱ⁾, y⁽ⁱ⁾) with labels (x and o in the figure). The goal is to find a partition of the input space so as to **predict the label** associated to any new point x*. **Clustering** (unsupervised): the data are only x⁽¹⁾, …, x⁽ᴺ⁾, without labels. The goal is to find a partition of the input space so as to **divide the N points into k sets S₁, S₂, …, S_k with corresponding centroids μ₁, μ₂, …, μ_k** (in the figure k = 2, S₁ and S₂ separated by the dashed line). Extra consequences: there is no "right answer" y to compare with, so there is no misclassification error; quality is measured by how compact the groups are (E_in of slide 5); the groups have no names, a human must interpret them (see the marketing example).

### L9.Q3 – How can clustering simplify a supervised learning task?
*Source: Lecture 9, slide 3*

Regression problem: learn a model of the target function f(x) from data that are flat for small x and steep for large x. One straight line (green, "linear") fits badly everywhere; a nonlinear model (orange) fits well but is more complex. Alternative: cluster the data into two regions separated by a threshold ε, then fit one linear model per cluster (red line for x ≤ ε, purple line for x > ε): the **piecewise linear model** ŷ = ŵ₁ᵀx if x ≤ ε, ŷ = ŵ₂ᵀx if x > ε. Each piece is an ordinary least-squares problem (Lecture 2), so we get a nonlinear predictor using only linear tools. Extra: to predict a new point we need to know its cluster from x alone, so the clusters are defined (or the boundary ε is read) in the input space; this is a feature transform in the sense of Lecture 7, with indicator features ⟦x ≤ ε⟧.

### L9.Q4 – What is the key idea of k-means? Define centroid and cluster error. ★
*Source: Lecture 9, slide 4*

**Key idea:** the centroids are representative of the data if every point in S_j is close to μ_j. The centroid is the mean of the points of the cluster, **μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾**. A cluster is "good" if its error **E_j = Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖²** (sum of squared Euclidean distances from the centroid) is "small". E_j is N_j times the variance of the cluster: it measures how spread the cluster is.

### L9.Q5 – Why must the red point of slide 4 belong to S_j?
*Source: Lecture 9, slide 4*

Its distance from μ_j is d_j and from μ_{j+1} is d_{j+1}; since d_{j+1} > d_j, it must belong to S_j: each point goes to the cluster of the **nearest centroid** (this choice gives the smallest contribution ‖x⁽ⁱ⁾ − μ_j‖² to the error). Extra: the border between two clusters is the set of points equidistant from the two centroids, i.e. the perpendicular bisector of the segment μ_jμ_{j+1}; so k-means divides the space into the **Voronoi cells of the centroids** (the same geometry as 1-NN in Lecture 8, slide 6, but with k centroids instead of N training points). Consequence: k-means boundaries are straight lines (hyperplanes) and clusters are convex.

### L9.Q6 – What is the ideal clustering problem and why can it not be solved exactly?
*Source: Lecture 9, slide 5*

Ideally we would like to minimize **E_in = Σ_{j=1}^{k} E_j** over all partitions S₁, …, S_k (and centroids). This problem is **NP-hard**: no known algorithm solves it in polynomial time. Extra intuition: the number of ways to split N points into k non-empty groups (Stirling number) is about k^N/k!; for the marketing example (N = 1000, k = 5) it is about 10⁶⁹⁷, impossible to enumerate. Hence we use **Lloyd's algorithm**, a **heuristic rule** with **k given**: fast, but it only guarantees a local minimum. (Same situation as ‖w‖₀ in Lecture 6, slide 4: the exact problem is combinatorial, so we accept an approximate method.)

### L9.Q7 – Describe Lloyd's algorithm. ★
*Source: Lecture 9, slide 6*

Given k:
1. **Initialize** the centroids μ₁, μ₂, …, μ_k (see slide 8).
2. **Assignment:** set S_j = the set of all points closest to μ_j, for all j.
3. **Update:** recompute the centroids μ_j, j = 1, …, k, so that they are the REAL centroids of S_j: **μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾**.
4. **Repeat** (2) and (3) until E_in stops decreasing (equivalently, until the assignments no longer change).

**Remark:** steps (2) and (3) **cannot increase E_in**. It is the "k-means" algorithm used in practice.

### L9.Q8 – (Derivation) Why can steps (2) and (3) not increase E_in? Why does Lloyd's algorithm converge? ★
*Source: Lecture 9, slide 6 (derivation extra)*

E_in depends on two things: the partition and the centroids. Lloyd's algorithm minimizes over one while the other is fixed (alternating or "coordinate" minimization).
- **Step (2), centroids fixed:** E_in = Σ_i ‖x⁽ⁱ⁾ − μ_{c(i)}‖², where c(i) is the cluster of point i. Each term depends only on the choice of c(i), and is smallest when c(i) is the nearest centroid. So reassigning every point to its nearest centroid cannot increase any term, hence cannot increase E_in.
- **Step (3), clusters fixed:** each E_j depends only on μ_j. Setting the gradient to zero: ∇_μ Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ‖² = −2 Σ_{i∈S_j} (x⁽ⁱ⁾ − μ) = 0 ⇒ **μ = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾** (the Hessian 2N_j·I is positive definite, so it is the minimum: same reasoning as OLS in Lecture 2). Equivalently, Σ‖x⁽ⁱ⁾ − μ‖² = Σ‖x⁽ⁱ⁾ − μ_j‖² + N_j‖μ_j − μ‖² ≥ E_j for any μ. So moving μ_j to the mean cannot increase E_j.
- **Convergence:** E_in ≥ 0 and never increases; there is a finite number of partitions, and a partition cannot repeat once E_in has strictly decreased. Therefore the algorithm stops after a finite number of iterations, at a point where neither step can improve E_in: a **local minimum**, not necessarily the global one.

### L9.Q9 – Describe the example of slide 7.
*Source: Lecture 9, slide 7*

k = 2, points in the (x₁, x₂) plane. Iteration 0 (red): random initial centroids μ₁⁽⁰⁾ (top) and μ₂⁽⁰⁾ (bottom); the assignment gives S₁⁽⁰⁾ and S₂⁽⁰⁾, which mix points of the two visible groups. Iteration 1 (blue): the centroids move to the means μ₁⁽¹⁾, μ₂⁽¹⁾ and the new assignment gives S₁⁽¹⁾ (top-right group) and S₂⁽¹⁾ (bottom-left group). Iteration 2 (green): μ₁⁽²⁾, μ₂⁽²⁾ are the centres of the two natural groups and nothing changes any more. Messages on the slide: **the algorithm converges!**, but **random initialization might not be the best choice**: with an unlucky start Lloyd can stop in a bad local minimum.

### L9.Q10 – (Extra) Show that Lloyd's algorithm can stop in a bad local minimum.
*Source: Lecture 9, slides 7–8 (extra)*

Four points at the corners of a rectangle: A = (0, 0), B = (0, 1), C = (4, 0), D = (4, 1), k = 2. The best clustering is {A, B}, {C, D}, centroids (0, 0.5) and (4, 0.5), E_in = 4·0.5² = **1**. Random initialization with μ₁⁽⁰⁾ = A, μ₂⁽⁰⁾ = B: C is closer to A (4 < √17), D is closer to B ⇒ S₁ = {A, C}, S₂ = {B, D}, centroids (2, 0) and (2, 1); the new assignment does not change anything ⇒ Lloyd stops with E_in = 4·2² = **16**. Of the 6 possible pairs of starting points, AB and CD give this bad solution (probability 1/3). Farthest-point initialization (slide 8): from any corner the farthest point is the opposite corner (√17), and from {A, D} (or {B, C}) Lloyd reaches the optimum E_in = 1 in one step. In practice one also runs Lloyd from several initializations and keeps the solution with the smallest E_in.

### L9.Q11 – How should the centroids be initialized? ★
*Source: Lecture 9, slide 8*

(a) Take one point at random in the dataset → μ₁⁽⁰⁾. (b) Take μ₂⁽⁰⁾ as the point in the dataset **furthest from μ₁⁽⁰⁾**. (c) Proceed as in (b) until all centroids are defined (each new centroid is the point farthest from the centroids already chosen, i.e. the one with the largest distance from its nearest chosen centroid). Why: the initial centroids are spread over the data, one in each far-away region, so it is unlikely that two of them start in the same group (the problem of L9.Q10). Extra weakness: the farthest points are often **outliers**, which can become centroids of one-point clusters; a common remedy (k-means++) picks each new centroid at random with probability proportional to its squared distance from the chosen centroids, and restarts the algorithm several times.

### L9.Q12 – How is k selected when it is not given? ★
*Source: Lecture 9, slide 8*

Run k-means for k = 1, 2, 3, … and plot the final E_in against k. E_in decreases with k (more centroids ⇒ every point is closer to a centroid) and becomes 0 for k = N (every point is its own centroid), so we **cannot choose k by minimizing E_in** (it would give k = N, the same issue as 1-NN with E_in = 0 in Lecture 8 or overfitting in Lecture 5). The curve has a **"knee-type" behaviour**: a fast decrease while k is smaller than the real number of groups (we are splitting real groups apart), then a slow decrease (we only split compact groups). **Best choice of k = the knee.** Right figure: k = 1 (one purple ellipse), k = 2 (two red groups, the natural choice), k = 6 (orange small circles: too many). In the exams it is also called the "elbow" method.

### L9.Q13 – What is hierarchical clustering and what is a dendrogram? ★
*Source: Lecture 9, slide 9*

It is the way **to avoid the selection of k**. Agglomerative version: start with N clusters (one per point, E_in = 0); at each step merge the two closest clusters (in the version consistent with the slide, the pair whose merge increases E_in the least); stop when only one cluster is left (maximum E_in, k = 1). The **dendrogram** is the tree of the merges: the leaves are the points (1, …, 6), each horizontal bar is a merge drawn at the height of the E_in reached, the top is "max E_in (k = 1)". Cutting the tree with a horizontal line gives the clustering for any k: the dendrogram **returns the performance for each k**. The clusters are **nested** (right figure: small orange groups inside red groups inside the blue one). A good cut is just below a long vertical gap (a big jump of E_in means two well-separated groups have been merged): this is the knee seen in another form.

### L9.Q14 – (Extra) Compare k-means and hierarchical clustering. ★
*Source: Lecture 9, slides 6–9 (extra)*

| | k-means (Lloyd) | Hierarchical clustering |
|---|---|---|
| k | must be given (or chosen by the knee) | not needed in advance: all k from one dendrogram |
| Result | one partition | nested partitions, easy to visualize (dendrogram) |
| Initialization | needed, result depends on it | none: deterministic |
| Optimality | local minimum of E_in | greedy: a merge is never undone, not optimal for a given k |
| Cost | O(N·k·d) per iteration: fast, large N | O(N²) memory, ≥ O(N²) time: small or medium N |
| Convergence | guaranteed in a finite number of steps | always ends after N − 1 merges |

Common limits: Euclidean distances ⇒ sensitive to feature scaling and outliers; k-means prefers compact, convex ("round") groups of similar size.

### L9.Q15 – (Extra) Worked example: x = {1, 2, 4, 7, 8, 10}, k = 2.
*Source: Lecture 9, slides 5–9 (extra)*

**Lloyd, random start μ = (1, 2):**
- Assignment (border at 1.5): S₁ = {1}, S₂ = {2, 4, 7, 8, 10}; E_in with the old centroids = 0 + 0 + 4 + 25 + 36 + 64 = 129. Update: μ = (1, 6.2), E_in = 0 + 4.2² + 2.2² + 0.8² + 1.8² + 3.8² = 40.8.
- Assignment (border at 3.6): S₁ = {1, 2}, S₂ = {4, 7, 8, 10}; E_in = 24.16. Update: μ = (1.5, 7.25), E_in = 0.5 + 18.75 = 19.25.
- Assignment (border at 4.375): S₁ = {1, 2, 4}, S₂ = {7, 8, 10}; E_in = 14.94. Update: μ = (2.33, 8.33), E_in = 4.67 + 4.67 = **9.33**.
- Assignment (border at 5.33): nothing changes ⇒ stop. E_in: 129 → 40.8 → 24.16 → 19.25 → 14.94 → 9.33, it never increases.

**Farthest-point start:** μ₁ = 1, the farthest point is 10 ⇒ μ = (1, 10), border 5.5 ⇒ {1, 2, 4}, {7, 8, 10} immediately.

**Knee:** best E_in for k = 1, …, 6: 63.33, 9.33, 5.17, 1.0, 0.5, 0. Huge drop from k = 1 to k = 2, then small drops ⇒ k = 2.

**Dendrogram (merge with the smallest increase of E_in):** {1, 2} (E_in 0.5), {7, 8} (1.0), {1, 2, 4} (5.17), {7, 8, 10} (9.33), all (63.33). The last merge costs 3·3/6·(8.33 − 2.33)² = 54: the long gap says "cut at k = 2".

### L9.Q16 – Describe the marketing segmentation problem and what the data show.
*Source: Lecture 9, slides 10–11*

A Telco company must decide which offers to give to its customers based on their **usage behaviour data**. N = 1000 customers, d = 5 variables: **call** (hours of calls per month), **intern** (hours of international calls per month), **text** (text messages per month), **data** (GB per month), **age**. There are no labels: we want to discover groups of similar customers (segments) and design one offer per segment. First look at the data (slide 11): a scatter-plot matrix (every pair of variables, histograms on the diagonal), coloured by age class (red 10 ≤ age < 30, blue 30 ≤ age < 50, green 50 ≤ age < 75). Observations: old people call less; young people text more; young people do not make international calls; high data usage seems correlated with high text usage; young people call less but use data more.

### L9.Q17 – Describe the results with 8 and with 5 groups.
*Source: Lecture 9, slides 12–13*

**8 groups (hierarchical clustering):** for each group the table shows the mean of each variable and the proportion of customers; green = high value, red = low value in that column. Three groups are clear: **Young adults** (25.4%: few calls 1.7, almost no international 0.10, many texts 509, most data 3.73 GB, age 19), **Silver** (19.5%: lowest usage, age 61), **Pro** (18.4%: most calls 5.07 and international calls 1.01, few texts 26). The other five ("30's", "40's", "40's", "30's", "50's") are similar to each other. **5 groups:** redo the clustering asking for 5 groups: Young adults, Silver and Pro stay identical; four small groups merge into **Heavy users** (27.5%: lots of calls, texts and data); the remaining 40's group becomes **Light users** (9.2%: similar to Silver, but younger, 35.6). Check (extra): Heavy users = 16.1% + 5.4% + 3.4% + 2.6% = 27.5%, and its means are the proportion-weighted means of the four merged groups, e.g. calls (3.1366·16.1 + 1.8552·5.4 + 4.9832·3.4 + 2.9569·2.6)/27.5 = 3.096: the 5-group solution is the 8-group dendrogram cut higher (nested clusters). Note: the Pro "mean data" is 26.085 on slide 12 (the same number as its mean text, a copy error) and 2.0148 on slide 13.

### L9.Q18 – What is a radar plot and what does it show here?
*Source: Lecture 9, slide 14*

One axis per variable (Age, Calls, Intern, Text, Data), each group is a polygon joining its (normalized) means: the shape is the **profile** of the group, so the groups can be compared at a glance. Heavy users: lots of calls, texts and data; Young adults: few calls, lots of texts and data, low age (polygon pulled toward Text and Data); Silver: low usage, seasoned (pulled toward Age only); Pro: lots of calls and data (wide toward Calls and Intern); Light users: similar to Silver, but younger (small polygon). This is the "human" step of clustering: the algorithm returns groups, the analyst names them and decides the offers.

### L9.Q19 – (Extra) Which practical issues must be considered before clustering?
*Source: Lecture 9, slides 4 and 10–13 (extra)*

- **Feature scaling:** k-means uses Euclidean distances, so a variable with large numbers dominates: in the marketing data "text" is in the hundreds while "intern" is between 0 and 1; without standardization (subtract the mean, divide by the standard deviation) the clusters would depend almost only on texts. Same issue as k-NN (Lecture 8, weighted distance Q).
- **Outliers:** a far point pulls its centroid (the mean is sensitive) and can become a cluster of its own.
- **Shape:** k-means finds convex, compact groups; elongated or nested groups are split badly.
- **Choice of k and initialization:** knee, dendrogram, several restarts.
- **Interpretation:** clusters have no names; the means (tables, radar plot) are needed to give them a meaning.

### L9.Q20 – (Extra) k-means vs k-NN: what does "k" mean in each?
*Source: Lecture 9, slides 2 and 6; Lecture 8, slide 7 (extra)*

| | k-NN (Lecture 8) | k-means (Lecture 9) |
|---|---|---|
| Learning type | supervised (needs labels y) | unsupervised (no labels) |
| Task | classify a new point x* | group the N points |
| k | number of neighbours that vote | number of clusters |
| Training | none (store the data) | Lloyd's iterations |
| Output | a label ŷ* | partition S₁, …, S_k and centroids μ₁, …, μ_k |

Common points: both use distances (scaling matters); a new point can be assigned to the nearest centroid, which is 1-NN on the k centroids.

### L9.Q21 – (Extra) How does Lecture 9 connect to the previous lectures?
*Source: Lecture 9 (extra)*

- Lecture 1: so far learning meant a target f with labels y; clustering is **unsupervised learning**: only x is available.
- Lecture 2: the centroid is a least-squares estimate (mean = minimizer of a sum of squares, gradient = 0, positive definite Hessian).
- Lectures 4 and 7: Lloyd's algorithm is an iterative method that never increases the cost and can stop in a local minimum, like gradient descent / backpropagation on a non-convex error; it needs an initialization, a stopping rule and restarts.
- Lecture 5/6: E_in(k) always decreases with k, like E_in with model complexity, so k cannot be chosen by E_in (knee instead of validation); the NP-hard exact problem, like ‖w‖₀.
- Lecture 7: clustering + one linear model per cluster = a piecewise linear (nonlinear) predictor.
- Lecture 8: distances, Voronoi cells, feature scaling; k-means is not k-NN.

---

## Lecture 10 – Data pre-processing (centering, normalization, whitening, cleaning, PCA)

### Key formulas
- Pre-processing as a transform: z = Φ(x), final hypothesis h(z) = h(Φ(x)); the aim is to STANDARDIZE the data, not to model nonlinearities
- ★ Centering: **z⁽ⁱ⁾ = x⁽ⁱ⁾ − x̄**, x̄ = (1/N) Σ_{i=1}^{N} x⁽ⁱ⁾ (empirical input bias)
- ★ Normalization (X centered): **z_j⁽ⁱ⁾ = x_j⁽ⁱ⁾/σ_j**, σ_j² = (1/N) Σ_{i=1}^{N} (x_j⁽ⁱ⁾)² (empirical input variance)
- ★ Whitening (X centered): A = (1/N) Σ_{i=1}^{N} x⁽ⁱ⁾x⁽ⁱ⁾ᵀ (empirical covariance), **z⁽ⁱ⁾ = A^(−1/2) x⁽ⁱ⁾**; if A = QΛQᵀ then A^(−1/2) = QΛ^(−1/2)Qᵀ; result: (1/N) Σ z⁽ⁱ⁾z⁽ⁱ⁾ᵀ = I
- ★ PCA, one direction: z = vᵀx, Var[z] = (1/N) Σ vᵀx⁽ⁱ⁾x⁽ⁱ⁾ᵀv = **vᵀAv**, to be maximized with ‖v‖ = 1
- ★ SVD: **X = U·Γ·Vᵀ**, U ∈ ℝ^{N×(m+1)} with UᵀU = I, V ∈ ℝ^{(m+1)×(m+1)} with VᵀV = VVᵀ = I, Γ = diag(γ₁, …, γ_{m+1}), γ₁ ≥ γ₂ ≥ … ≥ γ_{m+1} ≥ 0 (singular values)
- ★ Link (extra): A = (1/N)XᵀX = V(Γ²/N)Vᵀ ⇒ the columns of V are the principal directions and the variance along v_i is γ_i²/N
- ★ New coordinates: **Z = X·V** (= UΓ); reduced: **Ẑ = X·V_k** (N×k), V_k = first k columns of V ((m+1)×k)
- Extra: share of variance of component i = γ_i²/Σ_j γ_j²; information lost with k components = Σ_{i>k} γ_i² = ‖X − ẐV_kᵀ‖²

### L10.Q1 – Where does data pre-processing sit in the learning pipeline, and which steps does it include?
*Source: Lecture 10, slides 1–2*

Between the dataset and the learning algorithm: f(x) → 𝒟_N → **data pre-processing** → learning algorithm 𝒜 (searching in ℋ) → hypothesis/model. The title picture (kneading dough) says it: the raw data are "worked" into a good shape before learning. Five steps: (1) input centering, (2) input normalization, (3) input whitening, (4) data cleaning, (5) dimensionality reduction.

### L10.Q2 – How can any pre-processing step be formulated? What is the warning?
*Source: Lecture 10, slide 3*

As a generic (possibly nonlinear) transform of X: **z = Φ(x)**, and the final hypothesis is **h(z) = h(Φ(x))**. Same notation as the feature transforms of Lecture 7, but **WARNING:** here the aim is **not** to model some nonlinearities, it is to **STANDARDIZE** the measured data. Extra: the parameters of Φ (x̄, σ_j, A, V) are estimated from the data, so they must be computed on the **training set only** and then applied unchanged to validation, test and new points (z* = Φ(x*)); computing them on all data is data snooping (Lecture 7). Centering and normalization are affine maps: a linear model in z is still a linear model in x, so they do not change the hypothesis set of linear regression; they change the numerics, the distances and the effect of regularization.

### L10.Q3 – Define input centering. ★
*Source: Lecture 10, slide 4*

**Goal:** remove any bias in X by translating the origin. **z⁽ⁱ⁾ = x⁽ⁱ⁾ − x̄, i = 1, …, N**, with **x̄ = (1/N) Σ_{i=1}^{N} x⁽ⁱ⁾** the empirical input bias (vector of the feature means). After centering every feature has zero mean and the cloud is around the origin (slide 6, red). It is required before normalization, whitening and PCA, whose formulas assume centered data. Example: feature values {2, 4, 6} ⇒ x̄ = 4 ⇒ {−2, 0, 2}.

### L10.Q4 – Define input normalization. ★
*Source: Lecture 10, slide 4*

**Goal:** scale the input X (assuming X is centered). Each feature is divided by its standard deviation: **z⁽ⁱ⁾ = [x₁⁽ⁱ⁾/σ₁, …, x_m⁽ⁱ⁾/σ_m]ᵀ**, with **σ_j² = (1/N) Σ_{i=1}^{N} (x_j⁽ⁱ⁾)²** the empirical input variance of feature j. After it, every feature has variance 1 (centering + normalization = standardization, "z-score"). Example: centered {−2, 0, 2} ⇒ σ² = 8/3, σ = 1.633 ⇒ {−1.22, 0, 1.22}. Extra: a common alternative is min-max scaling to [0, 1], x' = (x − min)/(max − min) (used in some exam texts).

### L10.Q5 – (Extra) Why is normalization important? Give examples from the previous lectures.
*Source: Lecture 10, slides 4 and 6 (extra)*

- **Distances (Lectures 8–9):** k-NN and k-means use Euclidean distances, so a feature with large numbers dominates (e.g. texts in the hundreds vs international calls below 1 in the Lecture 9 marketing data).
- **Regularization (Lecture 6):** λ‖w‖² penalizes all weights equally, which is fair only if the features have comparable scales.
- **Gradient descent (Lecture 4):** very different scales give long, thin elliptical contours of E_in, so GD zig-zags and needs a small step; normalized (or whitened) inputs give rounder contours and faster convergence.
- **Neural networks (Lecture 7):** large inputs saturate tanh, where the gradient is almost zero.
- **PCA (this lecture):** without scaling, the feature with the largest variance takes the first component (USArrests: Assault).

### L10.Q6 – Define input whitening. Why does it decorrelate the inputs? ★
*Source: Lecture 10, slide 5*

**Goal:** decorrelate the input samples (assuming X is centered). Empirical covariance matrix **A = (1/N) Σ_{i=1}^{N} x⁽ⁱ⁾x⁽ⁱ⁾ᵀ** (each term is a column times a row, an m×m matrix; diagonal = variances, off-diagonal = covariances). Whitened inputs: **z⁽ⁱ⁾ = A^(−1/2) x⁽ⁱ⁾**. With the eigen-decomposition A = QΛQᵀ (A symmetric, Λ > 0 if A is invertible), A^(−1/2) = QΛ^(−1/2)Qᵀ. **Proof:** (1/N) Σ z⁽ⁱ⁾z⁽ⁱ⁾ᵀ = A^(−1/2) [(1/N) Σ x⁽ⁱ⁾x⁽ⁱ⁾ᵀ] A^(−1/2) = A^(−1/2) A A^(−1/2) = **I**: unit variances and zero correlations. Normalization fixes only the diagonal of A; whitening fixes the whole matrix. Requirement: A invertible (no input is an exact linear combination of the others).

### L10.Q7 – (Extra) Worked example: centering, normalization, whitening and PCA on four points.
*Source: Lecture 10, slides 4–6 and 10–12 (extra)*

Centered points (−3, −3), (−1, 1), (1, −1), (3, 3), N = 4 (e.g. raw data with x̄ = (10, 20), all shifted). **Covariance:** A = ¼[[20, 16], [16, 20]] = [[5, 4], [4, 5]]: variances 5, covariance 4, correlation 0.8. **Normalization:** σ₁ = σ₂ = √5 ⇒ the shape does not change, the correlation stays 0.8. **Eigen/SVD:** λ₁ = 9 along v₁ = (1, 1)/√2, λ₂ = 1 along v₂ = (1, −1)/√2; singular values γ_i = √(Nλ_i) = 6 and 2. **Whitening:** A^(−1/2) = [[2/3, −1/3], [−1/3, 2/3]] ⇒ (3, 3) → (1, 1), (−1, 1) → (−1, 1), (1, −1) → (1, −1), (−3, −3) → (−1, −1): a square with covariance I. **PCA:** scores z₁ = v₁ᵀx = −4.24, 0, 0, 4.24 (variance 9), z₂ = 0, −1.41, 1.41, 0 (variance 1); keeping k = 1 keeps 36/40 = 90% of the variance; the information lost is γ₂² = 4 = sum of the squared distances of the points from the line through v₁.

### L10.Q8 – Describe the example of slide 6.
*Source: Lecture 10, slide 6*

Raw data (blue): an elongated, tilted cloud far from the origin, with mean x̄ (red dot). **Input centering** (red): same cloud translated to the origin. **Input normalization** (purple): the spreads Δz₁ and Δz₂ along the two axes become similar, but the cloud is still diagonal (correlated). **Input whitening** (green): a round cloud, no correlation and the same spread in every direction.

### L10.Q9 – What is data cleaning? Why are outliers dangerous? ★
*Source: Lecture 10, slide 7*

**Goal:** remove **outliers** (points that are wrong or not representative: sensor faults, typing errors, rare events) from the original dataset. The squared error (Lecture 2) grows with the square of the residual, so one far point can rotate the whole least-squares line: in the figure the red line (without outlier) is flat and follows the data, the blue line (with the outlier) is steep and wrong for all points. Numeric example (extra): (1, 1.2), (2, 0.9), (3, 1.0), (4, 0.8), (5, 1.1) ⇒ ŷ = 1.09 − 0.03x; adding (1.5, 6) ⇒ ŷ = 3.35 − 0.55x. **Simplest solution:** remove the outliers manually (after plotting the data). Extra: automatic rules (|z-score| > 3, very large residual of a first fit, points far from all centroids in k-means, Lecture 9) and robust losses (absolute error) exist; never remove points only because they disagree with the model you would like.

### L10.Q10 – Why do we need dimensionality reduction? Describe the digit example.
*Source: Lecture 10, slide 8*

Digit recognition ("5" or "8") from 8×8 images: **64 pixels = 64 inputs!** A linear model needs 65 parameters: larger generalization gap (O(m/N), Lecture 2), more variance (Lecture 5), more data and computation needed. But we can represent basically the same information with **2 numbers**: (1) **intensity** (amount of ink, e.g. mean pixel value) and (2) **symmetry** (how much the image equals its mirror image). In the (z₁, z₂) plane the two digits are almost linearly separable (green dashed line), with only 3 parameters. These features were designed by hand (domain knowledge, like Φ in Lecture 7); PCA finds features automatically.

### L10.Q11 – What is the goal of PCA? Describe the picture of slide 9.
*Source: Lecture 10, slide 9*

**Goal:** extract input features z in a **lower-dimensional** space while keeping all the important information. In the figure the points (x₁, x₂) lie along a line through the origin (orange axis z₁); the new axes (z₁ along the cloud, z₂ perpendicular) are a rotation of the old ones. In the new coordinates the points have z₂ ≈ 0, so representing each point with only z₁ could be sufficient to "explain" the differences among the points; z₂ only contains small deviations (noise).

### L10.Q12 – (Derivation) How does PCA find the direction of the largest fluctuations? ★
*Source: Lecture 10, slide 10*

Take a direction v and the new feature **z ≜ vᵀx** (projection). With centered data: Var[z] = (1/N) Σ z⁽ⁱ⁾z⁽ⁱ⁾ᵀ = (1/N) Σ vᵀx⁽ⁱ⁾x⁽ⁱ⁾ᵀv = **vᵀ[(1/N) Σ x⁽ⁱ⁾x⁽ⁱ⁾ᵀ]v = vᵀAv**, where A is the covariance of X. We need to capture the **largest fluctuations**: maximize vᵀAv with ‖v‖ = 1 (without the constraint the variance grows without limit by scaling v). Which v? ⇒ SVD (slides 11–12): the solution is v₁, the first column of V (eigenvector of A with the largest eigenvalue), and the maximum variance is γ₁²/N. Note: the slide writes vᵀx⁽ⁱ⁾·x⁽ⁱ⁾·v, meaning vᵀx⁽ⁱ⁾x⁽ⁱ⁾ᵀv.

### L10.Q13 – Define the singular value decomposition.
*Source: Lecture 10, slides 11–12*

Any rectangular X ∈ ℝ^{N×(m+1)} can be written as **X = U·Γ·Vᵀ** with: **U** ∈ ℝ^{N×(m+1)} orthonormal matrix (UᵀU = I_{m+1}); **V** ∈ ℝ^{(m+1)×(m+1)} orthogonal matrix (VᵀV = VVᵀ = I_{m+1}); **Γ** ∈ ℝ^{(m+1)×(m+1)} non-negative diagonal matrix whose elements are the **singular values**, ordered γ₁ ≥ γ₂ ≥ … ≥ γ_{m+1} ≥ 0. Geometric meaning: Vᵀ rotates, Γ stretches along the axes, U maps to the N-dimensional space.

### L10.Q14 – (Derivation) Why are the principal components given by V and their importance by the singular values? ★
*Source: Lecture 10, slides 10–12 (derivation extra)*

From X = UΓVᵀ and UᵀU = I: XᵀX = VΓUᵀUΓVᵀ = VΓ²Vᵀ ⇒ **A = (1/N)XᵀX = V(Γ²/N)Vᵀ**. So the columns of V are the eigenvectors of the covariance and **γ_i²/N is the variance along v_i**. Maximization: any unit v can be written v = Vc with ‖c‖ = 1 ⇒ vᵀAv = Σ_i c_i² γ_i²/N ≤ γ₁²/N, with equality for c = e₁, i.e. **v = v₁**; the best direction orthogonal to v₁ is v₂, and so on. Moreover Z = XV = UΓ has ZᵀZ/N = Γ²/N diagonal: the principal components are **uncorrelated**. This is why "γ_i says the importance of each element of z" and why PCA keeps the components with the largest singular values.

### L10.Q15 – How is the reduced representation built and how is k chosen? ★
*Source: Lecture 10, slide 12*

**Z ≜ X·V**: the coordinates of all points in the new axes (a rotation, nothing is lost: X = ZVᵀ). If only the first k singular values are large (in the slide's plot only the first 2), keep **Ẑ = X·V_k**, with Ẑ ∈ ℝ^{N×k} and V_k ∈ ℝ^{(m+1)×k} the first k columns of V (k = 2 in the slide). The learning algorithm then uses ẑ instead of x; a new point is transformed with the same centering/scaling and ẑ* = V_kᵀx*. **Choice of k (extra):** the "knee" of the γ_i plot (scree plot, same idea as Lecture 9), or the smallest k with Σ_{i≤k} γ_i² / Σ_i γ_i² ≥ e.g. 90% of the variance; or treat k as a hyperparameter chosen by cross-validation (Lecture 6). **Information lost:** Σ_{i>k} γ_i², the sum of squared distances of the points from the k-dimensional subspace (variance along the discarded directions). Note: X must be centered (and usually standardized) and without the column of ones, which would be all zeros after centering.

### L10.Q16 – (Extra) Summarize the PCA algorithm in steps.
*Source: Lecture 10, slides 4 and 9–12 (extra)*

1. Center the training inputs (z = x − x̄) and, if the features have different units or scales, normalize them (divide by σ_j).
2. Compute the SVD X = UΓVᵀ of the N×m data matrix (or the eigen-decomposition of A = XᵀX/N).
3. Order the singular values γ₁ ≥ γ₂ ≥ …; component i explains a share γ_i²/Σ_j γ_j² of the variance.
4. Choose k (knee or cumulative share of variance).
5. Project: Ẑ = XV_k; new points: ẑ* = V_kᵀ((x* − x̄)/σ).
6. Train the model on ẑ (or plot ẑ for visualization when k = 2).

### L10.Q17 – (Extra) Advantages and limitations of PCA. ★
*Source: Lecture 10, slides 8–12 (extra)*

**Advantages:** fewer inputs (less computation, fewer parameters, lower variance and less overfitting); uncorrelated features; noise reduction (small components are often noise); visualization of high-dimensional data in 2D (USArrests). **Limitations:** (i) it is **unsupervised**: it ignores y, so a low-variance direction that is discarded may be exactly the one that separates the classes; (ii) it is **linear** (projections only), so curved structures are not captured; (iii) it depends on the **scaling** of the features (standardize first); (iv) it is sensitive to **outliers** (squared criterion, like OLS); (v) the new features are combinations of the original ones, so they are harder to interpret (the loadings help); (vi) it is not feature *selection*: all original features are still needed to compute ẑ.

### L10.Q18 – (Extra) Compare normalization, whitening and PCA.
*Source: Lecture 10, slides 4–5 and 10–12 (extra)*

| | Normalization | Whitening | PCA |
|---|---|---|---|
| Transform | z_j = x_j/σ_j | z = A^(−1/2)x | z = V_kᵀx |
| Effect on the covariance | unit variances, correlations unchanged | identity: unit variances, zero correlations | diagonal (Γ²/N), uncorrelated components |
| Dimension | m → m | m → m | m → k < m |
| Uses | the diagonal of A | the whole A | eigenvectors of A (SVD of X) |

Link: A^(−1/2) = V(Γ²/N)^(−1/2)Vᵀ, i.e. whitening = rotate onto the principal axes, divide each by its standard deviation, rotate back.

### L10.Q19 – Describe the USArrests dataset and the problem.
*Source: Lecture 10, slide 13*

Crime statistics per 100,000 residents in the 50 USA states; 4 variables: (1) Murder arrests, (2) Assault arrests, (3) Percent urban population, (4) Rape arrests (e.g. Alabama 13.2, 236, 58, 21.2). Question: **how to visualize a 4-dimensional dataset?** Pairwise scatter plots would need 6 plots and still not show the global structure; PCA with k = 2 gives a single 2D picture. Extra (computed from the dataset): the variables have very different variances (Murder 18.6, Assault 6806, UrbanPop 205, Rape 86), so they must be **standardized** first; without scaling, PC1 ≈ Assault alone (loading 0.995, 97% of the variance). (Minor: the slide shows Alaska Assault = 260, the dataset has 263.)

### L10.Q20 – What are the principal component loadings of USArrests and how are they used?
*Source: Lecture 10, slide 14*

The **loadings** are the columns of V ∈ ℝ^{4×4}; the first two give **V_q ∈ ℝ^{4×2}** and **Ẑ = X·V₂** (z₁ = PC1, z₂ = PC2). PC1 = (−0.5359, −0.5832, −0.2782, −0.5432), PC2 = (0.4182, 0.1880, −0.8728, −0.1673) for (Murder, Assault, UrbanPop, Rape). Each score is a weighted sum of the standardized features, e.g. z₁ = −0.5359·Murder − 0.5832·Assault − 0.2782·UrbanPop − 0.5432·Rape. We can plot both the **projected data** (scores) and the **new dimensions**: the direction of the Murder feature is the line from the origin to [−0.5359, 0.4182] (its row of V₂). Extra (computed): the four components explain 62.0%, 24.7%, 8.9% and 4.3% of the variance, so two components keep 86.8%. Worked score (standardized with 1/N): Alabama z = (1.255, 0.791, −0.526, −0.003) ⇒ PC1 = −0.536·1.255 − 0.583·0.791 + 0.278·0.526 + 0.543·0.003 ≈ −0.99, PC2 ≈ 1.13.

### L10.Q21 – Interpret the two principal components of USArrests.
*Source: Lecture 10, slide 15*

The 1st loading vector gives similar weight to Assault, Murder and Rape and less weight to UrbanPop ⇒ **the 1st component roughly corresponds to a measure of overall rates of serious crimes**. The 2nd loading vector places most of its weight on UrbanPop (−0.8728) and much less on the other three ⇒ **the 2nd component roughly corresponds to the level of urbanization of the state**. The plot is a **biplot**: dots = states (scores), arrows = original variables (loadings).

### L10.Q22 – What does the biplot say about the correlation of the variables?
*Source: Lecture 10, slide 16*

The crime-related variables (Murder, Assault, Rape) are located close to each other, and UrbanPop is far from the other three ⇒ the crime variables are **correlated** with each other (states with high murder rates tend to have high assault and rape rates), while UrbanPop is **less correlated** with the other three. Check (extra, from the data): correlation Murder–Assault 0.80, Assault–Rape 0.67, Murder–Rape 0.56, but UrbanPop–Murder 0.07, UrbanPop–Assault 0.26, UrbanPop–Rape 0.41.

### L10.Q23 – How are the states placed in the biplot? Why are the signs in the plot opposite to the table?
*Source: Lecture 10, slide 17*

States with **large positive scores on the 1st component** (California, Nevada, Florida) have high crime rates; states with negative scores (South Dakota, Vermont, North Dakota) have low crime rates. California also has a **high score on the 2nd component** ⇒ high urbanization; the opposite for Mississippi (and the Carolinas). States **close to zero** on both components, such as Virginia, have approximately **average** levels of crime and urbanization. **Signs:** a principal direction is defined only up to its sign (v and −v give the same variance vᵀAv), and the plot uses −v₁ and −v₂ of the table: the arrows point to (0.54, −0.42) for Murder and (0.28, 0.87) for UrbanPop, which is why "positive PC1 = more crime" and "positive PC2 = more urban" even though the table loadings are negative (the circled −0.8728 on the slide). The scores in the plot are also rescaled to fit with the arrows.

### L10.Q24 – (Extra) How does Lecture 10 connect to the previous lectures?
*Source: Lecture 10 (extra)*

- Lecture 1: pre-processing is a new block between 𝒟_N and the learning algorithm; its statistics are part of what is learned from data.
- Lecture 2: outliers are dangerous because of the squared error; fewer inputs ⇒ smaller gap O(m/N); XᵀX appears again (as N·A).
- Lecture 4: normalized/whitened inputs make the error contours rounder ⇒ less zig-zag in GD.
- Lectures 5–6: dimensionality reduction lowers variance (like feature selection with ‖w‖₀ or regularization); k can be chosen by cross-validation; scaling makes the ℓ₂ penalty fair.
- Lecture 7: z = Φ(x) is the same notation as feature transforms, but the aim is standardization; statistics computed on the test set = data snooping; tanh saturation.
- Lecture 8: k-NN distances need normalized features; curse of dimensionality ⇒ fewer dimensions help.
- Lecture 9: k-means also needs scaling and suffers from outliers; the "knee" is used again for k in PCA; k-means groups points (rows), PCA reduces features (columns).

---

## Lecture 11 – Stochastic processes

### Key formulas
- A stochastic process (SP) x(s, t): s = outcome of the random experiment, t = time (integer). Fix s: one **realization**, a deterministic signal. Fix t: a **random variable**
- ★ Mean function: **m(t) = 𝔼_s[x(s, t)]**
- ★ Covariance function: **γ(t₁, t₂) = 𝔼_s[(x(s, t₁) − m(t₁))(x(s, t₂) − m(t₂))]**; γ(t, t) = variance at time t
- ★ (Wide-sense) stationary SP (SSP): **m(t) = m ∀t** and **γ(t₁, t₂) depends only on τ = t₂ − t₁**: γ(τ) = 𝔼[(x(t) − m)(x(t − τ) − m)]
- ★ Properties of γ(τ): **γ(0) ≥ 0**, **|γ(τ)| ≤ γ(0)**, **γ(τ) = γ(−τ)**
- ★ White noise e ~ WN(μ, λ²): 𝔼[e(t)] = μ, Var = λ², **γ(τ) = 0 ∀τ ≠ 0**
- ★ Spectral density: **Γ(ω) = Σ_{τ=−∞}^{+∞} γ(τ) e^(−jωτ)**; properties: real, Γ(ω) ≥ 0, Γ(ω) = Γ(−ω), periodic of period 2π
- ★ Inverse transform: **γ(τ) = (1/2π) ∫_{−π}^{π} Γ(ω) e^(jωτ) dω** ⇒ **γ(0) = Var = (1/2π) ∫_{−π}^{π} Γ(ω) dω**
- ★ White noise spectrum: **Γ(ω) = λ²** (flat)

### L11.Q1 – Why do we need stochastic processes in systems and control?
*Source: Lecture 11, slide 2*

So far the model was a **static map** y = f(x): the output depends only on the current input. In systems and control the output y_t also depends on **past values** of x and y, e.g. y_t = f(y_{t−1}, y_{t−2}, …, x_t, x_{t−1}, …, x_{t−m}). We must take the **time variable** into account, and the data are signals affected by randomness: the right mathematical object is a stochastic process.

### L11.Q2 – Define a stochastic process. What is a realization?
*Source: Lecture 11, slides 1 and 3*

A stochastic process is an **infinite sequence of random variables, all defined on the same probabilistic space**: x(s, t), with s the outcome of the random experiment and t the (discrete) time. Running the experiment once gives one **realization**: a whole signal in time. Different outcomes s give different realizations of the **same** process (the coloured curves on the slides).

### L11.Q3 – What do you obtain by fixing s, and by fixing t? ★
*Source: Lecture 11, slide 4*

- **Fix s = s̄**: x(s̄, t) = x(t) is a **deterministic signal** (one realization).
- **Fix t = t̄**: x(s, t̄) is a **random variable** (a scalar) with its own pdf (the orange curve on the slide).

So a SP is a signal in t for every s, and a random variable for every t.

### L11.Q4 – Why do we use a “wide-sense” characterization?
*Source: Lecture 11, slide 5*

To fully characterize a SP we would need the **pdf at every time and its joint evolution in time**: too complex for learning. In the **wide-sense** characterization we describe x(s, t) only through its **mean** and its **covariance function**, i.e. first- and second-order moments.

### L11.Q5 – Define the mean function of a SP.
*Source: Lecture 11, slide 6*

**m(t) = 𝔼_s[x(s, t)]**: at each time t, the average over all realizations (the expectation is over s, not over time). In general it changes with t: the red curve running through the green realizations on the slide.

### L11.Q6 – Define the covariance function. What does it measure? ★
*Source: Lecture 11, slide 7*

**γ(t₁, t₂) = 𝔼_s[(x(s, t₁) − m(t₁))(x(s, t₂) − m(t₂))]**.
- For t₁ ≠ t₂ it quantifies **how much x(t₁) and x(t₂) are correlated**: large |γ| means knowing the process at t₁ tells a lot about t₂.
- For t₁ = t₂ = t: γ(t, t) = 𝔼[(x(t) − m(t))²] is the **variance** of the SP at time t.

### L11.Q7 – When is a stochastic process stationary (in the wide sense)? ★
*Source: Lecture 11, slides 8–9*

A SP is **stationary (SSP)** if:
1. **m(t) = m for all t** (constant mean);
2. **γ(t₁, t₂) depends only on τ = t₂ − t₁**: γ(t₁, t₂) = γ(τ).

Then γ(0) = 𝔼[(x(t) − m)²] is the (constant) variance and γ(τ) = 𝔼[(x(t) − m)(x(t − τ) − m)] the covariance at lag τ. Stationary processes are a **small subset** of all SPs, but very important in systems and control: their statistics do not change with time.

### L11.Q8 – State the properties of γ(τ) for a SSP and explain them. ★
*Source: Lecture 11, slide 10*

- **P1 Non-negativity**: γ(0) = 𝔼[(x(t) − m)²] ≥ 0 (it is a variance).
- **P2 Variance prevalence**: |γ(τ)| ≤ γ(0) for all τ (extra: Cauchy–Schwarz, |𝔼[XY]| ≤ √(𝔼[X²]𝔼[Y²]) = γ(0) with X = x(t) − m, Y = x(t − τ) − m).
- **P3 Symmetry**: γ(τ) = γ(−τ), an **even** function (swap t and t − τ; by stationarity only the distance counts).

Graphically: the maximum of |γ| is at τ = 0, the function is mirrored around τ = 0 and lives in the band [−γ(0), γ(0)].

### L11.Q9 – (Example) x(s, t) = α(s) for all t, α ~ N(1, 3). Is it stationary? ★
*Source: Lecture 11, slides 11–12*

Check the two conditions.
1. m(t) = 𝔼[α] = **1** for all t ✓.
2. γ(t₁, t₂) = 𝔼[(α − 1)(α − 1)] = 𝔼[(α − 1)²] = Var(α) = **3** for all t₁, t₂ ⇒ γ(τ) = 3 ✓.

**The process is stationary.** Each realization is a constant line at the random value α(s). Extra: knowing x at one time tells you x at every time, so γ(τ) = γ(0) for all τ (maximal correlation, never decays). Note also that one single realization does not reveal m = 1 (its time average is α(s)): the process is stationary but not *ergodic*.

### L11.Q10 – (Example) x(s, t) = α(s)·t − t, α ~ N(1, 3). Is it stationary? ★
*Source: Lecture 11, slides 13–14*

1. m(t) = 𝔼[αt − t] = t·𝔼[α] − t = t − t = **0** for all t ✓.
2. With m = 0: γ(t₁, t₂) = 𝔼[x(t₁)x(t₂)] = 𝔼[(α − 1)t₁ · (α − 1)t₂] = t₁t₂·𝔼[(α − 1)²] = **3·t₁·t₂** ✗: it depends on t₁ and t₂, not only on τ.

**Not stationary.** Picture: x = t(α − 1) are lines through the origin with random slope (up if α > 1, down if α < 1): the spread grows with |t| (variance 3t²), although the mean is always 0. A constant mean is not enough.

### L11.Q11 – Define white noise. ★
*Source: Lecture 11, slide 15*

A SSP e(s, t) is a **white noise**, written **e ~ WN(μ, λ²)**, if:
- 𝔼[e(t)] = μ;
- 𝔼[(e(t) − μ)²] = λ² (variance);
- **γ_e(τ) = 𝔼[(e(t) − μ)(e(t − τ) − μ)] = 0 for all τ ≠ 0**: values at different times are uncorrelated.

Its covariance function is a single spike λ² at τ = 0. White noise is the basic building block: other SSPs are obtained by filtering it.

### L11.Q12 – Why a frequency-domain description? Why not the Fourier transform of a realization?
*Source: Lecture 11, slides 16–17*

For deterministic signals the frequency domain is natural: a periodic signal has a Fourier series (lines), a non-periodic one a Fourier transform. For a SP each realization is different, so **transforming one single realization is meaningless**: it describes that outcome, not the process. We need a frequency description of the **process**: transform its covariance function, which is the same for all realizations.

### L11.Q13 – Define the spectral density of a SSP. ★
*Source: Lecture 11, slide 18*

The spectral density (or spectrum) is the Fourier transform of the covariance function:

**Γ(ω) = Σ_{τ=−∞}^{+∞} γ(τ) e^(−jωτ)**.

It tells how the variance of the process is distributed over frequencies. A γ(τ) that decays slowly (strong correlation between nearby samples) gives a spectrum concentrated at low frequencies.

### L11.Q14 – (Derivation) Compute the spectrum of a white noise. ★
*Source: Lecture 11, slide 19*

e ~ WN(0, λ²): γ_e(0) = λ² and γ_e(τ) = 0 for τ ≠ 0. Then

Γ_e(ω) = Σ γ_e(τ)e^(−jωτ) = γ_e(0)·e^(−jω·0) + 0 + 0 + … = **λ²** for all ω.

The spectrum of a WN is **flat**: all frequencies are equally present in one WN realization (the reason for the name “white”, like white light).

### L11.Q15 – List the properties of Γ(ω). ★
*Source: Lecture 11, slide 20*

- **P1 Realness**: Im(Γ(ω)) = 0 ∀ω (because γ is real and even, the imaginary parts of the τ and −τ terms cancel).
- **P2 Non-negativity**: Γ(ω) ≥ 0 ∀ω.
- **P3 Symmetry**: Γ(ω) = Γ(−ω) (even function).
- **P4 Periodicity** of period 2π: Γ(ω) = Γ(ω + 2πk), k ∈ ℤ (τ is an integer, so e^(−j(ω+2πk)τ) = e^(−jωτ)).

Because of P3 and P4 it is enough to draw Γ on [0, π].

### L11.Q16 – Write the inverse transform. How do you get the variance from the plot of Γ(ω)? ★
*Source: Lecture 11, slides 21–22; exam 3 Jul 2023, Ex. 4a*

**γ(τ) = (1/2π) ∫_{−π}^{π} Γ(ω) e^(jωτ) dω**. At τ = 0, e^0 = 1:

**γ(0) = Var[x(t)] = (1/2π) ∫_{−π}^{π} Γ(ω) dω**: the variance is the **area under the graph of Γ on [−π, π], divided by 2π**.

Check on WN: area = λ²·2π ⇒ γ(0) = λ²·2π/2π = λ² ✓.

### L11.Q17 – In which ways can a SSP be represented?
*Source: Lecture 11, slide 23*

1. **Mean m and covariance function γ(τ)**: time-domain description.
2. **Mean m and spectral density Γ(ω)**: frequency-domain description.

They carry the same information, since γ and Γ are a Fourier-transform pair.

### L11.Q18 – (Extra) Practice: x(t) = e(t) + c·e(t − 1), e ~ WN(0, λ²). Compute m, γ(τ), Γ(ω).
*Source: Lecture 11, slides 8–22 (extra, practice on paper)*

- m = 𝔼[e(t)] + c·𝔼[e(t − 1)] = **0**.
- γ(0) = 𝔼[(e(t) + c e(t−1))²] = λ² + c²λ² = **(1 + c²)λ²** (cross terms vanish: WN is uncorrelated).
- γ(±1) = 𝔼[(e(t) + c e(t−1))(e(t−1) + c e(t−2))] = **cλ²**; γ(τ) = 0 for |τ| ≥ 2. It depends only on τ ⇒ stationary.
- Γ(ω) = γ(0) + γ(1)(e^(−jω) + e^(jω)) = **λ²(1 + c² + 2c·cos ω)**.

Check the properties: real ✓, even ✓, 2π-periodic ✓, ≥ 0 since 1 + c² + 2c cos ω ≥ (1 − |c|)² ≥ 0 ✓. With c > 0 the spectrum is larger at low frequencies (smoother realizations); with c < 0 at high frequencies.

---

## Lecture 12 – Learning with stochastic processes

### Key formulas
- Data: **one realization** 𝒟_N = {x(s̄, 1), x(s̄, 2), …, x(s̄, N)} of a SSP x(s, t)
- ★ Error of an estimator Q̂_N of Q, with Q* = 𝔼[Q̂_N]: **𝔼[(Q − Q̂_N)²] = (Q − Q*)² + 𝔼[(Q̂_N − Q*)²]** (bias + variance)
- ★ **Correct** (unbiased): 𝔼[Q̂_N] = Q. **Consistent**: 𝔼[(Q* − Q̂_N)²] → 0 as N → ∞ (asymptotically zero variance)
- ★ Sample mean: **m̂_N = (1/N) Σ_{t=1}^{N} x(s̄, t)**: correct; consistent if **γ(τ) → 0 as |τ| → ∞**
- ★ Sample covariance (zero-mean SSP): **γ̂_N(τ) = 1/(N − |τ|) Σ_{t=1}^{N−|τ|} x(t)x(t + |τ|)**, −N < τ < N: correct; consistent if γ(τ) → 0. In practice **|τ| ≪ N**
- ★ Sample spectral density: **Γ̂_N(ω) = Σ_{τ=−(N−1)}^{N−1} γ̂_N(τ) e^(−jωτ)**, −π < ω ≤ π: two approximations (truncated sum, estimated γ); **asymptotically correct, NOT consistent**
- ★ Periodogram (weights 1/N instead of 1/(N − |τ|)): **Γ̂″_N(ω) = (1/N) |Σ_{t=1}^{N} x(t) e^(−jωt)|²**, computed with the FFT
- ★ Averaging: split the data in K parts, **Γ̂*_N = (1/K) Σ_k Γ̂″^[k]_{N/K}** ⇒ 𝔼[(Γ̂* − Γ)²] ≈ (1/K)·𝔼[(Γ̂″ − Γ)²]
- ★ Nonstationary model: **x(s, t) = x_SSP(s, t) + v(t)** (deterministic part v)
- ★ Linear trend v(t) = kt + q by OLS: **[k̂; q̂] = [Σt² Σt; Σt N]⁻¹ [Σ t·x(t); Σ x(t)]**
- ★ Seasonality v(t) = v(t + kT), T known: **v̂(t) = (1/M) Σ_{k=0}^{M−1} x(t + kT)**, t = 1…T, M·T ≤ N

### L12.Q1 – Which learning problems arise with stochastic processes? What are the data?
*Source: Lecture 12, slide 2*

Working with time realizations of a SSP we want to estimate: (1) the **mean** m; (2) the **covariance function** γ(τ); (3) the **spectral density** Γ(ω). The common starting point is a **finite set** of samples of **one specific realization**: 𝒟_N = {x(s̄, 1), …, x(s̄, N)}. Unlike Lectures 1–10 we do not have many independent pairs (x⁽ⁱ⁾, y⁽ⁱ⁾): we have one signal, and its samples are correlated.

### L12.Q2 – Decompose the error of an estimator into bias and variance. ★
*Source: Lecture 12, slide 3*

Given a quantity Q and an estimator Q̂_N computed from 𝒟_N, the out-of-sample error is E_out(Q̂_N) = 𝔼[(Q − Q̂_N)²]. With Q* = 𝔼[Q̂_N] (the average estimate):

**𝔼[(Q − Q̂_N)²] = (Q − Q*)² + 𝔼[(Q̂_N − Q*)²]** = **bias** + **variance**

(the cross term vanishes because 𝔼[Q̂_N − Q*] = 0). It is the same decomposition as Lecture 5, with the estimator in place of the learned model.

### L12.Q3 – When is an estimator correct? When is it consistent? ★
*Source: Lecture 12, slide 4*

- **Correct** (unbiased): **𝔼[Q̂_N] = Q* = Q**, zero bias.
- **Consistent**: **𝔼[(Q* − Q̂_N)²] → 0 as N → ∞**, asymptotically zero variance.

A correct and consistent estimator gets arbitrarily close to Q with more data. Correctness alone is not enough: an unbiased estimator can stay very noisy.

### L12.Q4 – Define the sample mean and prove that it is correct. ★
*Source: Lecture 12, slides 5–6*

Goal: from 𝒟_N infer m = 𝔼_s[x(s, t)]. Proposal: **m̂_N = (1/N) Σ_{t=1}^{N} x(s̄, t)** (a time average over one realization).

Correctness: 𝔼_s[m̂_N] = (1/N) Σ_t 𝔼_s[x(s, t)] = (1/N) Σ_t m = (1/N)·N·m = **m** ✓ (stationarity: the mean is the same at every t).

### L12.Q5 – When is the sample mean consistent? Give a counterexample. ★
*Source: Lecture 12, slides 6–7*

**m̂_N is consistent if γ(τ) → 0 as |τ| → ∞**: samples far apart in time are almost uncorrelated, so a long realization keeps bringing new information. Good news: this holds for **all the SSPs of this course**.

Counterexample: x(s, t) = α(s), α ~ N(0, 1), so 𝔼[x] = 0. Then m̂_N = (1/N) Σ α(s̄) = α(s̄): generally different from zero, and **increasing N does not decrease the error** (every realization is a flat line). In fact γ(τ) = 1 for every τ: it does not go to 0.

### L12.Q6 – Define the sample covariance function. Why must |τ| ≪ N? ★
*Source: Lecture 12, slides 8–9*

For a zero-mean SSP, γ(τ) = 𝔼[x(t)x(t − τ)]. Proposal:

**γ̂_N(τ) = 1/(N − τ) Σ_{t=1}^{N−τ} x(s̄, t)x(s̄, t + τ)**, 0 ≤ τ < N; variation for −N < τ < N: γ̂′_N(τ) = 1/(N − |τ|) Σ_{t=1}^{N−|τ|} x(t)x(t + |τ|).

Each lag averages a different number of products: γ̂(0) averages N squares, γ̂(1) N − 1 products, …, **γ̂(N − 1) = x(1)·x(N) relies on only two points**. So: in theory |τ| < N, **in practice |τ| ≪ N**, otherwise the estimate is very noisy. (For non-zero mean, subtract m̂_N first.)

### L12.Q7 – Prove that γ̂_N(τ) is correct. When is it consistent? ★
*Source: Lecture 12, slide 10*

𝔼[γ̂_N(τ)] = 1/(N − τ) Σ_{t=1}^{N−τ} 𝔼[x(t)x(t + τ)] = 1/(N − τ) Σ_{t=1}^{N−τ} γ(τ) = ((N − τ)/(N − τ))·γ(τ) = **γ(τ)** ✓.

**Consistency**: γ̂_N(τ) is consistent if **γ(τ) → 0 as |τ| → ∞**, the same condition as for the mean.

### L12.Q8 – Define the sample spectral density. Which approximations does it contain? ★
*Source: Lecture 12, slide 11*

Goal (zero-mean SSP): Γ(ω) = Σ_{τ=−∞}^{+∞} γ(τ)e^(−jωτ), with ω in (−π, π] by periodicity. Proposal:

**Γ̂_N(ω) = Σ_{τ=−(N−1)}^{N−1} γ̂′_N(τ) e^(−jωτ)**, −π < ω ≤ π.

Two sources of approximation: (1) the **sum is truncated** to |τ| ≤ N − 1 (we have no lags beyond); (2) the true γ(τ) is replaced by its **estimate** γ̂′_N(τ).

### L12.Q9 – Is Γ̂_N(ω) correct?
*Source: Lecture 12, slide 12*

𝔼[Γ̂_N(ω)] = Σ_{τ=−(N−1)}^{N−1} 𝔼[γ̂_N(τ)]e^(−jωτ) = Σ_{τ=−(N−1)}^{N−1} γ(τ)e^(−jωτ), since γ̂_N is correct. This is the truncated version of Γ(ω), so it is not exactly Γ(ω) for finite N, but **𝔼[Γ̂_N(ω)] → Γ(ω) as N → ∞**: **asymptotic correctness**.

### L12.Q10 – Is Γ̂_N consistent? What is the computational problem and its solution? ★
*Source: Lecture 12, slide 13*

**Γ̂_N(ω) is NOT consistent, even if γ(τ) → 0**: its variance does not vanish as N grows (the large lags, estimated from very few products, keep adding noise).

Computational problem: we would first compute all γ̂_N(τ) and then Γ̂_N(ω). Solution: use the estimator γ̂″_N(τ) = (1/N) Σ_{t=1}^{N−|τ|} x(t)x(t + |τ|) (weight 1/N). Then one can prove

**Γ̂″_N(ω) = Σ_{τ=−(N−1)}^{N−1} γ̂″_N(τ)e^(−jωτ) = (1/N) |Σ_{t=1}^{N} x(t)e^(−jωt)|²**,

the **periodogram**: the squared magnitude of the discrete Fourier transform of the data, efficiently computable with the **FFT**.

### L12.Q11 – How is the consistency problem solved in practice? ★
*Source: Lecture 12, slides 14–15; exam 3 Jul 2023, Ex. 4c*

Γ̂″_N is not consistent: 𝔼[(Γ̂″_N(ω) − Γ(ω))²] does not go to 0 as N → ∞, so **increasing N is not a solution** (the plot stays a noisy band around Γ). Solution: **averaging**. Split the data into K parts (the slides use K = 3, each of length N/3), compute a periodogram on each, and average them:

**Γ̂*_N(ω) = (1/K) Σ_{k=1}^{K} Γ̂″^[k]_{N/K}(ω)** ⇒ **𝔼[(Γ̂*_N − Γ)²] ≈ (1/K)·𝔼[(Γ̂″_N − Γ)²]**.

The variance drops by a factor K (red curve vs blue on slide 15). Extra: the price is a lower frequency resolution, since each piece is shorter (bias–variance again). Useful only when N is large.

### L12.Q12 – What if the process is not stationary? What is the standard procedure? ★
*Source: Lecture 12, slides 16–17*

Two types of nonstationarity are considered: (1) a **linear trend**; (2) **seasonality**. Model: **x(s, t) = x_SSP(s, t) + v(t)**: a stationary part plus a **deterministic** component.

Procedure:
1. estimate the nonstationary part **v̂(t)**;
2. remove it: **x̂_SSP(s̄, t) = x(s̄, t) − v̂(t)**;
3. work with the SSP: estimate m̂, γ̂(τ), Γ̂(ω);
4. go back to x: **m(t) = m̂ + v̂(t)**.

### L12.Q13 – (Derivation) How is a linear trend estimated and removed? ★
*Source: Lecture 12, slides 18–20; exams 13 Feb 2025, Ex. 4.3 and 21 Jan 2026, Ex. 4c*

Model of the trend: **v(t) = kt + q**. Intuition: 𝔼[x_SSP(t)] = 𝔼[x(t) − v(t)] = 0, so choose k, q minimizing 𝔼[(x(t) − (kt + q))²]. Replace the expectation with the sample average:

v̂ = argmin_{k,q} (1/N) Σ_{t=1}^{N} (x(t) − [k q][t; 1])²

This is **linear regression** (Lecture 2) with input [t, 1] and output x(t). OLS gives

**[k̂; q̂] = [Σt² Σt; Σt N]⁻¹ [Σ t·x(t); Σ x(t)]**, then **v̂(t) = k̂t + q̂** and **x̂_SSP(t) = x(t) − v̂(t)**.

### L12.Q14 – How is seasonality estimated? Why does it work? ★
*Source: Lecture 12, slides 21–22; exam 21 Jan 2026, Ex. 4c(iv)*

Assume x(t) = x_SSP(t) + v(t) with **v(t) = v(t + kT)**, k ∈ ℤ, and the **period T known**. Key idea: average the data over the M periods available (M·T ≤ N):

**v̂(t) = (1/M) Σ_{k=0}^{M−1} x(t + kT)**, t = 1, …, T.

Why it works: v̂(t) = (1/M) Σ x_SSP(t + kT) + (1/M) Σ v(t + kT). The second term is (1/M)·M·v(t) = v(t) by periodicity; the first is an average of a zero-mean SSP, which goes to **0 if M is large**. So v̂(t) → v(t).

### L12.Q15 – (Extra) Weights 1/(N − |τ|) or 1/N for the covariance estimate?
*Source: Lecture 12, slides 8–13 (extra)*

- 1/(N − |τ|) (γ̂_N): **correct** for every τ, but at large lags it divides a few products by a small number: high variance.
- 1/N (γ̂″_N): **biased**, 𝔼[γ̂″_N(τ)] = ((N − |τ|)/N)·γ(τ), but **asymptotically correct**, with smaller values (and variance) at large lags; its transform is the periodogram (1/N)|Σx(t)e^(−jωt)|², which is always ≥ 0 like a true spectrum and computable by FFT.

A small bias in exchange for less variance: the bias–variance trade-off again.

### L12.Q16 – (Extra) Why is γ(τ) → 0 the key condition?
*Source: Lecture 12, slides 6–10 (extra)*

The data are N samples of **one** realization, so they are not independent as in Lectures 1–10. If γ(τ) → 0, samples far apart are nearly uncorrelated: a long realization behaves like many almost-independent samples, and averages over time converge (the time average replaces the ensemble average, like Hoeffding in Lecture 1). If γ does not decay (x = α), the realization never “forgets” its starting value and more data do not help.

### L12.Q17 – (Exam) Given one measured time series, describe the complete workflow.
*Source: Lecture 12, slides 2–22; exams 13 Feb 2025, Ex. 4 and 21 Jan 2026, Ex. 4*

1. Plot the data and look for a **trend** and **seasonality** (known period T).
2. Estimate the trend by **OLS**, v̂ = k̂t + q̂, and subtract it.
3. Estimate the seasonal component by **averaging over periods**, and subtract it.
4. On the residual x̂_SSP: **m̂_N**, then **γ̂_N(τ)** for |τ| ≪ N, then the spectrum with the **periodogram (FFT) + averaging** over K pieces.
5. Check consistency conditions (γ̂ decays to 0) and report m(t) = m̂ + v̂(t).

Statements to know: a trend violates stationarity (time-varying mean); removing the trend does **not** guarantee stationarity (seasonality or changing variance may remain); a periodic component appears as **spectral lines**.

---

## Big picture (Lectures 1–10)

### BP.Q1 – What is the real goal of the course?
*Source: Lecture 1, slides 3 and 15; Lecture 5, slide 2*

Use past data to build a model that predicts well on **new** data. The dataset is only the teacher: after learning, everything useful is in w, and new inputs go through the model. Two questions: how to build the model (fitting, low E_in) and how to be sure it works on new data (generalization, E_out ≈ E_in). Automation uses: system identification (regression), fault detection (classification), predictive maintenance (logistic regression), soft sensors (regression).

### BP.Q2 – Compare linear regression, perceptron and logistic regression.
*Source: Lecture 2, slide 3; Lecture 3, slide 3; Lecture 4, slides 4–5*

All compute the same score s = wᵀx; they differ in what they do with it. Linear regression: ŷ = s (a number), squared error, closed-form OLS (quadratic bowl). Perceptron: ŷ = sign(s) (a class), 0/1 error, PLA (staircase, no usable gradient). Logistic regression: ŷ = h(s) (a probability), cross-entropy from maximum likelihood, gradient descent (smooth convex valley, no formula). Chain of reasoning: type of output → appropriate error → shape of the error → learning method.

### BP.Q3 – Distinguish "what is minimized" from "how it is minimized".
*Source: Lecture 3, slides 4 and 8; Lecture 4, slides 10 and 16*

What: the error function E_in (squared, 0/1, cross-entropy). How: the method (formula, PLA, GD, SGD, Newton). E.g. y·s(k+1) = y·s(k) + ‖x‖² belongs to the PLA *method*, not to the function minimized, which is the fraction of misclassified points.

### BP.Q4 – Compare the generalization results.
*Source: Lecture 1, slide 14; Lecture 2, slide 14; Lecture 3, slide 9; Lecture 5, slides 2 and 7*

Lecture 1: 2M·e^(−2ε²N) (useless for M = ∞). Linear regression: O(m/N). Perceptron: O(√((m/N)·ln N)). Lecture 5 explains where the gap comes from: 𝔼_𝒟[E_out] = bias + variance; more parameters ⇒ less bias, more variance; more data ⇒ less variance.

### BP.Q5 – Summarize the lectures in one table.
*Source: all lectures*

| | L2 Regression | L3 Perceptron | L4 Logistic |
|---|---|---|---|
| Question | How much? | Which class? | How likely? |
| Output | s | sign(s) | h(s) |
| Pointwise error | (y − s)² | ⟦y ≠ sign(s)⟧ | ln(1 + e^(−ys)) |
| Comes from | least squares | counting mistakes | maximum likelihood |
| Shape of E_in | bowl | staircase | convex valley |
| Method | formula (OLS) | PLA | GD / SGD / Newton |
| Fails when | XᵀX singular | non-separable data | separable data (w → ∞) |
| Generalization | O(m/N) | O(√((m/N) ln N)) | not stated yet |
| With ℓ₂ penalty (L6) | (XᵀX + λI)⁻¹XᵀY | not discussed | GD on E_aug (finite minimum even if separable) |

Lecture 1 says when E_in can be trusted (Hoeffding). Lecture 5 says why E_out ends up where it does (bias + variance) and how to choose complexity (according to N). Lecture 6 gives the practical tools to do it: regularization (a knob λ on the effective complexity) and cross-validation (a measured estimate of E_out). Lecture 7 goes beyond linear models: a fixed feature map Φ(x) keeps the model linear in w (only with prior knowledge), while neural networks learn the features themselves (tanh hidden layers, trained by SGD with backpropagation). Lecture 8 drops the model altogether: k-NN predicts by majority vote of the nearest stored points (nonparametric), and closes the classification part with the tools to judge any classifier (confusion matrix, precision, recall, F1). Lecture 9 removes the labels: clustering (unsupervised learning) groups the points; k-means minimizes the within-cluster squared distances E_in = Σ_j Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖² with Lloyd's algorithm (assignment to the nearest centroid + centroid = mean, never increasing E_in, local minimum), k is chosen at the knee of E_in(k) or avoided with hierarchical clustering (dendrogram). Lecture 10 prepares the inputs before any of these methods: z = Φ(x) to standardize the data (centering x − x̄, normalization x_j/σ_j, whitening A^(−1/2)x), data cleaning (outliers) and dimensionality reduction with PCA (SVD X = UΓVᵀ, keep Ẑ = XV_k, the directions of largest variance γ_i²/N); all statistics are computed on the training data only.

### BP.Q6 – Which derivations must you be able to write on paper?
*Source: all lectures*

1. Hoeffding → N ≥ ln(2/δ)/(2ε²) (Lecture 1, slides 10 and 13).
2. OLS: expansion → gradient → normal equations → formula → Hessian (Lecture 2, slides 11–13).
3. PLA: y·s(k+1) = y·s(k) + ‖x‖² (Lecture 3, slide 8).
4. 1 − h(s) = h(−s); likelihood → cross-entropy (Lecture 4, slides 6, 8–10).
5. Gradient of the cross-entropy (Lecture 4, slide 12).
6. GD direction (Taylor + Cauchy–Schwarz) and η(k) heuristic ⇒ w − η∇E_in (Lecture 4, slides 17–23).
7. Newton from the second-order Taylor expansion; Newton on linear regression = OLS (Lecture 4, slide 28).
8. Bias-variance decomposition (Lecture 5, slides 4–7).
9. Regularized least squares ŵ_reg = (XᵀX + λI)⁻¹XᵀY, and why XᵀX + λI is always invertible (Lecture 6, slide 6).
10. Unbiasedness and O(1/k) variance of the validation error; unbiasedness of E_cv (Lecture 6, slides 10 and 12).
11. A single perceptron cannot implement XOR (contradiction on w₀); OR, AND and the XOR multilayer perceptron with explicit weights (Lecture 7, slides 7–10).
12. Linear hidden layers collapse into one linear layer; one backpropagation step on a small network (Lecture 7, slides 13–14).
13. A k-NN prediction by hand (distances, sorting, majority vote) and precision, recall, F1 from a confusion matrix (Lecture 8, slides 7–9 and 14–15).
14. k-means: the mean is the optimal centroid (gradient = 0), the nearest-centroid assignment is optimal, so Lloyd's algorithm never increases E_in and stops in a finite number of steps; one Lloyd run by hand, the knee and a small dendrogram (Lecture 9, slides 4–9).
15. Whitening gives covariance I: A^(−1/2)AA^(−1/2) = I; PCA: Var[vᵀx] = vᵀAv, and from X = UΓVᵀ ⇒ A = V(Γ²/N)Vᵀ ⇒ the best direction is v₁ with variance γ₁²/N; Ẑ = XV_k and the lost information Σ_{i>k} γ_i²; a 2×2 example by hand (Lecture 10, slides 4–5 and 10–12).

### BP.Q7 – Describe the complete workflow to build and certify a model.
*Source: Lecture 6, slides 13–14; Lecture 1, slide 14; Lecture 5, slide 11; Lecture 10, slides 2–3*

1. Split the data: training (≈ 70%), validation (≈ 15%), test (≈ 15%); or, with few data, use cross-validation on training + validation and keep only a test set.
1b. Pre-process (Lecture 10): clean outliers, center and normalize (or whiten), optionally reduce the inputs with PCA; compute x̄, σ, A, V on the training data only and apply the same transform to validation, test and new data.
2. For each candidate (model class ℋ, complexity m, λ, …) train on the training data: OLS / ridge formula, PLA, GD / SGD / Newton on E_in or E_aug, SGD with backpropagation for neural networks.
3. Compute the validation or cross-validation error of each candidate and pick the minimum (the "right complexity" of Lecture 5, chosen according to the data available).
4. Retrain the chosen candidate on all training + validation data: the product.
5. Evaluate it once on the test set: the certificate of performance (an unbiased estimate of E_out, Hoeffding with M = 1).

---

# Part II – Past exam questions with solutions

Exams used: 7 Nov 2022, 6 Feb 2023, 3 Jul 2023, 7 Sep 2023 (Model Identification and Data Analysis – Part 1), 12 Nov 2024, 9 Jan 2025, 13 Feb 2025, 21 Jan 2026, 11 Feb 2026. Only the parts covered in Lectures 1–10 are solved here. Where an official solution is incomplete or inconsistent with the lectures, a note says so.

| Topic | Exams | Count |
|---|---|---|
| Logistic regression | Nov 22, Feb 23, Jan 25, Feb 25, Jan 26 | 5 |
| Least squares / OLS | Sep 23, Nov 24, Feb 25, Feb 26 | 4 |
| Perceptron / PLA | Nov 22, Feb 23, Jul 23, Jan 26 | 4 |
| Bias-variance, overfitting, learning curves | Jul 23, Sep 23, Nov 24, Jan 26 | 4 |
| GD / SGD / step size / stopping / quasi-Newton | Nov 22 (×2), Jan 25 | 3 |
| Feasibility of learning, Hoeffding, PAC | Feb 25, Jan 26 | 2 |
| Regularization (ℓ₂, constraints, feature selection) | Nov 22, Feb 23, Nov 24 (×2), Jan 26, Feb 26 | 5 |
| Validation / cross-validation | Jul 23, Feb 26 (+ partial: Feb 23, Jul 23, Sep 23, Jan 25) | 2 (+4) |
| Nonlinear models / neural networks | Nov 22, Sep 23, Nov 24, Feb 25 (×2), Jan 26 | 5 |
| k-NN / nonparametric classification | Feb 23, Jul 23, Sep 23, Jan 25, Feb 26 | 5 |
| Classifier quality (confusion matrix, precision, recall) | Nov 22, Jul 23 | 2 |
| Clustering (k-means, Lloyd, choice of k, hierarchical) | Nov 22, Feb 23, Nov 24, Feb 25 (+ partial: Jan 25) | 4 (+1) |
| Data pre-processing / PCA | Nov 22, Jan 25, Feb 25, Feb 26 | 4 |

## A. Feasibility of learning

### EX.A1 – Why is learning impossible in a deterministic framework? Give an example.
*Exam: 13 Feb 2025, Ex. 1.1. Theory: Lecture 1, slides 5–7*

In a deterministic framework we want f **exactly** outside the samples, and the data alone cannot determine it: given N points, infinitely many functions pass through all of them and behave arbitrarily elsewhere, and all are equally consistent with the data (Hume's problem of induction). Example: three-marble sequences labelled ±1; "y = +1 if the first marble is black" and "y = −1 if more than one marble is white" fit all six training examples but give opposite predictions for ● ○ ○.

### EX.A2 – Probabilistic perspective: state Hoeffding, bound |E_in − E_out|, explain PAC feasibility.
*Exams: 13 Feb 2025, Ex. 1.2; 21 Jan 2026, Ex. 2c–2d. Theory: Lecture 1, slides 9–14*

1. Assume data i.i.d. from an unknown but fixed P(x, y), and future inputs from the same P. We ask for a guarantee with high probability, not certainty.
2. Hoeffding: for i.i.d. Z₁…Z_N in [0, 1] with mean μ, ℙ(|(1/N)ΣZᵢ − μ| > ε) ≤ 2e^(−2Nε²).
3. For a fixed h, set Zᵢ = ℓ(h(xᵢ), yᵢ) with 0 ≤ ℓ ≤ 1: (1/N)ΣZᵢ = E_in(h), μ = E_out(h) ⇒ **ℙ(|E_in(h) − E_out(h)| > ε) ≤ 2e^(−2Nε²)**.
4. The probability of being misled decays exponentially in N and does not depend on f or E_out.
5. PAC: imposing 2e^(−2Nε²) ≤ δ gives N ≥ (1/(2ε²))·ln(2/δ) ⇒ ℙ(|E_in − E_out| ≤ ε) ≥ 1 − δ: with probability at least 1 − δ (probably), E_out is within ε of E_in (approximately correct). Example: ε = δ = 0.05 ⇒ N ≥ 738.

Note: the bound holds for h fixed before seeing the data; for ĥ chosen among M hypotheses it becomes 2M·e^(−2Nε²) (Lecture 1, slide 14).

### EX.A3 – Define E_out; difference between fitting and generalizing.
*Exam: 21 Jan 2026, Ex. 2a. Theory: Lecture 1, slides 12 and 15*

E_in(h) = (1/N)Σ ℓ(h(xᵢ), yᵢ) (measurable); **E_out(h) = 𝔼₍ₓ,ᵧ₎∼P[ℓ(h(x), y)]**, the expected loss on new data from the same distribution (not measurable). Fitting = small E_in; generalizing = small E_out (E_in a reliable estimate of E_out). A hypothesis can have small E_in and large E_out if it captures noise or accidental regularities: overfitting.

## B. Linear regression

### EX.B1 – Derive the OLS formula from the squared loss (matrix form, cost, estimate).
*Exams: 7 Sep 2023, Ex. 1b; 11 Feb 2026, Ex. 1a–1b. Theory: Lecture 2, slides 7–12*

Y = Xθ + v, X ∈ ℝ^(N×(m+1)) with rows xᵢᵀ. J(θ) = ‖Y − Xθ‖² = YᵀY − 2θᵀXᵀY + θᵀXᵀXθ (cross terms are equal scalars). ∇J = −2XᵀY + 2XᵀXθ = 0 ⇒ **XᵀXθ̂ = XᵀY** ⇒ **θ̂ = (XᵀX)⁻¹XᵀY**. Hessian 2XᵀX ⪰ 0 (zᵀXᵀXz = ‖Xz‖²) ⇒ minimum, unique if X has full column rank.

### EX.B2 – Uniqueness condition and how to promote it during data collection.
*Exam: 11 Feb 2026, Ex. 1b. Theory: Lecture 2, slide 13*

Unique ⇔ XᵀX invertible ⇔ X has full column rank (linearly independent columns), which needs N ≥ number of parameters. During data collection make the inputs vary enough and independently of each other (no constant features, no features that always move together, sufficiently exciting experiment). Counter-example: machinery hours always proportional to loading operations ⇒ collinear columns ⇒ their effects cannot be separated. If singular: flat valley, infinitely many minimizers (experimental or structural cause).

### EX.B3 – OLS on a dataset: assembly time vs size.
*Exam: 12 Nov 2024, Ex. 1a. Data: (1.5, 2.5), (2.0, 5.5), (2.5, 5.2), (3.0, 8.1). Theory: Lecture 2, slides 5–12*

Problem: min over (w₀, w₁) of J = Σ(yᵢ − w₀ − w₁xᵢ)², a convex bowl. With S_x = Σxᵢ, S_y = Σyᵢ, S_xx = Σxᵢ², S_xy = Σxᵢyᵢ: w₁ = (N·S_xy − S_x·S_y)/(N·S_xx − S_x²), w₀ = (S_y − w₁S_x)/N (the 2×2 case of (XᵀX)⁻¹XᵀY, with XᵀX = [N S_x; S_x S_xx]). Numbers: N = 4, S_x = 9, S_y = 21.3, S_xx = 21.5, S_xy = 52.05; denominator 86 − 81 = 5; **w₁ = 16.5/5 = 3.3, w₀ = (21.3 − 29.7)/4 = −2.1**. Model: time ≈ −2.1 + 3.3·size. Residuals −0.35, +1.00, −0.95, +0.30 (sum 0), E_in ≈ 0.53.

### EX.B4 – Effect of the noise on the LS estimate.
*Exam: 11 Feb 2026, Ex. 1d. Theory: Lecture 2, slide 12 (extra)*

Noiseless: θ̂ = (XᵀX)⁻¹XᵀXθ = θ. With noise: θ̂ = θ + (XᵀX)⁻¹Xᵀv; the error depends on the noise and on the conditioning of XᵀX. 𝔼[v] = 0 ⇒ unbiased. White noise with variance σ² ⇒ Var(θ̂) = σ²(XᵀX)⁻¹: ill-conditioned XᵀX amplifies noise; more and more varied data reduce the variance.

### EX.B5 – Generalization capabilities of linear regression.
*Exam: 7 Sep 2023, Ex. 1c. Theory: Lecture 2, slide 14; Lecture 5, slides 7–10*

E_out = E_in + O(m/N): the gap grows with m and shrinks with N; with N ≫ m, E_in is reliable. 𝔼_𝒟[E_out] = bias + variance (+ σ²): adding features lowers E_in and bias but raises variance. Classical result (linear target, noise σ²): 𝔼[E_in] ≈ σ²(1 − (m+1)/N), 𝔼[E_out] ≈ σ²(1 + (m+1)/N).

### EX.B6 – Training and test error as the training set grows.
*Exam: 7 Sep 2023, Ex. 1a. Theory: Lecture 5, slide 10*

Training error increases (more points to fit, harder to come close to all of them); test error decreases (more information, less variance); both approach the bias level.

### EX.B7 – Removing a linear trend with OLS.
*Exam: 13 Feb 2025, Ex. 4.3 (regression part only). Theory: Lecture 2, slides 5–12*

yₜ = α + βt + ỹₜ is a linear regression on [1, t]: β̂ = Σ(t − t̄)(yₜ − ȳ)/Σ(t − t̄)², α̂ = ȳ − β̂t̄ (the OLS formula of EX.B3 in centred form). Detrended signal ỹₜ = yₜ − (α̂ + β̂t).

## C. Perceptron

### EX.C1 – Perceptron from a drawn boundary; check two points.
*Exam: 7 Nov 2022, Ex. 1a–1b. Line through (−1, 0) and (0, 2), right side = +1; C = (−2, 1), D = (2, 1). Theory: Lecture 3, slides 3 and 6*

h(x) = sign(w₀ + w₁x₁ + w₂x₂); the bias is needed because the line misses the origin. Intercepts: w₀ − w₁ = 0 and w₀ + 2w₂ = 0 ⇒ w₁ = w₀, w₂ = −w₀/2. The origin is on the +1 side ⇒ w₀ > 0; with w₀ = 2: **h(x) = sign(2 + 2x₁ − x₂)**. Check: C: 2 − 4 − 1 = −3 ⇒ −1 ✓; D: 2 + 4 − 1 = 5 ⇒ +1 ✓. Note: w is defined up to a positive factor; [w₁, w₂] = [2, −1] is normal to the line and points to the +1 side.

### EX.C2 – Perceptron and PLA; first update; stopping and generalization.
*Exam: 3 Jul 2023, Ex. 2. Data: (−1, 2) → −1; (−2, −2) → +1; (1, −1) → +1; (−3, 1) → −1; w = [0; 0], b = 0. Theory: Lecture 3, slides 3–9*

(a) ŷ = sign(wᵀx + b); PLA: take a misclassified point, w ← w + y·x, b ← b + y, repeat until no errors; y·s grows by ‖x̃‖² at each update; converges in finite steps if separable.
(b) Example 1: score 0 ⇒ ŷ = +1 ≠ −1 ⇒ update with y = −1: **w = [1, −2], b = −1**. Extra: new scores −6, 1, 2, −6 ⇒ all four already correct, PLA stops.
(c) Yes, stop: PLA's goal is E_in = 0. But E_in = 0 says nothing by itself about new points (PLA stops at the first separating line, possibly close to a point); generalization depends on N relative to the input dimension, O(√((m/N) ln N)). With N = 4 the guarantee is weak.

### EX.C3 – Ellipse: why a perceptron fails; feature map.
*Exam: 21 Jan 2026, Ex. 3a–3b. Theory: Lecture 3, slide 3; Lecture 2, slide 3 (linear in the parameters)*

(a) sign(w₀ + w₁x₁ + w₂x₂) splits the plane into two unbounded half-planes; the interior of an ellipse is bounded ⇒ not representable. (b) With φ(x) = [1, x₁², x₂²]: g(x) = w₀ + w₁x₁² + w₂x₂²; choose w₀ = 1, w₁ = −1/a², w₂ = −1/b² ⇒ g ≥ 0 ⇔ x₁²/a² + x₂²/b² ≤ 1 ⇒ STOP inside, GO outside. Linear in the features, nonlinear in x. Part (c), learning the features from data: EX.H1.

### EX.C4 – Single neuron w = [−0.780, −0.012, 0.128]: training and prediction at (5, 10).
*Exam: 6 Feb 2023, Ex. 2a, 2b, 2d. Theory: Lecture 3, slides 3–7; Lecture 4, slides 5 and 8*

(a) Sign activation ⇒ perceptron ⇒ PLA; sigmoid activation ⇒ logistic regression ⇒ maximum likelihood (cross-entropy minimized by GD). (b) s = −0.780 − 0.060 + 1.280 = **0.44 > 0 ⇒ ŷ = +1**. (d) The perceptron and the sigmoid neuron with threshold 0.5 always agree with the same weights, since σ(s) > 0.5 ⇔ s > 0 (same boundary). Which prediction is "right" cannot be known without the true label. The 5-NN rule on the same data also gives ŷ = 1 (EX.I1), but in general k-NN and a linear neuron can disagree, because they draw different boundaries.

## D. Logistic regression

### EX.D1 – Meaning of p(y|x;θ) and sign of θ_V.
*Exam: 21 Jan 2026, Ex. 1a. Theory: Lecture 4, slides 5–6*

p(yᵢ|xᵢ;θ) = σ(yᵢθᵀxᵢ) is the probability the model assigns to the observed label: σ(θᵀxᵢ) for failure, σ(−θᵀxᵢ) = 1 − σ(θᵀxᵢ) for no failure. θᵀx is a risk score: θ_V > 0 ⇒ more vibration, higher failure probability (sensible); θ_V < 0 would be implausible. Extra: ln[P(+1)/P(−1)] = θᵀx, so one unit of V multiplies the odds of failure by e^(θ_V).

### EX.D2 – Likelihood → negative log-likelihood → cross-entropy.
*Exam: 21 Jan 2026, Ex. 1b. Theory: Lecture 4, slides 8–10*

Independence ⇒ L(θ) = Πσ(yᵢθᵀxᵢ). J(θ) = −ln L = −Σ ln σ(yᵢθᵀxᵢ) (log monotonic, product → sum, minus turns max into min). Since −ln σ(z) = ln(1 + e⁻ᶻ): **J(θ) = Σ ln(1 + e^(−yᵢθᵀxᵢ))**, the cross-entropy.

### EX.D3 – Decision rule and effect of the threshold.
*Exam: 21 Jan 2026, Ex. 1c(ii). Theory: Lecture 4, slide 5*

ŷ = +1 if θᵀx ≥ γ, else −1 (γ = 0 ⇔ probability ≥ 0.5). Raising γ ⇒ more conservative ⇒ fewer false positives, more false negatives. Missing a real failure is costly, so a lower threshold may be preferable.

### EX.D4 – Prediction of a sigmoid neuron at (5, 10).
*Exam: 6 Feb 2023, Ex. 2c. Theory: Lecture 4, slide 5*

s = 0.44 ⇒ σ(0.44) = 1.553/2.553 ≈ **0.608 > 0.5 ⇒ ŷ = +1**: about 61% probability, an uncertain prediction near the boundary.

### EX.D5 – Customer churn: model, 0.7, log-likelihood on a dataset, pros and cons.
*Exam: 9 Jan 2025, Ex. 1. Features: monthly charges x₁, tenure x₂, contract type x₃; churn ∈ {0, 1}. Theory: Lecture 4, slides 5–12, 14*

(a) P(churn = 1|x) = σ(w₀ + w₁x₁ + w₂x₂ + w₃x₃), σ(z) = 1/(1 + e⁻ᶻ): the linear score is squeezed into (0, 1).
(b) 0.7 = a 70% chance that the customer leaves (e.g. target retention actions above a threshold).
(c) With 0/1 labels: ℓ(w) = Σ[yᵢ ln σ(zᵢ) + (1 − yᵢ) ln(1 − σ(zᵢ))]. Rows: ln σ(w₀ + 0.8w₁ + 0.2w₂) + ln[1 − σ(w₀ + 0.5w₁ + 0.5w₂ + w₃)] + ln[1 − σ(w₀ + 0.3w₁ + 0.8w₂ + w₃)] + ln σ(w₀ + 0.7w₁ + 0.4w₂). Maximize ℓ (minimize −ℓ/N, the cross-entropy) numerically with GD: no closed form.
(d) Advantage: interpretable probabilities and weights (log-odds). Limitation: linear decision boundary (linear log-odds).

Notes: the official solution writes −(1/N)Σ ln σ(zᵢ) for every row, ignoring the labels; use the version above. This dataset is **linearly separable** (all churners have contract 0, all others 1), so the likelihood keeps increasing as w₃ → −∞ and the maximum-likelihood estimate does not exist (Lecture 4, slide 14 caveat): stop early or regularize.

### EX.D6 – Probability of loan default: logistic regression, linear regression or neural network?
*Exam: 13 Feb 2025, Ex. 2 (neural networks only briefly). Theory: Lecture 4, slides 2–5; Lecture 5, slide 11*

Linear regression is unsuitable (output not in [0, 1], squared loss does not match 0/1 outcomes). Logistic regression P(y = 1|x) = σ(wᵀx + b) gives a valid probability, is trained by maximum likelihood and is interpretable: appropriate when the log-odds are roughly linear and data are small to moderate. More flexible models (neural networks, Lecture 7) capture strong nonlinearities but need much more data and are less interpretable. Choose complexity according to the available data (Lecture 5, slide 11). Complete answer, including neural networks: EX.H7.

### EX.D7 – Step size of gradient descent in logistic regression.
*Exam: 7 Nov 2022, Ex. 1c. Theory: Lecture 4, slides 14, 22–23*

The cross-entropy is convex, so GD can get arbitrarily close to the global minimum. Too small η ⇒ very slow; too large η ⇒ overshoot ("going up the valley"), bouncing or divergence (quadratic case: converges only for 0 < η < 2/a). Fix: η(k) = η‖∇E_in‖ ⇒ w − η∇E_in.

## E. Optimization

### EX.E1 – Nonlinear optimization: challenges, GD, varying step, stopping criteria, GD vs SGD vs quasi-Newton.
*Exam: 9 Jan 2025, Ex. 2. Theory: Lecture 4, slides 11–28*

(a) No closed form ⇒ iterate; non-convex costs have local minima and saddle points ⇒ result depends on initialization; η and stopping must be chosen.
(b) w(k+1) = w(k) − η∇E_in(w(k)): steepest descent direction (Taylor + Cauchy–Schwarz). Too large η diverges/bounces, too small is slow.
(c) η(k) = η‖∇E_in(w(k))‖: large steps far away (steep), small steps near the minimum (flat); the norms cancel ⇒ w − η∇E_in. Note: the official solution states the opposite (smaller steps where the gradient is steep); answer with the lecture's version.
(d) ‖∇E_in‖ < threshold; maximum iterations; E_in threshold (or no more improvement); combination.
(e) GD: full gradient, smooth path, expensive for large N. SGD: one random point per step, very cheap, online, noisy zig-zag, jitters near the minimum with constant η. Quasi-Newton: w − M⁻¹∇E_in with M ≈ Hessian, uses curvature, few iterations, more memory and computation per step (classical Newton uses the exact Hessian).

### EX.E2 – Why is SGD preferred to GD for large training sets?
*Exam: 7 Nov 2022, Ex. 2d. Theory: Lecture 4, slides 26–27*

One GD update needs the gradient over all N samples; SGD uses one sample per update, about N times cheaper, and starts improving immediately. The single-sample gradient is unbiased, so SGD converges on average, but with a noisy path that keeps oscillating near the minimum unless η decreases. Neural networks are trained exactly this way, with the single-point gradient computed by backpropagation (Lecture 7, slide 14).

## F. Bias-variance, overfitting, learning curves

### EX.F1 – Definitions of bias and variance and their sources.
*Exam: 12 Nov 2024, Ex. 2a. Theory: Lecture 5, slides 4–7*

h̄(x) = 𝔼_𝒟[h^𝒟(x)]; 𝔼_𝒟[E_out] = 𝔼ₓ[(h̄ − f)²] (bias) + 𝔼ₓ{𝔼_𝒟[(h^𝒟 − h̄)²]} (variance) (+ σ² with noise). Bias comes from the model class (its assumptions prevent it from representing f, even on average); variance comes from the finite random dataset (the learned model changes with the data). Note: the official solution writes the variance without 𝔼_𝒟; the correct definition averages over both 𝒟 and x. Mention the derivation: swap expectations, expand, add and subtract h̄², Var = 𝔼[Z²] − (𝔼Z)².

### EX.F2 – Effect of model complexity on bias, variance, training and test error.
*Exams: 12 Nov 2024, Ex. 2b; 21 Jan 2026, Ex. 2b. Theory: Lecture 5, slides 8–9*

More complexity ⇒ bias decreases, variance increases. Training error decreases monotonically; test error is U-shaped (underfitting on the left: high bias, both errors large; overfitting on the right: small E_in, large E_out, high variance). Best: intermediate complexity. Power-grid example: a linear classifier may miss the demand patterns (high bias); a very deep network fits the history perfectly but reacts to noise (high variance).

### EX.F3 – Very low training error, high test error: diagnosis and remedies.
*Exams: 12 Nov 2024, Ex. 2c; 3 Jul 2023, Ex. 1a. Theory: Lecture 5, slides 9–11*

Overfitting (low bias, high variance). Without changing the training cost: collect more data (variance decreases) or reduce complexity (fewer features, simpler ℋ). Regularization is a third remedy, but it changes the training cost (Lecture 6, EX.G3), so it does not fit the Jul 2023 constraint. In all cases the complexity or λ should be chosen by cross-validation (EX.G11).

### EX.F4 – Learning curves: which one is the training error?
*Exams: 3 Jul 2023, Ex. 1b; 7 Sep 2023, Ex. 1a. Theory: Lecture 5, slide 10*

Curve (ii), rising from below, is the training error: with few points the model fits them almost perfectly, with many it cannot. Curve (i), decreasing from above, is the test error: more data ⇒ less variance. Both converge to the bias level (bias + σ² with noise), not zero. The gap (generalization error) shrinks with N; on average the test error stays above the training error.

---

## G. Model selection: regularization and cross-validation

### EX.G1 – Regularization on the assembly-time data; plot OLS, λ = 10 and λ = 1000.
*Exam: 12 Nov 2024, Ex. 1b–1c. Data: (1.5, 2.5), (2.0, 5.5), (2.5, 5.2), (3.0, 8.1). Theory: Lecture 6, slides 4–6 and 14*

(b) With only 4 points, the fit may be sensitive to noise: add an ℓ₂ penalty and minimize J(w) = Σ(yᵢ − w₀ − w₁xᵢ)² + λ(w₀² + w₁²) (same minimizer as E_in + (λ/N)‖w‖²). Solution ŵ_reg = (XᵀX + λI)⁻¹XᵀY, with XᵀX = [4 9; 9 21.5], XᵀY = [21.3; 52.05]. Larger λ ⇒ smaller weights ⇒ less variance, more bias.

(c) Numbers (penalty on both weights, as on the slides):

| λ | w₀ | w₁ | E_in | ‖w‖² |
|---|---|---|---|---|
| 0 (OLS) | −2.10 | 3.30 | 0.53 | 15.3 |
| 10 | 0.56 | 1.49 | 3.53 | 2.54 |
| 1000 | 0.02 | 0.05 | 30.8 | 0.003 |

Qualitative plot: OLS follows the points; λ = 10 is flatter (smaller slope) and its intercept is pulled toward 0; λ = 1000 almost coincides with the x axis (w ≈ 0): strong underfitting.

Notes: the official solution writes the cost with λ(w₀² + w₁²) but then gives w₁ = (N·S_xy − S_x·S_y)/(N·S_xx − S_x² + λN), w₀ = (S_y − w₁S_x)/N, which is the solution when **only the slope** is penalized. With that formula λ = 1000 gives a horizontal line at ȳ ≈ 5.3, not the x axis drawn in the official plot. Say explicitly which weights you penalize. Extra: LOO cross-validation on these 4 points gives E_cv = 1.56 for λ = 0, 2.07 for λ = 1 and 6.27 for λ = 10, so here the best choice is λ = 0. A 2-parameter line on a clearly linear trend is not overfitting; regularization is not automatically beneficial, and λ must be chosen by validation (Lecture 6, slide 14).

### EX.G2 – How can ℓ₂ regularization help if XᵀX is singular?
*Exam: 11 Feb 2026, Ex. 1c. Theory: Lecture 6, slide 6; Lecture 2, slide 13*

Minimize J_λ(θ) = ‖y − Xθ‖² + λ‖θ‖², λ > 0 ⇒ θ̂_λ = (XᵀX + λI)⁻¹Xᵀy. This is always well defined: zᵀ(XᵀX + λI)z = ‖Xz‖² + λ‖z‖² > 0 for all z ≠ 0, so XᵀX + λI is positive definite and invertible even when XᵀX is singular. Geometrically, the flat valley of E_in (infinitely many minimizers) gets a bowl λ‖θ‖² added and becomes a bowl with a unique minimum. The ill-conditioned case improves too: every eigenvalue of XᵀX is raised by λ, so the noise is no longer amplified (Var(θ̂) = σ²(XᵀX)⁻¹ in the unregularized case). Example: two samples with the same input, XᵀX = [2 2; 2 2], λ = 1 ⇒ [3 2; 2 3], det = 5.

### EX.G3 – How does L2 regularization reduce overfitting by balancing bias and variance?
*Exam: 12 Nov 2024, Ex. 2d. Theory: Lecture 6, slides 4–6; Lecture 5, slides 7–9*

Objective: min_w (1/N) Σ(yᵢ − ŷᵢ)² + λ Σ_j w_j² (equivalently E_in + (λ/N)‖w‖²). The penalty keeps the weights small: it is the relaxation of "use fewer parameters" (‖w‖₀ is NP-hard) and reduces the **effective** complexity without changing ℋ. A smaller effective complexity means the learned model changes less from one dataset to another ⇒ **variance decreases**, at the cost of a **slightly higher bias** (the model can no longer reach the best fit). Since overfitting is a variance problem, a suitable λ (chosen by cross-validation) lowers E_out. λ = 0 gives back OLS; λ too large underfits.

### EX.G4 – Formulate ℓ₂-regularized logistic regression.
*Exam: 21 Jan 2026, Ex. 1c(i). Theory: Lecture 6, slide 5; Lecture 4, slides 10 and 14*

**min_{θ∈ℝ⁴} (1/N) Σᵢ log(1 + e^(−yᵢθᵀxᵢ)) + λ‖θ‖²₂**, λ ≥ 0 (or with λ/N, as on the slides). The penalty discourages large weights, limits complexity and improves generalization. In logistic regression it also keeps θ finite when the data are separable (without it, the cross-entropy keeps decreasing as ‖θ‖ → ∞). No closed form: solve with GD, where it acts as weight decay: θ ← (1 − 2ηλ)θ − η∇E_in. λ is chosen by cross-validation.

### EX.G5 – Regularizing one parameter at a time (very large C): how does the training error change?
*Exam: 6 Feb 2023, Ex. 3a. Data: "+" (y = 1) below and around the x₁ axis, "O" (y = 0) above; separable by a horizontal line slightly above the axis. Theory: Lecture 6, slides 4–5; Lecture 4, slides 5 and 14*

Penalty −C·w_j² on one weight only, C very large ⇒ that weight is driven to (almost) zero. Boundary: w₀ + w₁x₁ + w₂x₂ = 0.
- **w₂ → 0:** boundary w₀ + w₁x₁ = 0, a **vertical** line. The classes are not separated by any vertical line ⇒ **training error increases**.
- **w₁ → 0:** boundary w₀ + w₂x₂ = 0, a **horizontal** line. The data are separable by a horizontal line ⇒ **training error stays 0**.
- **w₀ → 0:** the boundary must pass **through the origin**. From the figure, no line through the origin separates the two classes (some "+" points sit on or just above the x₁ axis, on both sides of the x₂ axis) ⇒ **training error increases** (best case one misclassified point).

Notes: this is the "force some weights to zero" idea of slide 4 applied to one weight. The exam writes max Π P − C w_j²; it should be the log-likelihood, max Σ log P(y⁽ⁱ⁾|x⁽ⁱ⁾, w) − C w_j². Also, here **C is the penalty weight** (the role of λ), while on slide 5 C is the **budget** (large C = weak constraint): opposite meanings.

### EX.G6 – Express the regularizer as a hard constraint and explain why it helps against overfitting.
*Exam: 6 Feb 2023, Ex. 3b. Theory: Lecture 6, slides 4–5*

Penalized form: max_w Σ log P(y⁽ⁱ⁾|x⁽ⁱ⁾, w) − C w_j². Equivalent constrained form: **max_w Σ log P(y⁽ⁱ⁾|x⁽ⁱ⁾, w) s.t. w_j² ≤ B** (in general ‖w‖² ≤ B), with B a budget. For each penalty weight there is a budget giving the same solution (Lagrange multiplier; stronger penalty ⇔ smaller budget). Why it helps: the constraint restricts the hypothesis set to models with small weights, which lowers the effective complexity ⇒ lower variance ⇒ less overfitting (at the price of some bias). For logistic regression on separable data it also prevents ‖w‖ → ∞, i.e. an infinitely confident model.

### EX.G7 – Feature selection by bounding the number of non-zero parameters.
*Exam: 7 Nov 2022, Ex. 3 (PCA part in EX.K1). Theory: Lecture 6, slides 4–5; Lecture 2, slide 14; Lecture 5, slides 8–9*

(a) min_w E_in(w) s.t. ‖w‖₀ ≤ s, with ‖w‖₀ the number of non-zero weights: only s features can be used, and the features with non-zero weights are the selected ones.
(b) Feature selection reduces the number of effective parameters m ⇒ smaller generalization gap O(m/N) and lower variance; it also removes irrelevant or noisy features that the model would otherwise use to fit noise. The price is some bias if useful features are removed.
(c) The ‖w‖₀ constraint is combinatorial (2^m subsets, NP-hard). Approximation: relax it to ‖w‖₂² ≤ C, or equivalently minimize E_in + (λ/N)‖w‖₂² (ℓ₂ regularization), which has the closed form (XᵀX + λI)⁻¹XᵀY. Extra: ℓ₂ shrinks the weights but does not set them exactly to zero; the relaxation that produces exact zeros (true feature selection) is the ℓ₁ penalty ‖w‖₁ = Σ|wᵢ| (LASSO), not covered in the slides.

### EX.G8 – Purpose of cross-validation; why it is preferred to the training error; why it combats overfitting.
*Exams: 3 Jul 2023, Ex. 1c; 11 Feb 2026, Ex. 3a. Theory: Lecture 6, slides 7–10 and 13*

Cross-validation estimates the out-of-sample error using only the available data, by testing each model on data it was not trained on. The training error is **optimistic**: the model was fitted to those very data, and E_in decreases monotonically with complexity (Lecture 5, slide 9), so it always favours the most complex model. The validation error is **unbiased** (𝔼[E_val] = E_out, slide 10). Using E_cv as the selection criterion (choose the complexity or λ with minimum E_cv) avoids picking overfitted models: E_cv follows the U-shaped E_out curve, not the decreasing E_in curve. Cross-validation does not change the model; it combats overfitting by guiding the choice of model and hyperparameters.

### EX.G9 – Describe leave-one-out cross-validation and how it estimates the generalization error.
*Exam: 11 Feb 2026, Ex. 3b. Theory: Lecture 6, slides 11–12*

For j = 1, …, N: train the model on the N − 1 points other than j ⇒ ŵ⁻⁽ʲ⁾; evaluate it on point j ⇒ e_j (squared error, or 0/1 error for classification). Then **E_cv = (1/N) Σ e_j**. Each e_j is an unbiased estimate of the E_out of a model trained on N − 1 points, so 𝔼[E_cv] = 𝔼[E_out(ŵ⁻)] ≈ E_out(ŵ) for large N. The final model is trained on all N points; E_cv is its performance estimate. Cost: N trainings; it is the natural choice for small N. k-fold CV is the generalization with more points per fold.

### EX.G10 – Trade-off in the number K of data used for validation.
*Exam: 11 Feb 2026, Ex. 3c. Theory: Lecture 6, slides 9–10*

**K small:** N − K training points ⇒ a good model, but the validation error is an average of few terms ⇒ high variance (Var = O(1/K)), an unreliable estimate. **K large:** reliable estimate, but few training points ⇒ a worse model, and a pessimistic estimate of the performance of the final model trained on all N points. We want E_val(ŵ⁻) ≈ E_out(ŵ⁻) (K large) and E_out(ŵ⁻) ≈ E_out(ŵ) (K small). Rule of thumb **K ≈ N/5** (20% validation). Leave-one-out and k-fold CV reduce the dilemma by rotating the validation set.

### EX.G11 – Cross-validation for model selection; can it be used jointly with regularization?
*Exam: 11 Feb 2026, Ex. 3d. Theory: Lecture 6, slides 13–14*

For each candidate (model class, e.g. logistic regression or k-NN, and each hyperparameter value), compute the cross-validation error; select the candidate with the minimum; retrain it on all the data. **Yes**, it is used jointly with regularization: λ is a hyperparameter, chosen as λ_opt = argmin E_cv(λ) (U-shaped curve: small λ overfits, large λ underfits). The weights w are then learned by minimizing E_aug with λ_opt. Caveat: the E_cv of the selected candidate is optimistic, because it was used to choose; for an honest estimate of E_out keep a separate test set (training ≈ 70%, validation ≈ 15%, test ≈ 15%).

### EX.G12 – How can the quality of a classifier be assessed, apart from the training error?
*Exam: 3 Jul 2023, Ex. 3d (k-NN exercise). Theory: Lecture 6, slides 7–10 and 14; Lecture 1, slide 13*

Evaluate the misclassification rate on data **not used for training**: a held-out test set, a validation set, or cross-validation (LOO is especially cheap for k-NN). On a test set of k points the error rate is an unbiased estimate of E_out with variance O(1/k), and Hoeffding gives ℙ(|E_test − E_out| > ε) ≤ 2e^(−2ε²k). Complete answer with the confusion matrix, precision, recall and F1: EX.I2(d) and EX.I6 (Lecture 8, slides 14–15).

### EX.G13 – How should k be selected in k-NN?
*Exams: 6 Feb 2023, Ex. 1c; 7 Sep 2023, Ex. 3c; 9 Jan 2025, Ex. 3c. Theory: Lecture 6, slides 13–14*

k is a **hyperparameter**: it cannot be chosen by minimizing the training error (k = 1 always gives zero training error, because each point is its own nearest neighbour). Choose it by cross-validation: for each candidate k compute E_cv (LOO: classify each point using the other N − 1) and pick the minimum. Small k ⇒ low bias, high variance; large k ⇒ smoother, higher bias. Theory and rule of thumb k = ⌊√N⌋ (Lecture 8, slide 11): see EX.I1(c) and EX.I3(c).

---

## H. Nonlinear modeling and neural networks

### EX.H1 – Cite an approach that learns suitable nonlinear representations from data (robot and elliptic obstacle).
*Exam: 21 Jan 2026, Ex. 3c (parts a–b: EX.C3). Theory: Lecture 7, slides 5, 10–15*

**Feed-forward neural networks** (e.g. a multilayer perceptron with tanh hidden units). The hidden layers compute a nonlinear transformation of the inputs (a learned Φ(x)), and the output neuron is a linear classifier (perceptron or logistic) on those learned features, so the network learns the feature map and the decision boundary jointly from labelled allowed/forbidden positions. With enough hidden neurons it can represent the elliptic region (combining half-planes, slide 11; universal approximation, slide 15). It is trained by SGD with backpropagation; the number of neurons/layers and the regularization are chosen by cross-validation to avoid overfitting. Note: in parts a–b the hand-made Φ = [1, x₁², x₂²] was legitimate because the obstacle shape was known in advance (prior knowledge, Lecture 7, slide 5).

### EX.H2 – Network with linear hidden activations h(s) = cs and sigmoid output: output, boundary, equivalent network, generalization.
*Exams: 7 Nov 2022, Ex. 2a–c; 7 Sep 2023, Ex. 2a–d. Weights: bias → hidden 1 w₁, hidden 2 w₂; x₁ → w₃, w₄; x₂ → w₅, w₆; output bias w₇, hidden 1 → w₈, hidden 2 → w₉. Theory: Lecture 7, slides 9–13; Lecture 4, slide 5*

(a) Hidden outputs: a₁ = c(w₁ + w₃x₁ + w₅x₂), a₂ = c(w₂ + w₄x₁ + w₆x₂). Output:
P(y = 1 | x, w) = σ(w₇ + w₈a₁ + w₉a₂) = 1 / (1 + e^−[w₇ + c w₈w₁ + c w₉w₂ + (c w₈w₃ + c w₉w₄)x₁ + (c w₈w₅ + c w₉w₆)x₂]).
(b) Boundary: P = 0.5 ⇔ the exponent is 0: **w₇ + c w₈w₁ + c w₉w₂ + (c w₈w₃ + c w₉w₄)x₁ + (c w₈w₅ + c w₉w₆)x₂ = 0**, a **straight line** in (x₁, x₂).
(c) Equivalent network without hidden layer (= logistic regression): bias w̃₀ = w₇ + c(w₈w₁ + w₉w₂), w̃₁ = c(w₈w₃ + w₉w₄), w̃₂ = c(w₈w₅ + w₉w₆), output σ(w̃₀ + w̃₁x₁ + w̃₂x₂).
(d) **Yes.** With linear activations every hidden layer outputs a linear combination of its inputs, and a composition of linear maps is linear (W₂W₁x = (W₂W₁)x), so any number of linear hidden layers collapses into one linear layer. Consequence: hidden layers add expressive power only with a **nonlinear** activation such as tanh (Lecture 7, slide 13); otherwise the network cannot even solve XOR.

### EX.H3 – Purpose of the tanh activation in the hidden layer; why not sign?
*Exam: 12 Nov 2024, Ex. 3a. Theory: Lecture 7, slides 12–14*

tanh introduces the **nonlinearity** that lets the network represent nonlinear relations between features and demand (with linear activations the network would collapse to a linear model, EX.H2). It is preferable to sign because it is **differentiable** (tanh′(s) = 1 − tanh²(s)), so the weights can be trained with SGD and backpropagation; the sign function is flat almost everywhere (zero gradient) and jumps at 0, so gradients carry no information. tanh is also smooth and zero-centred, with outputs in (−1, 1).

### EX.H4 – How does the logistic output give a probability? How do you classify?
*Exam: 12 Nov 2024, Ex. 3b. Theory: Lecture 4, slides 5–10; Lecture 7, slide 12*

The output neuron computes σ(s) = 1/(1 + e⁻ˢ) ∈ (0, 1), where s is a weighted sum of the hidden outputs. Trained with the cross-entropy (maximum likelihood), the output estimates **P(high demand | x)**: the output layer is a logistic regression on the features learned by the hidden layer. Classification: high demand if the probability ≥ 0.5 (s ≥ 0), low otherwise; the threshold can be moved to trade false positives for false negatives (as in EX.D3).

### EX.H5 – Why might the network predict high demand for a product with few reviews and a high price?
*Exam: 12 Nov 2024, Ex. 3c. Theory: Lecture 7, slides 11–15; Lecture 5, slides 7–9; Lecture 6, slide 14*

1. **Learned patterns and interactions:** the weights reflect the training data. If expensive products with good ratings were often in high demand, the network learned that association; through the nonlinear tanh units it can also combine features (e.g. high rating and high price together outweighing the few reviews). The prediction reflects correlations in the data, not causes.
2. **Extrapolation / overfitting:** products with very few reviews and a high price may be rare in the training set. In regions with little data the network's output is poorly constrained (high variance), so a flexible model can give an over-confident, unreliable probability. Remedies: more data, regularization, cross-validated architecture.

(A third possible cause: unnormalized features, e.g. the number of reviews, which can dominate the weighted sums.)

### EX.H6 – Adding a second hidden layer: one advantage and one disadvantage.
*Exam: 12 Nov 2024, Ex. 3d. Theory: Lecture 7, slides 15–16; Lecture 5, slides 8–11; Lecture 6, slides 13–14*

**Advantage:** more capacity and a hierarchical representation (the second layer combines the features of the first, slide 16), so complex interactions can be captured, often with fewer neurons than a very wide shallow network; lower bias. **Disadvantage:** more parameters ⇒ higher variance and risk of overfitting with a limited dataset (Lecture 5); training is slower and harder (non-convex E_in, more local minima, more computation per SGD step); less interpretable. The choice should be made by cross-validation (Lecture 6).

### EX.H7 – Probability of loan default: logistic regression, linear regression or neural network? (complete answer)
*Exam: 13 Feb 2025, Ex. 2 (replaces the partial EX.D6). Theory: Lecture 4, slides 2–5; Lecture 7, slides 12–16; Lecture 5, slide 11*

2.1 Linear regression is **not** appropriate: ŷ = wᵀx + b is not constrained to [0, 1]. **Logistic regression** is appropriate when the log-odds are roughly linear in the features and the dataset is small or moderate; a **neural network** with a sigmoid output is preferable when the relation is strongly nonlinear and many data are available.
2.2 Linear regression: ŷ = wᵀx + b, trained by least squares. Logistic regression: P(y = 1|x) = σ(wᵀx + b), trained by maximum likelihood (cross-entropy). Neural network: hidden layer h = θ(W₁x + b₁) with θ = tanh, output P(y = 1|x) = σ(w₂ᵀh + b₂), trained by SGD with backpropagation; it is logistic regression on learned features (with no hidden layer it reduces exactly to logistic regression).
2.3 Linear regression: continuous outputs, not probabilities. Logistic regression: small/moderate data, approximately linear relation, interpretability needed (credit decisions often must be explained). Neural networks: large datasets, complex nonlinear interactions, accuracy more important than interpretability. In general, model complexity must match the amount of data (Lecture 5, slide 11), and the choice should be validated by cross-validation.

### EX.H8 – Two supervised regression algorithms for house prices: compare.
*Exam: 13 Feb 2025, Ex. 3.2 (3.1 in EX.J5 and EX.K5). Theory: Lecture 2; Lecture 6, slides 4–6; Lecture 7, slides 12–15*

**Linear regression** ŷ = w₀ + Σ wᵢxᵢ: closed-form OLS (or ridge (XᵀX + λI)⁻¹XᵀY), fast, interpretable (each weight is the effect of a feature), needs few data; but it only captures linear relations unless features are engineered by hand. **Neural network with a final linear layer**: h = tanh(W₁x + b₁), ŷ = W₂h + b₂ (linear output because the target is a real number): captures nonlinear relations and feature interactions automatically (universal approximation); but it needs much more data to avoid overfitting, training is iterative and non-convex (SGD + backpropagation), and it is hard to interpret. With thousands of features, both need regularization (weight decay) and cross-validation.

| Aspect | Linear regression | Neural network |
|---|---|---|
| Interpretability | high | low |
| Computational cost | low (closed form) | high (SGD, backpropagation) |
| Nonlinear relations | only with hand-made features | learned automatically |
| Data needed | few | many |

---

## I. Nonparametric classification and classifier quality

### EX.I1 – 5-NN prediction for (5, 10); parametric vs nonparametric; choice of k.
*Exam: 6 Feb 2023, Ex. 1. Data (x₁, x₂, y): (10, 2, 1), (15, 2, 1), (6, 11, 1), (2, 3, 0), (5, 15, 1), (5, 14, 1), (10, 1, 0), (1, 6, 0), (17, 19, 1), (15, 13, 0), (19, 9, 0). Theory: Lecture 8, slides 5–11*

(a) Euclidean distances from (5, 10), e.g. to (10, 2): √(25 + 64) ≈ 9.43. Sorted: (6, 11) 1.41 [y = 1]; (5, 14) 4.00 [1]; (5, 15) 5.00 [1]; (1, 6) 5.66 [0]; (2, 3) 7.62 [0]; then (10, 2) 9.43, … The 5 nearest have three labels 1 and two labels 0 ⇒ majority vote **ŷ = 1**. (With k = 1 or k = 3 the answer is also 1.)
(b) **Nonparametric**: it assumes no structure h(x, w) and builds no model; it predicts directly from the stored data. Pros of nonparametric: no assumptions on the shape of the boundary, no training, consistent as N → ∞. Cons: stores the whole dataset (memory O(N·m)), slow prediction (O(N·m + N log k)), no interpretable model, sensitive to k, to the distance metric and to feature scaling. Parametric methods (perceptron, logistic regression, NN) have the opposite profile: compact, fast, interpretable model, but they can be biased if the assumed structure is wrong, and they need training.
(c) k is the most critical choice (slide 11). Small k: low bias, high variance (k = 1 follows every noisy point, E_in = 0); large k: smooth, high bias (k = N always predicts the majority class). Theory: k → ∞ with k/N → 0 gives the minimum achievable error; rule of thumb k = ⌊√N⌋ (here ⌊√11⌋ = 3). In practice choose k by cross-validation (LOO is cheap: classify each point with the other N − 1) and use odd k to avoid ties. The training error cannot be used: it always favours k = 1.

### EX.I2 – k-NN on a plotted dataset: k = 1, 3, 5; is k = 11 a good idea; pros/cons; quality assessment.
*Exam: 3 Jul 2023, Ex. 3. Theory: Lecture 8, slides 9–15*

(a) Count the labels of the k nearest neighbours of the test point in the figure: k = 1 ⇒ the single nearest is negative ⇒ **negative**; k = 3 ⇒ two negatives, one positive ⇒ **negative**; k = 5 ⇒ two negatives, three positives ⇒ **positive**. The answer changes with k.
(b) **No.** There are only 5 negative instances in the training set. With k ≥ 11, any test point has at least 6 positive neighbours, so it is always classified positive: the classifier becomes constant (the k = N limit, maximum bias) and ignores the position of the point. k should be chosen by cross-validation (rule of thumb ⌊√N⌋).
(c) Pros and cons: see EX.I1(b) (and L8.Q16).
(d) Do not rely on the training misclassification error (optimistic; 1-NN has E_in = 0): estimate E_out on a validation/test set or by cross-validation (Lecture 6). Look at the **confusion matrix** (true/false positives and negatives) and at **precision** a/(a + b), **recall** a/(a + c) and **F1**, which reveal the type of errors, especially with unbalanced classes or different error costs.

### EX.I3 – k-NN labels for k = 1, 3, 5; pros/cons; how to select k.
*Exam: 7 Sep 2023, Ex. 3. Theory: Lecture 8, slides 9–11*

(a) k = 1: the nearest point is a "+" just to the left ⇒ **positive**; k = 3: that "+" plus the two "o" to the right ⇒ two negatives ⇒ **negative**; k = 5: two more "+" on the left ⇒ three positives vs two negatives ⇒ **positive**.
(b) See EX.I1(b).
(c) By cross-validation: for each candidate k (odd values), compute the validation or leave-one-out error and pick the minimum (the U-shaped E_cv of Lecture 6); starting point ⌊√N⌋; theoretical guideline k → ∞, k/N → 0.

### EX.I4 – Fruits (weight, colour intensity): describe k-NN; classify (160 g, 0.7) with k = 3; effect of k; compare with logistic regression and perceptron.
*Exam: 9 Jan 2025, Ex. 3. Data: Apple (150, 0.8), (170, 0.6), (140, 0.9); Orange (200, 0.4), (180, 0.5), (190, 0.3). Theory: Lecture 8, slides 5–12*

(a) Compute the distance from the new point to every training point, take the k closest, and assign the majority class.
(b) Euclidean distances: (150, 0.8) and (170, 0.6) at 10.0 (Apple), then **a tie** at 20.0 between (140, 0.9) Apple and (180, 0.5) Orange, then (190, 0.3) 30.0 and (200, 0.4) 40.0. The first two neighbours are apples, so the majority of the 3 nearest is **Apple** whatever the tie-break. Note: the official solution lists three apples as the three nearest, which ignores the tie. More importantly, the distance is dominated by the weight (differences of tens of grams vs tenths of intensity): colour intensity contributes almost nothing. Features should be scaled, e.g. a weighted distance with Q = diag(1/σ²_weight, 1/σ²_colour) (slide 6); with min-max scaling the ranking here does not change (Apple, Apple, then a tie), but in general it can.
(c) Small k: captures local patterns but is sensitive to noise and outliers (high variance). Large k: smoother, more stable, but it washes out local structure (high bias); k = N predicts the majority class everywhere. Rule of thumb k = ⌊√N⌋ (here ⌊√6⌋ = 2; use 3 to have an odd k), best chosen by cross-validation.
(d) k-NN vs logistic regression and perceptron: **k-NN** makes no assumption on the boundary (can be nonlinear and arbitrarily complex), needs no training, but stores all data, is slow at prediction, sensitive to k, to the metric and to scaling, and gives no interpretable model. **Logistic regression/perceptron** assume a linear boundary (or need hand-made features), require training, but are compact, fast to evaluate and interpretable (logistic regression also gives probabilities).

### EX.I5 – k-NN for urban noise zones: rule and why nonparametric; choice of k; distance metric and scaling; consistency and dimension.
*Exam: 11 Feb 2026, Ex. 2. Theory: Lecture 8, slides 6–12*

(a) Find the set N_k(x) of the k training points closest to x and vote: **ŷ(x) = sign(Σ_{i∈N_k(x)} y⁽ⁱ⁾)**. Nonparametric because no parametric form h(x, w) is assumed and no finite-dimensional parameter vector is estimated: the prediction uses the stored data directly, and the "model" grows with N.
(b) k = 1: low bias, high variance (sensitive to noise and to individual samples). Increasing k: smoother rule, lower variance, higher bias. k = N: always the majority class. An intermediate k (≈ √N, refined by cross-validation) gives the best generalization.
(c) The distance decides which samples are "nearest" (Euclidean, weighted Euclidean (x − x*)ᵀQ(x − x*), cosine similarity). k-NN depends on the relative magnitude of the features: a feature with a larger numerical range (e.g. sound pressure in dB vs a normalized spectral descriptor) dominates the distance and distorts the neighbourhoods. Scaling (standardization, or Q = diag(1/σ²ᵢ)) gives each feature a comparable influence.
(d) Consistency (slide 11): N → ∞, k → ∞ and k/N → 0 ⇒ the error converges to the minimum achievable error. Dimension d (extra, curse of dimensionality): as d grows the volume of the space grows exponentially, the data become sparse and the "nearest" neighbours are no longer close, so exponentially more data are needed; irrelevant features make it worse.

### EX.I6 – Why and when use precision and recall instead of only accuracy?
*Exam: 7 Nov 2022, Ex. 1d. Theory: Lecture 8, slides 14–15*

Accuracy = (a + d)/N hides **which** errors are made. With **unbalanced classes**, a classifier that always predicts the majority class can have high accuracy and be useless: if 1% of the population is infected, "everyone negative" is 99% accurate but has recall a/(a + c) = 0 (it finds no infected person). When the **costs of errors differ**, look at the relevant metric: for a medical test, missing a sick person (false negative) is serious ⇒ high **recall**; for a spam filter (spam = positive), losing a legitimate email in the spam folder (false positive) is serious ⇒ high **precision** a/(a + b), even at the cost of imperfect recall. F1 = 2·PPV·TPR/(PPV + TPR) summarizes both. The threshold of the classifier (e.g. ε in logistic regression) moves the balance between them.

### EX.I7 – k-NN or k-means? (classification part)
*Exam: 7 Nov 2022, Ex. 4a (complete answer with k-means in EX.J1). Theory: Lecture 8, slides 3–9*

k-NN is a **supervised classification** method: it needs labelled training data. (i) No labels, looking for outlier images ⇒ k-NN cannot be used (a clustering/unsupervised method such as k-means is needed: see EX.J1). (ii) Labels cat/dog available, classify a new image ⇒ **k-NN**: assign the majority label of the k nearest labelled images.

---

## J. Clustering

### EX.J1 – k-NN or k-means for images; Lloyd's algorithm and choice of k; how to avoid choosing k.
*Exam: 7 Nov 2022, Ex. 4. Theory: Lecture 9, slides 2 and 6–9; Lecture 8, slides 5–7*

(a)(i) No labels, find images very different from the rest (outliers) ⇒ **k-means** (clustering, unsupervised); k-NN cannot be used because it needs labels. Outliers show up as points very far from their centroid (large ‖x⁽ⁱ⁾ − μ_j‖²) or as tiny clusters of their own. (a)(ii) Labels cat/dog available, classify a new image ⇒ **k-NN** (supervised classification); k-means is not a classifier.

(b) **Lloyd's algorithm** (k given): (1) initialize μ₁, …, μ_k (random data points or, better, farthest-point initialization); (2) assign each point to the nearest centroid, S_j = points closest to μ_j; (3) update μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾; (4) repeat (2)–(3) until E_in = Σ_j Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖² stops decreasing. Both steps cannot increase E_in, so it converges in a finite number of steps to a local minimum. **Choice of k** ("knee" method): run k-means for k = 1, 2, 3, …, plot E_in(k); it always decreases (E_in = 0 for k = N), so choose the k at the knee, where the fast decrease turns into a slow one.

(c) Yes: **hierarchical clustering**. Merge the closest clusters step by step from N clusters to 1; the dendrogram gives E_in for every k, and the clustering for any k is obtained by cutting the tree (good cut: below a big jump of E_in).

### EX.J2 – Formulate the k-means problem rigorously and describe a training algorithm.
*Exam: 6 Feb 2023, Ex. 4a. Theory: Lecture 9, slides 2 and 4–6*

Given N points x⁽¹⁾, …, x⁽ᴺ⁾ ∈ ℝᵈ (no labels) and the number of clusters k, find a partition of {1, …, N} into k disjoint non-empty sets S₁, …, S_k and centroids μ₁, …, μ_k ∈ ℝᵈ that solve

min_{S₁,…,S_k, μ₁,…,μ_k} E_in = Σ_{j=1}^{k} Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖²,

with S_j ∩ S_l = ∅ for j ≠ l and S₁ ∪ … ∪ S_k = {1, …, N}. For fixed sets the optimal centroids are the means μ_j = (1/N_j) Σ_{i∈S_j} x⁽ⁱ⁾ (gradient −2Σ(x⁽ⁱ⁾ − μ_j) = 0), so the problem is a search over partitions: it is **NP-hard**. **Training algorithm (Lloyd):** initialize the centroids (farthest-point rule); repeat the assignment step (each point to the nearest centroid) and the update step (each centroid = mean of its cluster) until E_in stops decreasing. Each step cannot increase E_in ⇒ convergence in a finite number of iterations to a local minimum; restart from different initializations and keep the lowest E_in. k, if unknown, is chosen with the knee of E_in(k).

### EX.J3 – One advantage and one disadvantage of hierarchical clustering compared to k-means.
*Exam: 6 Feb 2023, Ex. 4b. Theory: Lecture 9, slides 6–9*

**Advantages:** no need to fix k in advance (the dendrogram gives the result for every k: "returns performance for each k"); a readable picture of the nested groups; no initialization, so the result is deterministic. **Disadvantages:** it is much slower and needs all pairwise distances (O(N²) memory, at least O(N²) time), so it does not scale to very large N; it is greedy (a merge is never undone), so for a given k it may be worse than k-means; k-means is faster and its convergence is formally guaranteed.

### EX.J4 – Image compression with k-means (k = 16 colours).
*Exam: 12 Nov 2024, Ex. 4. Theory: Lecture 9, slides 4–9*

Each pixel is a point x⁽ⁱ⁾ = (R, G, B) ∈ ℝ³; N = number of pixels; each centroid μ_j is a colour.

(a) **Initialization:** choose 16 initial centroids, e.g. 16 pixel colours at random or with the farthest-point rule (first colour random, then each time the colour farthest from those already chosen, so the palette covers the whole colour range). Lloyd only reaches a local minimum of E_in, and the minimum reached depends on the starting point: different initializations give different palettes and so slightly different compressed images (e.g. a rare but visible colour may get its own centroid or be merged into a neighbouring one). Remedy: several restarts, keep the lowest E_in.

(b) **Assignment step:** each pixel is assigned to the nearest centroid colour in RGB space (squared Euclidean distance), giving 16 groups of similar pixels. **Update step:** each centroid becomes the mean colour (mean R, mean G, mean B) of the pixels assigned to it. Repeat until the assignments (or E_in) stop changing; neither step can increase E_in.

(c) **Reconstruction:** store the palette (16 colours × 3 bytes) and, for every pixel, only the index of its centroid (4 bits, since 2⁴ = 16); to display the image, replace each pixel by the colour of its centroid. Extra: for a 1000 × 1000 image, 24 bits/pixel = 24·10⁶ bits become 4·10⁶ + 16·24 bits ≈ **6 times smaller**.

(d) **Advantage:** simple, fast (O(N·k·d) per iteration) and the compression ratio is controlled directly by k. **Disadvantage:** loss of quality: smooth gradients and subtle shades become visible bands, since only 16 colours remain; plus dependence on the initialization and on the choice of k (and the squared RGB distance is not the distance perceived by the eye).

(e) **Choice of k (knee/elbow):** run k-means for k = 2, 4, 8, 16, 32, …; for each k compute E_in = Σ_j Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖² (the total squared colour error of the compressed image, i.e. N × mean squared error) together with the size of the file (log₂k bits per pixel); plot E_in against k and choose the knee, where increasing k no longer improves the quality much. A visual check of the compressed images completes the evaluation.

### EX.J5 – How can unsupervised learning help reduce the data for house-price prediction? Why is it a good idea, what are the limits? (clustering part)
*Exam: 13 Feb 2025, Ex. 3.1 (PCA part in EX.K5). Theory: Lecture 9, slides 3–9 and 10–14*

**How:** apply k-means to the houses (or to the neighbourhoods) using their features: minimize J = Σ_{j=1}^{k} Σ_{x∈S_j} ‖x − μ_j‖² with Lloyd's algorithm. Then (i) replace the many raw features by the cluster label (one categorical feature, or k indicator features) or by the distance to each centroid, and/or (ii) train one simpler model per cluster (slide 3: piecewise models), or summarize groups of similar data by their centroids. **Why it is a good idea:** fewer inputs ⇒ less computation, fewer parameters, so lower variance and less overfitting (Lectures 5–6); the clusters reveal structure (market segments, like slides 12–14) and improve interpretability. **Limits:** information is lost (all houses of a cluster look the same to the model); the clustering ignores the price (unsupervised), so the groups may be irrelevant for prediction; k must be chosen (knee); the result depends on the initialization and on feature scaling; with thousands of features Euclidean distances become less meaningful. Note: strictly, clustering reduces the number of distinct points or replaces features with a label; reducing the number of features while keeping the variance is the job of PCA.

### EX.J6 – Could k-means have been used instead of PCA to reduce the house dataset to two dimensions? (k-means part)
*Exam: 9 Jan 2025, Ex. 4d (PCA parts a–c in EX.K2–K4). Theory: Lecture 9, slides 2 and 4–6*

Not for the same task. Different objectives: **k-means** groups the **data points** (the houses, rows of the dataset) into k clusters by minimizing the within-cluster squared distance Σ_j Σ_{i∈S_j} ‖x⁽ⁱ⁾ − μ_j‖²; its output is a discrete label per house (and k centroids), not two continuous coordinates. **PCA** reduces the number of **features** (columns), projecting every house onto a few directions that keep the maximum variance. k-means could at most summarize each house by its cluster, losing all the variation inside the cluster.

---

## K. Data pre-processing and PCA

### EX.K1 – PCA and bounding the number of non-zero parameters for feature selection (PCA part).
*Exam: 7 Nov 2022, Ex. 3a (‖w‖₀ part and 3b–3c in EX.G7). Theory: Lecture 10, slides 8–12; Lecture 6, slides 4–5*

**PCA:** center (and standardize) the inputs, compute the SVD X = UΓVᵀ, and replace the m inputs with the k < m principal components Ẑ = XV_k, the uncorrelated directions with the largest singular values (largest variance γ_i²/N). It reduces the number of inputs while keeping most of the information (the variance discarded is Σ_{i>k} γ_i²). **‖w‖₀ ≤ s:** train the model using at most s non-zero weights, so only s original features are used (combinatorial, approximated with ℓ₂ regularization, EX.G7). **Why both work for the task:** both reduce the number of effective parameters of the model (m + 1 → k + 1 or s + 1), which reduces the generalization gap O(m/N) and the variance (Lectures 2, 5, 6). Difference: ‖w‖₀ *selects* original features using the output y (supervised), PCA *builds* new features as combinations of all the inputs without looking at y (unsupervised, feature extraction).

### EX.K2 – Purpose of PCA and its main steps (houses: size, rooms, age, distance).
*Exam: 9 Jan 2025, Ex. 4a. Theory: Lecture 10, slides 4 and 9–12*

**Purpose:** reduce the dimensionality of the dataset (4 features → fewer) while retaining as much variability (information) as possible, by finding new uncorrelated features (principal components) along the directions of the largest fluctuations of the data. **Steps:** (1) **standardize**: subtract the mean of each feature (centering) and divide by its standard deviation (normalization), which is essential here because the features have different units (m², number, years, km); (2) compute the SVD of the standardized data matrix, X = UΓVᵀ (equivalently the eigenvectors of the covariance A = XᵀX/N); (3) order the singular values γ₁ ≥ γ₂ ≥ … and the corresponding columns of V (principal directions); (4) choose k (knee of the γ_i or share of variance Σ_{i≤k} γ_i²/Σ γ_i², here k = 2); (5) project: Ẑ = XV_k.

### EX.K3 – How is the SVD used in PCA? Why focus on the largest singular values?
*Exam: 9 Jan 2025, Ex. 4b. Theory: Lecture 10, slides 10–12*

With centered data X = UΓVᵀ (U orthonormal, V orthogonal, Γ diagonal with γ₁ ≥ … ≥ 0). The covariance is A = (1/N)XᵀX = V(Γ²/N)Vᵀ, so the **columns of V are the principal directions** (loadings) and the variance of the data along v_i is **γ_i²/N**. The variance of a projection z = vᵀx is vᵀAv, which is maximized by v = v₁ (value γ₁²/N); the next best direction orthogonal to it is v₂, and so on. Hence the singular values measure the "importance" of each component: the components with the largest γ_i capture most of the variance (most of the differences between the data points), while those with small γ_i contain little variation, mostly noise, and can be dropped losing only Σ_{i>k} γ_i².

### EX.K4 – How is the original dataset transformed into the two-dimensional space?
*Exam: 9 Jan 2025, Ex. 4c. Theory: Lecture 10, slide 12 and slides 14–17*

Take the first two columns of V (the loadings of PC1 and PC2), V₂ ∈ ℝ^{4×2}, and multiply the standardized data matrix X ∈ ℝ^{N×4} by it: **Ẑ = X·V₂ ∈ ℝ^{N×2}**. Each house i is represented by two numbers, ẑ⁽ⁱ⁾ = V₂ᵀx⁽ⁱ⁾, i.e. its coordinates along the two principal directions (each a weighted combination of size, rooms, age and distance). A new house is first standardized with the training mean and standard deviation and then projected in the same way. Example of the same operation: USArrests (4 variables → 2 components, Lecture 10, slide 14).

### EX.K5 – How can unsupervised learning help reduce the data for house-price prediction? (PCA part)
*Exam: 13 Feb 2025, Ex. 3.1 (clustering part in EX.J5). Theory: Lecture 10, slides 8–12*

**How:** standardize the thousands of features and apply PCA: compute the SVD of the training data, keep the k components that explain most of the variance (e.g. 90–95%), and train the regression model on Ẑ = XV_k (k ≪ number of features). **Why it is a good idea:** much less computation; far fewer parameters ⇒ lower variance and less overfitting (Lectures 5–6); correlated/redundant features are merged into uncorrelated components; small components (often noise) are removed. **Limitations:** some information is lost (Σ_{i>k} γ_i²); PCA is unsupervised, so it keeps the directions of largest variance, not necessarily those most related to the price; it is linear; it depends on scaling and is sensitive to outliers; the components are harder to interpret than the original features; k must be chosen (knee, share of variance, or cross-validation).

### EX.K6 – Machine measurements (vibration, acoustic emission) along one direction: PCA objective, projection, information lost, limitation.
*Exam: 11 Feb 2026, Ex. 4. Theory: Lecture 10, slides 9–12*

(a) **Objective:** find orthogonal directions in the feature space along which the variance of the (centered) data is maximal, and use the first ones as new, fewer features. The **first principal component** is the direction v₁ that maximizes Var[v₁ᵀx] = v₁ᵀAv₁: here it is the line along which the points are aligned (the dominant direction of joint variation of vibration and acoustic emission, the "first PC" arrow of Fig. 1).
(b) Since the points are approximately aligned along v₁, almost all their variance is along v₁ (γ₁ ≫ γ₂) and the spread in the orthogonal direction is small. The projection z = v₁ᵀx keeps the position of each measurement along the line, i.e. the differences among the points (slide 9: z₁ alone "explains" the differences), while halving the number of inputs.
(c) What is lost is the component of each point orthogonal to v₁ (its distance from the line), i.e. the variance along the second principal direction, γ₂²/N; the total squared reconstruction error equals γ₂². If γ₂² is small compared with γ₁² + γ₂², the loss is limited (e.g. γ = (6, 2): 90% of the variance kept, 10% lost). Note: the deviations from the line may sometimes be important (e.g. an anomaly of the machine could show up exactly there).
(d) **Limitation:** PCA is unsupervised: it does not use the output (e.g. faulty/healthy), so the direction of largest variance may not be the most useful for prediction or classification. Others: only linear projections; sensitive to scaling (vibration and acoustic levels have different units, standardize first) and to outliers.

---

## L. Stochastic processes

### EX.L1 – Spectral density: definition, properties, variance from its plot; estimation from data.
*Exam: 3 Jul 2023, Ex. 4. Theory: Lecture 11, slides 18–22; Lecture 12, slides 11–15*

(a) For a stationary SP, **Γ(ω) = Σ_{τ=−∞}^{+∞} γ(τ)e^(−jωτ)**. Properties: real, Γ(ω) ≥ 0, even (Γ(ω) = Γ(−ω)), periodic with period 2π. Variance from the plot: by the inverse transform at τ = 0, **γ(0) = (1/2π)∫_{−π}^{π}Γ(ω)dω**, so measure the area under Γ on [−π, π] and divide by 2π (by symmetry: twice the area on [0, π], divided by 2π). Example: white noise, Γ = λ² ⇒ γ(0) = λ².
(b)–(c) *Estimation from data: Lecture 12.* From one realization of length N one can compute the sample covariance γ̂(τ) and its transform, or the **periodogram** Γ̂(ω) = (1/N)|Σ_t x(t)e^(−jωt)|². It is asymptotically unbiased but **not a consistent estimator**: its variance does not go to 0 as N grows, so the plot stays noisy. With a large dataset, reduce the variance by **averaging**: split the data into sub-batches, compute a periodogram on each and average them (Bartlett/Welch), or smooth the spectrum (windowing γ̂(τ)). The price is a lower frequency resolution.

### EX.L2 – Output of a stable filter: sample mean, covariance, spectrum; consistency; linear trend.
*Exam: 13 Feb 2025, Ex. 4. Theory: Lecture 11, slides 6–10 and 18; Lecture 12, slides 5–15 and 18–20*

y_t = H(q)w_t with w ~ WN(0, 1) and H stable is a stationary process.
1. **Sample estimators**: μ̂ = (1/N)Σ y_t; γ̂(τ) = (1/N)Σ_{t=1}^{N−τ}(y_t − μ̂)(y_{t+τ} − μ̂); periodogram Γ̂(ω) = (1/N)|Σ_t y_t e^(−jωt)|², ω ∈ [−π, π].
2. **Consistency** (N → ∞): μ̂ is consistent; γ̂(τ) is asymptotically consistent but gets worse for large τ (few products to average); the periodogram is **not consistent** (high variance), improved by smoothing/averaging.
3. **Linear trend**: y_t = α + βt + ỹ_t is **not stationary** (the mean changes with t). Estimate α, β by **OLS** on (t, y_t) (EX.B7), subtract the trend ỹ_t = y_t − (α̂ + β̂t), then compute μ̂, γ̂, Γ̂ on ỹ_t.

### EX.L3 – Temperature as a stationary process: estimators, PSD estimation, trends and seasonality (true/false).
*Exam: 21 Jan 2026, Ex. 4. Theory: Lecture 11, slides 6–10, 18–23; Lecture 12, slides 5–22*

(a) μ̂ = (1/N)Σ x(t); γ̂(τ) = (1/N)Σ_{t=1}^{N−|τ|}(x(t) − μ̂)(x(t + |τ|) − μ̂), |τ| < N (1/(N − |τ|) is also used). Consistent if the process is stationary and γ(τ) → 0 as |τ| → ∞.
(b) Two ways: transform the estimated covariance, Γ̂(ω) = Σ_{|τ|<N} γ̂(τ)e^(−jωτ); or the periodogram Γ̂(ω) = (1/N)|X_N(ω)|² with X_N(ω) = Σ x(t)e^(−jωt). Theory: **Γ(ω) = Σ γ(τ)e^(−jωτ)**, the spectrum is the Fourier transform of the covariance (and γ(0) = (1/2π)∫Γ).
(c) (i) **True**: a linear trend makes the mean time-dependent, violating m(t) = m. (ii) **False**: removing the trend fixes only the mean; seasonality or a time-varying variance may remain. (iii) **True**: a periodic component gives discrete spectral lines (peaks) at its frequency and harmonics. (iv) **True**: estimate the periodic component (e.g. average over periods) and subtract it.

---

# Appendix – Exam topics not covered yet

These appear in past exams but are not in Lectures 1–12.

- **"Write your own question"** (Sep 23 Ex. 4).

---

## Template for new lectures

(Copy this block, fill it in, and paste it before "Big picture".)

```
## Lecture N – Title

### Key formulas
- ...

### LN.Q1 – Question
*Source: Lecture N, slide X*
Answer.

### LN.Q2 – Question
*Source: Lecture N, slides X–Y*
Answer.

---
```
