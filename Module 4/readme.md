## Module 4: Linear Regression with Multiple Variables

Welcome! In this module we upgrade linear regression to use many features (a.k.a. multiple variables). You’ll learn clean notation, how to write the hypothesis in vector form, and how to reason about dimensions.

## 📚 Table of Contents
- [Lecture 1: Multiple Features](#lecture-1-multiple-features)
  - [Why multiple features?](#why-multiple-features)
  - [Notation: m, n, x, y](#notation-m-n-x-y)
  - [Feature vectors and indexing](#feature-vectors-and-indexing)
  - [Add a bias feature x₀ = 1](#add-a-bias-feature-x₀--1)
  - [Hypothesis: expanded and vector form](#hypothesis-expanded-and-vector-form)
  - [Shapes at a glance](#shapes-at-a-glance)
  - [Tiny example](#tiny-example)
  - [Key takeaways](#key-takeaways)

---

## Lecture 1: Multiple Features

### Why multiple features?
Single‑feature regression used just one input (e.g., house size) to predict price. Real homes have more signals (bedrooms, bathrooms, age, location). Using multiple features lets the model combine these signals to make better predictions.

### Notation: m, n, x, y
Example dataset (snippet):

| Size (ft²) `x₁` | Bedrooms `x₂` | Floors `x₃` | Age (years) `x₄` | Price `y` ($1000) |
|---:|---:|---:|---:|---:|
| 2104 | 5 | 1 | 45 | 460 |
| 1416 | 3 | 2 | 40 | 232 |
| 1534 | 3 | 2 | 30 | 315 |
|  852 | 2 | 1 | 36 | 178 |
| … | … | … | … | … |

Notes:
- `n = 4` features (`x₁…x₄`).
- `m = 47` examples (rows) in the full dataset.
- The highlighted example in the slide: `x⁽²⁾ = [1416, 3, 2, 40]`, `y⁽²⁾ = 232`.

- `m`: number of training examples (rows in your dataset)
- `n`: number of features per example (columns of inputs)
- `x` (input): features; `y` (output): target value we want to predict

We’ll index training examples with superscripts (position in the dataset) and features with subscripts:
- `x⁽ⁱ⁾`: the feature vector of example `i`
- `x⁽ⁱ⁾ⱼ`: the value of feature `j` for example `i`

Example: if features are `[size, bedrooms, bathrooms, age]` then for the 2nd house:
```
x⁽²⁾ = [1416, 3, 2, 40]
```
Here, `x⁽²⁾₃ = 2` (third feature for example 2).

### Feature vectors and indexing
We treat each training example as a column vector of its features. To keep formulas simple we’ll add a bias feature next.

Example (without bias):
```
x⁽²⁾ = ⎡ 1416 ⎤
       ⎢   3  ⎥
       ⎢   2  ⎥
       ⎣  40  ⎦   (4 × 1)
```

With bias `x₀ = 1`:
```
x⁽²⁾ = ⎡   1  ⎤
       ⎢ 1416 ⎥
       ⎢   3  ⎥
       ⎢   2  ⎥
       ⎣  40  ⎦   (5 × 1)
```

### Add a bias feature x₀ = 1
We define an extra feature that’s always 1:
```
x₀ = 1  for every example
```
This lets the model learn an intercept term `θ₀` naturally. With the bias, each feature vector has `n+1` entries: `[x₀, x₁, …, xₙ]`.

### Hypothesis: expanded and vector form
With multiple features the hypothesis is a weighted sum of all features plus the bias term:

Expanded form (for `n` features):
```
h_θ(x) = θ₀·x₀ + θ₁·x₁ + θ₂·x₂ + ··· + θₙ·xₙ
```

Vector form (compact and preferred):
```
h_θ(x) = θᵀ x
```
Where
θ = [θ₀, θ₁, …, θₙ]ᵀ   and   x = [x₀, x₁, …, 
xₙ]ᵀ  with  x₀ = 1
```
θᵀ = [ θ₀  θ₁  …  θₙ ]   (1 × (n+1))

θ = ⎡ θ₀ ⎤
    ⎢ θ₁ ⎥
    ⎢  ⋮  ⎥   (n+1 × 1)
    ⎣ θₙ ⎦

x = ⎡ x₀ ⎤
    ⎢ x₁ ⎥
    ⎢  ⋮  ⎥   (n+1 × 1), with x₀ = 1
    ⎣ xₙ ⎦
```

Matrix product equals the hypothesis (a single scalar):
```
[ θ₀  θ₁  …  θₙ ]
⎡ x₀ ⎤
⎢ x₁ ⎥
⎢  ⋮  ⎥  =  h_θ(x)
⎣ xₙ ⎦
```

Side-by-side view:
```
θᵀ = [ θ₀  θ₁  …  θₙ ]      x = ⎡ x₀ ⎤
                                ⎢ x₁ ⎥
                                ⎢  ⋮ ⎥
                                ⎣ xₙ ⎦
h_θ(x) = θᵀ x
h_θ(x) = θ₀·x₀ + θ₁·x₁ + θ₂·x₂ + ··· + θₙ·xₙ
```

From the slide (pulling all pieces together):
```
Define:  x₀ = 1  (bias feature)


x = [x₀, x₁, …, xₙ]ᵀ  ∈ ℝⁿ⁺¹
θ = [θ₀, θ₁, …, θₙ]ᵀ  ∈ ℝⁿ⁺¹

θᵀ = [θ₀, θ₁, …, θₙ]  ∈ ℝ¹×⁽ⁿ⁺¹⁾

h_θ(x) = θ₀x₀ + θ₁x₁ + ··· + θₙxₙ = θᵀ x
```

### Shapes at a glance
```
θᵀ: 1 × (n+1)
x : (n+1) × 1
→ h_θ(x): scalar (1 × 1)
```

### Tiny example
Suppose we use 4 features for houses: size (sqft), bedrooms, bathrooms, age (years). Let
```
θ = [80, 0.10, 0.01, 3.0, −2.0]  (interpreting prices in thousands)
     θ₀   θ₁    θ₂   θ₃   θ₄

x = [1, x₁, x₂, x₃, x₄] = [1, 1416, 3, 2, 40]

h_θ(x) = θᵀx
        = 80 + 0.10·1416 + 0.01·3 + 3.0·2 + (−2.0)·40
        = 80 + 141.6 + 0.03 + 6 − 80
        = 147.63  (thousands)
```
So the predicted price is about 147,630 (same units as the training `y`).

### Key takeaways
- Use `n` for number of features, `m` for number of examples.
- Add a bias feature `x₀ = 1` so the model can learn an intercept `θ₀`.
- Write the hypothesis compactly as `h_θ(x) = θᵀx`.
- Always track shapes; vector form makes implementation and reasoning easier.


