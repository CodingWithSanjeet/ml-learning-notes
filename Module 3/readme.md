## Module 3: Linear Algebra Review

Welcome! In this module we quickly review the linear‑algebra building blocks you’ll use throughout machine learning. We’ll keep the language simple and add small, concrete examples you can refer back to at any time.

## 📚 Table of Contents
- [Lecture 1: Matrices and Vectors](#lecture-1-matrices-and-vectors)
  - [Matrices (definitions and notation)](#matrices-definitions-and-notation)
    - [What is a matrix?](#what-is-a-matrix)
    - [Matrix dimensions](#matrix-dimensions)
    - [Set notation](#set-notation)
    - [Referring to entries: Aᵢⱼ](#referring-to-entries-aᵢⱼ)
  - [Vectors (definitions and notation)](#vectors-definitions-and-notation)
    - [What is a vector?](#what-is-a-vector)
    - [Vector dimension and Rⁿ](#vector-dimension-and-rⁿ)
    - [Vector examples](#vector-examples)
    - [Vector indexing: yᵢ (and 1‑ vs 0‑indexing)](#vector-indexing-y-and-1--vs-0--indexing)
  - [Common notation conventions](#common-notation-conventions)
  - [Quick reference](#quick-reference)
- [Lecture 2: Addition and Scalar Multiplication](#lecture-2-addition-and-scalar-multiplication)
  - [1. Matrix addition and subtraction](#1-matrix-addition-and-subtraction)
  - [2. Scalar multiplication and division](#2-scalar-multiplication-and-division)
  - [3. Worked example: combine add/subtract and scalars](#3-worked-example-combine-addsubtract-and-scalars)
  - [4. Dimension rules to remember](#4-dimension-rules-to-remember)
  - [5. Key takeaways](#5-key-takeaways)

- [Lecture 3: Matrix-Vector Multiplication](#lecture-3-matrix-vector-multiplication)
  - [What is matrix-vector multiplication?](#what-is-matrix-vector-multiplication)
  - [Dimension rule (when A x is defined)](#dimension-rule-when-a-x-is-defined)
  - [How to compute y = A x](#how-to-compute-y--a-x)
  - [Beginner-friendly example (step-by-step)](#beginner-friendly-example-step-by-step)
  - [Worked example 1 (3×2)·(2×1)](#worked-example-1-32·21)
  - [Worked example 2 (3×4)·(4×1)](#worked-example-2-34·41)
  - [Quick analogy](#quick-analogy)
  - [Predict many outputs at once (design matrix)](#predict-many-outputs-at-once-design-matrix)
  - [Application: Predicting House Prices](#application-predicting-house-prices)
  - [Key takeaways (Lecture 3)](#key-takeaways-lecture-3)

- [Lecture 4: Matrix-Matrix Multiplication](#lecture-4-matrix-matrix-multiplication)
  - [What is matrix-matrix multiplication?](#what-is-matrix-matrix-multiplication)
  - [Dimension rule and output shape](#dimension-rule-and-output-shape)
  - [How to compute C = A B](#how-to-compute-c--a-b)
  - [Worked example 1 (2×3)·(3×2)](#worked-example-1-23·32)
  - [Worked example 2 (2×2)·(2×2)](#worked-example-2-22·22)
  - [Apply many hypotheses at once (Ŷ = X Θ)](#apply-many-hypotheses-at-once-ŷ--x-θ)
  - [Application to Linear Regression](#application-to-linear-regression)
  - [Efficiency](#efficiency)
  - [Key takeaways (Lecture 4)](#key-takeaways-lecture-4)

- [Lecture 5: Matrix Multiplication Properties](#lecture-5-matrix-multiplication-properties)
  - [Matrix Multiplication Is Not Commutative: AB ≠ BA](#matrix-multiplication-is-not-commutative-ab--ba)
  - [Matrix Multiplication Is Associative: A(BC) = (AB)C](#matrix-multiplication-is-associative-abc--abc)
  - [Identity matrix I](#identity-matrix-i)
  - [Dimension reminders](#dimension-reminders)
  - [Quick examples](#quick-examples)
  - [Key takeaways (Lecture 5)](#key-takeaways-lecture-5)

---

## Lecture 1: Matrices and Vectors

### Matrices (definitions and notation)

#### What is a matrix ?
A matrix is a rectangular array of numbers written between square brackets. Think of it as a 2‑D table of numbers.

Example matrices:

```
⎡ 1   9 ⎤
⎢ 4   1 ⎥
⎢ 0   1 ⎥
⎣ 2   7 ⎦   ← 4 × 2 matrix

⎡ 3  4  5 ⎤
⎣ 6  7  8 ⎦   ← 2 × 3 matrix
```

#### Matrix dimensions
- We write dimensions as `Number of rows × Number of columns`.
- The first matrix above has 4 rows and 2 columns → 4 × 2.
- The second has 2 rows and 3 columns → **2 × 3 matrix**.

#### Set notation
  - $\mathbb{R}^{m\times n}$ → the set of all real‑valued matrices with $m$ rows and $n$ columns
  - Examples:
    - $\mathbb{R}^{4\times 2}$ → all 4 × 2 real matrices
    - $\mathbb{R}^{2\times 3}$ → all 2 × 3 real matrices

#### Referring to entries: Aᵢⱼ
If `A` is a matrix, `Aᵢⱼ` (read “A i j”) means the entry in row `i`, column `j`.

- `A₁₁` → row 1, column 1
- `A₃₂` → row 3, column 2
- If an index is outside the matrix size (e.g., ask for column 3 in a 2‑column matrix), that entry is “undefined” (an error).

Example:

```
A = ⎡ 2  7  5 ⎤
    ⎣ 9  4  1 ⎦

A₁₂ = 7   (row 1, col 2)
A₂₃ = 1   (row 2, col 3)
A₂₁ = 9   (row 2, col 1)
```

### Vectors (definitions and notation)

#### What is a vector?
A vector is a special case of a matrix that has only one column — an `n × 1` matrix (a column vector).

Example (a 4‑dimensional vector):

```
⎡ 460 ⎤
⎢ 232 ⎥
⎢ 315 ⎥
⎣ 178 ⎦   ← 4 × 1 vector
```

#### Vector dimension and Rⁿ
- A vector with `n` entries is called an `n‑dimensional vector` and belongs to `Rⁿ` (all length‑`n` real vectors).

-  $\mathbb{R}^{4}$: The set of all 4-dimensional real-valued vectors.

#### Vector examples
- 2‑D vector in $\mathbb{R}^2$ (e.g., a point on a plane):

```
⎡ 3 ⎤
⎣ 5 ⎦
```

- 3‑D vector in $\mathbb{R}^3$ (e.g., a point in 3D space):

```
⎡ 1 ⎤
⎢ 4 ⎥
⎣ 2 ⎦
```

- Feature vector (heights, weights, ages) as a 5‑D vector in $\mathbb{R}^5$:

```
⎡ 170 ⎤
⎢  65 ⎥
⎢  29 ⎥
⎢ 182 ⎥
⎣  72 ⎦
```

#### Vector indexing: yᵢ (and 1‑ vs 0‑indexing)
- If `y` is a vector, `yᵢ` means the `i`‑th entry of `y`.
- Two indexing conventions exist:
  - 1‑indexed: entries are `y₁, y₂, …, yₙ` (common in math)
  - 0‑indexed: entries are `y₀, y₁, …, yₙ₋₁` (common in some programming languages)
- In these notes we use 1‑indexed vectors unless stated otherwise. When coding, you’ll often see 0‑indexed arrays.

Example (1‑indexed):

```
y = ⎡ 4 ⎤
    ⎢ 6 ⎥
    ⎢ 3 ⎥
    ⎣ 8 ⎦

y₁ = 4,  y₂ = 6,  y₃ = 3,  y₄ = 8
```

### Common notation conventions
- Uppercase letters (A, B, X) → matrices
- Lowercase letters (a, b, x, y) → scalars or vectors (context will make it clear)

### Quick reference
- Matrix: 2‑D array of numbers
- Size: rows × columns (e.g., 4 × 2, 2 × 3)
- Entry: `Aᵢⱼ` = row `i`, column `j`
- Vector: `n × 1` matrix (column vector), element `yᵢ`
- Sets: matrices in `R^{m×n}`, vectors in `Rⁿ`
- Indexing: math → mostly 1‑indexed; code → often 0‑indexed

> Tip: Don’t worry about memorizing every symbol. Keep this page handy as a mini‑cheat‑sheet and refer back whenever you need a reminder.



---

## Lecture 2: Addition and Scalar Multiplication

In this lecture we learn how to add/subtract matrices and vectors, and how to multiply/divide them by a number (a “scalar”). All rules are simple and consistent once you remember the shape (dimensions).

### 1. Matrix addition and subtraction
- You add matrices element‑by‑element.
- You can only add (or subtract) matrices of the **same size**.
- Result has the **same size** as the inputs.

Addition:

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\ + \
\begin{bmatrix}
w & x \\
y & z
\end{bmatrix}
\ = \
\begin{bmatrix}
a{+}w & b{+}x \\
c{+}y & d{+}z
\end{bmatrix}
$$

Subtraction:

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\ - \
\begin{bmatrix}
w & x \\
y & z
\end{bmatrix}
\ = \
\begin{bmatrix}
a{-}w & b{-}x \\
c{-}y & d{-}z
\end{bmatrix}
$$

Example (3 × 2 matrices):

```
⎡ 1  2 ⎤   ⎡ 4  2 ⎤   ⎡ 5  4 ⎤
⎢ 3  0 ⎥ + ⎢ 0  5 ⎥ = ⎢ 3  5 ⎥
⎣ 5  1 ⎦   ⎣ 6  1 ⎦   ⎣11  2 ⎦
```
**Why this works:**
- Both matrices have the **same dimensions (3 × 2)**.
- The **sum** also has **dimension 3 × 2** (same as the inputs).

**Not allowed:** sizes differ (e.g., **3 × 2** plus **2 × 2**) → the sum is **undefined (error)**.

Vectors follow the same rule because a column vector is an `n × 1` matrix.

### 2. Scalar multiplication and division
- Multiply a matrix by a number by multiplying **each entry** by that number.
- Division by a number is the same as multiplying by its reciprocal.

Multiplication:

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\ . \
x= \
\begin{bmatrix}
a{.}x & b{.}x \\
c{.}x & d{.}x
\end{bmatrix}
$$


Division:

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\ / \
x= \
\begin{bmatrix}
a{/}x & b{/}x \\
c{/}x & d{/}x
\end{bmatrix}
$$

Example (multiply by 3):

```
3 · ⎡ 1  2 ⎤ = ⎡  3   6 ⎤
    ⎢ 3  0 ⎥   ⎢  9   0 ⎥
    ⎣ 5  1 ⎦   ⎣ 15   3 ⎦
```

Example (divide by 4):

```
(1/4) · ⎡ 4  0 ⎤ = ⎡ 1      0   ⎤
        ⎣ 6  3 ⎦   ⎣ 6/4   3/4  ⎦
```

Note : `3 × A = A × 3` for any matrix `A` and any real number. You can put the number before or after the matrix — the result is the same.

### 3. Worked example: combine add/subtract and scalars
Compute

```
3·⎡1⎤   +   ⎡0⎤   −   (1/3)·⎡3⎤
  ⎢4⎥       ⎢0⎥             ⎢0⎥
  ⎣2⎦       ⎣5⎦             ⎣2⎦
```

Steps:
1) Scalar multiply/divide

```
3 × [1, 4, 2]   = [3, 12, 6]
(1/3) × [3, 0, 2] = [1, 0, 2/3]
```

2) Add/Subtract element‑wise

```
[3, 12, 6] + [0, 0, 5] − [1, 0, 2/3]
= [3 − 1, 12 − 0, 11 − 2/3]
= [2, 12, 31/3]
```

Result: a 3 × 1 matrix (3‑D vector)

```
⎡  2   ⎤
⎢  12  ⎥
⎣ 31/3 ⎦
```

### 4. Dimension rules to remember
- Add/Subtract: sizes must match (`m×n` with `m×n`).
- Scalar × Matrix/Vector: always OK; output keeps the same size.

### 5. Key takeaways
- Addition/subtraction: element‑wise, same sizes only.
- Scalar multiplication/division: element‑wise; multiply every entry.
- Vectors follow the same rules since they are `n × 1` matrices.
- Always keep an eye on the dimensions before you operate.


---

## Lecture 3: Matrix-Vector Multiplication

### What is matrix-vector multiplication?
Given a matrix `A` and a vector `x`, their product `A x` is a new vector `y` formed by taking dot products of the rows of `A` with `x`.

Intuition: each row of `A` mixes the entries of `x` to produce one output number.

### Dimension rule (when A x is defined)
- If `A` is `m × n` and `x` is `n × 1` (an `n`‑dimensional column vector), then `y = A x` is `m × 1` (an `m`‑dimensional vector).
- The inner dimensions must match: the number of columns of `A` equals the number of rows of `x`.

### How to compute y = A x
To compute the `i`‑th entry of `y`, take the dot product of the `i`‑th row of `A` with `x`:

$$
y_i \;=\; A_{i,:} \cdot x \;=\; \sum_{j=1}^{n} A_{ij} \, x_j
$$

So `y` stacks these `m` dot products, one per row of `A`.

Handy template (3×2)·(2×1):

$$
\begin{bmatrix}
a & b \\
c & d \\
e & f
\end{bmatrix}
\ * \
\begin{bmatrix}
w \\
y
\end{bmatrix}
= \
\begin{bmatrix}
aw + by \\
cw + dy \\
ew + fy
\end{bmatrix}
$$

### Beginner-friendly example (step-by-step)
Imagine you track two scores for each student: math and science. You want one “overall score” that weighs math double and science single. For three students, the math/science scores are the rows of `A`, and your weights are the vector `x`:

```
A = ⎡ 70  80 ⎤   (Student 1: math 70, science 80)
    ⎢ 90  60 ⎥   (Student 2: math 90, science 60)
    ⎣ 50  95 ⎦   (Student 3: math 50, science 95)

x = ⎡ 2 ⎤       (Weight math by 2,
    ⎣ 1 ⎦        science by 1)
```

Overall scores `y = A x` (row-by-row):

```
y₁ = 70·2 + 80·1 = 140 + 80 = 220
y₂ = 90·2 + 60·1 = 180 + 60 = 240
y₃ = 50·2 + 95·1 = 100 + 95 = 195

y = ⎡ 220 ⎤
    ⎢ 240 ⎥
    ⎣ 195 ⎦
```

Each row (one student) dot‑products with the weights to get one overall score. That’s matrix‑vector multiplication.

### Worked example 1 (3×2)·(2×1)
Suppose

```
A = ⎡ 1  3 ⎤
    ⎢ 4  0 ⎥   (3 × 2),   x = ⎡ 1 ⎤ (2 × 1)
    ⎣ 2  1 ⎦                  ⎣ 5 ⎦
```

Compute `y = A x`:

```
y₁ = 1·1 + 3·5 = 1 + 15 = 16
y₂ = 4·1 + 0·5 = 4 + 0  = 4
y₃ = 2·1 + 1·5 = 2 + 5  = 7
```

So

```
y = ⎡ 16 ⎤
    ⎢  4 ⎥
    ⎣  7 ⎦   (3 × 1)
```

### Worked example 2 (3×4)·(4×1)
Let

```
A = ⎡  1   2   1   5 ⎤
    ⎢  0   3   0   4 ⎥   (3 × 4),   x = ⎡ 1 ⎤ (4 × 1)
    ⎣ −1   2   0   0 ⎦                  ⎢ 3 ⎥
                                        ⎢ 2 ⎥
                                        ⎣ 1 ⎦
```

Compute `y = A x`:

```
y₁ = 1·1 + 2·3 + 1·2 + 5·1 = 1 + 6 + 2 + 5  = 14
y₂ = 0·1 + 3·3 + 0·2 + 4·1 = 0 + 9 + 0 + 4  = 13
y₃ = (−1)·1 + 2·3 + 0·2 + 0·1 = −1 + 6 + 0 + 0 = 5
```

So

```
y = ⎡ 14 ⎤
    ⎢ 13 ⎥
    ⎣  5 ⎦   (3 × 1)
```

### Predict many outputs at once (design matrix)
Suppose you have features `x` (e.g., house size) and hypothesis `h_θ(x) = θ₀ + θ₁ x`. For many inputs `x¹, …, xᵐ`, stack them into a design matrix with a leading column of ones:

```
X = ⎡ 1   x¹ ⎤
    ⎢ 1   x² ⎥
    ⎢ ⋮   ⋮   ⎥   (m × 2),   θ = ⎡ θ₀ ⎤ (2 × 1)
    ⎣ 1   xᵐ ⎦                  ⎣ θ₁ ⎦
```

Then all predictions at once are

```
ŷ = X θ  (m × 1)
```

Example (4 houses):

```
X = ⎡ 1  2104 ⎤
    ⎢ 1  1416 ⎥
    ⎢ 1  1534 ⎥
    ⎣ 1   852 ⎦    ,    θ =  ⎡ −40 ⎤
                            ⎣ 0.25 ⎦

Here’s `h_θ(x) = −40 + 0.25·x` applied to your house sizes:

 x = 2104 ⇒ h_θ(x) = −40 + 0.25·2104 = 486
 x = 1416 ⇒ h_θ(x) = −40 + 0.25·1416 = 314
 x = 1534 ⇒ h_θ(x) = −40 + 0.25·1534 = 343.5
 x =  852 ⇒ h_θ(x) = −40 + 0.25· 852 = 173

Vectorized form:


⎡ 1  2104 ⎤   ⎡ −40 ⎤    ⎡ 486   ⎤
⎢ 1  1416 ⎥ · ⎣ 0.25⎦ =  ⎢ 314   ⎥
⎢ 1  1534 ⎥             ⎢ 343.5 ⎥
⎣ 1   852 ⎦             ⎣ 173   ⎦

```

This vectorized form computes all `m` predictions in one matrix‑vector product, which is both simpler to write and typically faster in practice.

### Application: Predicting House Prices

Suppose we want to compute predictions for many houses at once using a linear model `h_θ(x) = θ₀ + θ₁ x`.

Construct the design matrix with a bias column of ones:

```
X = ⎡ 1  x₁ ⎤
    ⎢ 1  x₂ ⎥
    ⎢ 1  x₃ ⎥   (m × 2)
    ⎣ 1  x₄ ⎦
```

Parameters as a column vector:

```
θ = ⎡ θ₀ ⎤
    ⎣ θ₁ ⎦   (2 × 1)
```

Predict all `m` prices in one multiplication:

```
ŷ = X θ   (m × 1)
```

Result: an `m × 1` vector with the predicted price for each house.

Computational benefit
- One line of code instead of loops (cleaner, less error‑prone)
- Vectorized operations are optimized in libraries (faster)

### Key takeaways (Lecture 3)
- `A` (`m × n`) times `x` (`n × 1`) yields `y` (`m × 1`). Inner dimensions must match.
- Each output entry is a dot product of one row of `A` with `x`.
- Stacking inputs into a design matrix lets you compute many predictions at once: `ŷ = X θ`.


---

## Lecture 4: Matrix-Matrix Multiplication

### What is matrix-matrix multiplication?
Given `A` and `B`, their product `C = A B` is a new matrix built column‑by‑column: each column of `C` is `A` times the corresponding column of `B` (a standard matrix‑vector multiply).

### Dimension rule and output shape
- If `A` is `m × n` and `B` is `n × p`, then `C = A B` is `m × p`.
- Inner dimensions must match (`n` with `n`). The outer dimensions (`m` and `p`) become the shape of the result.

### How to compute C = A B
Column view (most intuitive for beginners):
- The i‑th column of `C` is: `C_{:,i} = A × B_{:,i}`, i.e., `A` multiplied with the i‑th column of `B`.

Entry‑wise (dot‑product) view:

$$
C_{ij} \,=\, \sum_{k=1}^{n} A_{ik} \, B_{kj}
$$

Handy template (3×2)·(2×2):

$$
\begin{bmatrix}
a & b \\
c & d \\
e & f
\end{bmatrix}
\ X \
\begin{bmatrix}
w & x \\
y & z
\end{bmatrix}
= \
\begin{bmatrix}
aw + by & ax + bz \\
cw + dy & cx + dz \\
ew + fy & ex + fz
\end{bmatrix}
$$

### Worked example 1 (2×3)·(3×2)
Let

```
A = ⎡ 1  3  2 ⎤
    ⎣ 4  0  1 ⎦   (2 × 3),

B = ⎡ 1  2 ⎤
    ⎢ 3  0 ⎥      (3 × 2)
    ⎣ 2  5 ⎦
```

Compute columns of `C = A B`:



Step‑by‑step (column view):

```
Take A and the first column of B (b₁):

A b₁ = ⎡ 1  3  2 ⎤  ⎡ 1 ⎤   = ⎡ 1·1 + 3·3 + 2·2 ⎤   = ⎡ 14 ⎤
       ⎣ 4  0  1 ⎦  ⎥ 3 ⎥     ⎣ 4·1 + 0·3 + 1·2 ⎦     ⎣  6 ⎦
                    ⎣ 2 ⎦
Take A and the second column of B (b₂):

A b₂ = ⎡ 1  3  2 ⎤ ⎡ 2 ⎤   = ⎡ 1·2 + 3·0 + 2·5 ⎤   = ⎡ 12 ⎤
       ⎣ 4  0  1 ⎦ ⎥ 0 ⎥     ⎣ 4·2 + 0·0 + 1·5 ⎦     ⎣ 13 ⎦
                   ⎣ 5 ⎦

Assemble C by placing the results as columns:

C = [ A b₁  A b₂ ] = ⎡ 14  12 ⎤
                     ⎣  6  13 ⎦
```


So

```
C = ⎡ 14  12 ⎤
    ⎣  6  13 ⎦   (2 × 2)
```

### Worked example 2 (2×2)·(2×2)
Let

```
A = ⎡ 1  3 ⎤
    ⎣ 2  5 ⎦,   B = ⎡ 0  1 ⎤
                   ⎣ 3  2 ⎦
```

Then

```
AB = ⎡ 1·0 + 3·3    1·1 + 3·2 ⎤
     ⎣ 2·0 + 5·3    2·1 + 5·2 ⎦
   = ⎡ 9   7 ⎤
     ⎣ 15  12⎦
```
### Application to Linear Regression

Goal: get predictions for many houses using several linear hypotheses at once.

- Construct the design matrix `X` with a bias column of 1s and a size column.
- Build the parameter matrix `Θ` where each column holds one hypothesis’ parameters `(θ₀, θ₁)`.
- Multiply to get all predictions at once:

```
Ŷ = X Θ   →  matrix of predictions (m × k)
```

- Columns of `Ŷ`: predictions from each hypothesis for all houses (predictions from one hypothesis.)
- Rows of `Ŷ`: predictions for one house across all hypotheses (predictions for one house.)

Example: 4 houses × 3 hypotheses = 12 predictions from one matrix multiply.

### Apply many hypotheses at once (Ŷ = X Θ)
Stack `m` inputs in `X` (`m × d`) with a leading column of ones; stack `k` hypotheses as columns of `Θ` (`d × k`). Then one matrix multiply gives all `m × k` predictions:

```
Ŷ = X Θ   (m × k)
```

Example (4 houses, 3 hypotheses):

Hypotheses
- h₁(x) = −40 + 0.25·x
- h₂(x) = 200 + 0.10·x
- h₃(x) = −150 + 0.40·x

House Sizes
- 2104
- 1416
- 1534
- 853

```
X = ⎡ 1  2104 ⎤
    ⎢ 1  1416 ⎥
    ⎢ 1  1534 ⎥        Θ = ⎡  −40   200  −150 ⎤
    ⎣ 1   852 ⎦            ⎣ 0.25  0.10  0.40 ⎦

Ŷ = X Θ =
⎡ 486   410   692 ⎤
⎢ 314   342   416 ⎥
⎢ 344   353   464 ⎥
⎣ 173   285   191 ⎦   (rounded to nearest integer)
```

This shows how a single `X Θ` computes many predictions efficiently (vectorization).



### Efficiency

Programming languages provide highly optimized libraries for matrix–matrix multiply (BLAS/LAPACK, NumPy, etc.). These exploit parallel hardware:
- Multi‑core CPUs and SIMD vector instructions
- GPUs and other accelerators

Result: large batches of predictions (many houses × many hypotheses) run fast and with simple code.

### Key takeaways (Lecture 4)
- Multiply matrices only when inner dimensions match: `(m × n)·(n × p) = (m × p)`.
- Column view: each column of `C` is `A` times the corresponding column of `B`.
- Entry view: `C_{ij} = Σ_k A_{ik} B_{kj}`.
- Vectorization: compute many predictions at once with `Ŷ = X Θ`.


---

## Lecture 5: Matrix Multiplication Properties

Matrix multiplication is powerful, but it doesn’t behave like ordinary number multiplication in all ways. Here are the key properties you should know.

### Matrix Multiplication Is Not Commutative: AB ≠ BA
- For real numbers, 3·5 = 5·3. For matrices, the order usually matters.
- In general, `A B ≠ B A`.
- Shapes can even prevent reversing: if `A` is `m×n` and `B` is `n×p`, then `A B` is `m×p`; but `B A` would be `n×n` only if `p = m`.

Dimension example (both defined but different shapes):

```
A (2×3),  B (3×2)
AB → (2×2)   but   BA → (3×3)
```

Since `AB` and `BA` do not even have the same dimensions, they cannot be equal.

Example (2×2):

```
A = ⎡1 1⎤,  B = ⎡0 0⎤
    ⎣0 0⎦       ⎣2 0⎦

AB = ⎡1·0 + 1·2   1·0 + 1·0⎤ = ⎡2  0⎤
     ⎣0·0 + 0·2   0·0 + 0·0⎦   ⎣0  0⎦

BA = ⎡0·1 + 0·0   0·1 + 0·0⎤ = ⎡0  0⎤
     ⎣2·1 + 0·0   2·1 + 0·0⎦   ⎣2  2⎦
```

Clearly, `AB ≠ BA`.

### Matrix Multiplication Is Associative: A(BC) = (AB)C
- Like real numbers, matrix multiplication is associative.
- You can group products without changing the result, provided the shapes allow both sides.

Number analogy:
```
3 × (5 × 2) = (3 × 5) × 2 = 30
```

Matrix example (shapes chosen so both groupings are valid):

```
A (2×3),  B (3×4),  C (4×2)

(AB) has shape (2×4), so (AB)C is (2×2)
(BC) has shape (3×2), so A(BC) is (2×2)

Compute either order and you get the same (2×2) result.
```

Operational view:
```
Let D = (B C), then A(BC) = A D
Let E = (A B), then (AB)C = E C
```

Both yield the same matrix.

### Identity matrix I
- Numbers: 1 is the multiplicative identity (`1 × z = z × 1 = z`).
- Matrices: the `n×n` identity `Iₙ` has 1s on the diagonal and 0s elsewhere, and acts like 1. For any `A (m×n)`:

```
Iₘ A = A    and    A Iₙ = A
```

Shapes must match the sides you multiply on (left uses `Iₘ`, right uses `Iₙ`).

Examples:

```
I₂ = ⎡1 0⎤,  I₃ = ⎡1 0 0⎤
     ⎣0 1⎦        ⎣0 1 0⎦
                 ⎣0 0 1⎦
```

Note: While `AB ≠ BA` in general, `A I = I A = A` does hold.

### Dimension reminders
- `A (m×n)` times `B (n×p)` is defined and yields `m×p`.
- Reversing order changes both validity and shape; don’t assume commutativity.
- When identities appear, ensure the `I` on each side has the correct size (`Iₘ` or `Iₙ`).

### Quick examples
- Non‑commutative shapes: if `A (2×3)` and `B (3×4)`, then `AB (2×4)` is valid; `BA` is not defined.
- Associativity in practice: compute `X(Θ W)` or `(X Θ)W` depending on which is cheaper; both are equal when defined.

### Key takeaways (Lecture 5)
- Matrix multiplication is generally not commutative (`AB ≠ BA`).
- It is associative: `A(BC) = (AB)C` when shapes allow.
- The identity `I` satisfies `I A = A I = A` with appropriate sizes.
- Always check dimensions before multiplying; order matters.

