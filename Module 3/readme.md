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
