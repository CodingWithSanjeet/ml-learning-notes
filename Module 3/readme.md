## Module 3: Linear Algebra Review

Welcome! In this module we quickly review the linear‑algebra building blocks you’ll use throughout machine learning. We’ll keep the language simple and add small, concrete examples you can refer back to at any time.

## 📚 Table of Contents
- [Lecture 1: Matrices and Vectors](#lecture-1-matrices-and-vectors)
  - [What is a matrix?](#what-is-a-matrix)
  - [Matrix dimensions](#matrix-dimensions)
  - [Referring to entries: Aᵢⱼ](#referring-to-entries-aᵢⱼ)
  - [What is a vector?](#what-is-a-vector)
  - [Vector dimension and Rⁿ](#vector-dimension-and-rⁿ)
  - [Vector examples](#vector-examples)
  - [Vector indexing: yᵢ (and 1‑ vs 0‑indexing)](#vector-indexing-y-and-1--vs-0--indexing)
  - [Common notation conventions](#common-notation-conventions)
  - [Quick reference](#quick-reference)

---

## Lecture 1: Matrices and Vectors

### What is a matrix ?
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

### What is a vector?
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

### Vector examples
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


