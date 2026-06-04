# GGML Quantization — Demystified

> **A deep-dive into how GGML actually compresses LLM weights — from the dead-simple Q4_0 to the surprisingly clever Q4_K.**

I spent time reading the actual GGML source code, tracing every line, and writing implementations to understand what's really going on under the hood. This repo is the result — my notes, my understanding, and working C code that mirrors the real GGML quantization pipeline.

This is not a surface-level "quantization reduces model size" explainer. This goes into **how** the numbers get crushed, **why** certain tricks exist, and **where** the cleverness actually lives.

---

## The Philosophy

The goal of quantization is **not** to find the best scale with different methods. The real goal is to **reduce round-off error** — that's the thread that runs through the entire quantization field. Every technique here is a different answer to the same question: *how do we lose less information when we shrink these numbers?*

### Sources

Not "trust me bro" — here are the actual files:

| What | Link |
|------|------|
| Block structures | [`ggml-common.h`](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h) |
| Core quant math | [`ggml-quants.c`](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c) |

The game maker / game changer: [**Georgi Gerganov**](https://github.com/ggerganov) 🐐

---

##  The Journey: Q4_0 → Q4_1 → Q4_K

```
Q4_0                    Q4_1                    Q4_K
 │                       │                       │
 │  1 scale              │  1 scale + 1 min      │  Many local scales/mins
 │  No min               │  Fits data range      │  Search for best Laux
 │  Assumes centered     │  Handles skew         │  Quantize scales too
 │  Simple but lossy     │  Better but limited   │  Complex but precise
 │                       │                       │
 ▼                       ▼                       ▼
 "Just get it done"      "Get it right"          "Get it *really* right"
```

---

##  What's In This Repo

| File | Description |
|------|-------------|
| [`Q40.c`](Q40.c) | Q4_0 quantization & dequantization — adapted from GGML source |
| [`Q41.c`](Q41.c) | Q4_1 block structure definition |
| [`Q4K.c`](Q4K.c) | Q4_K full pipeline — the beast, including the `make_qkx2_quants` optimizer |
| [`README.md`](README.md) | You're reading it |

---

##  Q4_0 — The Simplest One

**Idea:** Take 32 values, find one scale, squeeze everything into 4 bits. That's it.

- One block = one shared scale, no min value
- Assumes data is roughly centered around zero
- Quantized values land in the range **-8 … +7**

### How It Works

**1. Find the biggest magnitude**
```
max = largest |x| in the block
```

**2. Create the scale**
```
d = max / -8
```

**3. Normalize each value**
```
x_scaled = x / d
```

**4. Shift into 4-bit range and round**
```
q = round(x_scaled + 8)
```

**5. Clamp to valid range**
```
0 ≤ q ≤ 15
```

**6. Pack two values per byte** — two 4-bit values squeezed into one `uint8_t`.

### Dequantization

```
x ≈ d × (q - 8)
```

###  The Weakness

Imagine your data is all positive: `[10, 13, 15, 17]`. With Q4_0, the formula maps everything relative to the center, so the entire `[-8, 0]` side of the range sits there **completely unused**. You're wasting half your precision on values that don't exist. Skewed data = wasted bits.

---

##  Q4_1 — Adding a Floor

**Idea:** Same 4-bit weights, but now we also store a **min value**. Instead of assuming the data is centered, we fit the actual range.

### How It Works

**1. Find the real min and max**

**2. Create the scale from the actual range**
```
scale = (max - min) / 15
```

**3. Quantize relative to min**
```
q = round((x - min) / scale)
```

**4. Clamp**
```
0 ≤ q ≤ 15
```

**5. Store:** scale + min + 4-bit weights

### Dequantization

```
x ≈ q × scale + min
```

### Why It's Better

- Uses the **full** 0…15 range — no wasted levels
- Handles positively or negatively skewed data gracefully
- Small cost: just one extra `float16` (the min) per block

---

## 🔴 Q4_K — Where Things Get Scary (And Brilliant)

Q4_1 is already decent. Q4_K asks: **"Can we get better quality without increasing the weight bits?"**

The answer: **use many local scales and mins** — and then be *very* clever about finding them.

This is where GGML goes from "textbook quantization" to "hand-optimized black magic." Let's walk through it.

---

### The Superblock Architecture

Instead of one block → one scale → one min, Q4_K introduces a hierarchy:

```
┌─────────────────────────────────────────────┐
│               SUPERBLOCK (256 values)       │
│                                             │
│  ┌──────────┐ ┌──────────┐     ┌──────────┐│
│  │ subblock  │ │ subblock  │ ... │ subblock  ││
│  │ 32 values │ │ 32 values │     │ 32 values ││
│  │ scale₀    │ │ scale₁    │     │ scale₇    ││
│  │ min₀      │ │ min₁      │     │ min₇      ││
│  └──────────┘ └──────────┘     └──────────┘│
│                                             │
│  + d     (FP16 super-scale for scales)      │
│  + dmin  (FP16 super-scale for mins)        │
└─────────────────────────────────────────────┘
```

Each 32-value chunk gets its **own** scale and min — so different regions of the tensor can be quantized independently. This is the key insight.

---

### Step 1 — Find Best Scale/Min for Every 32 Values

For each 32-value subblock, GGML doesn't just do a naive `(max - min) / 15`. It actively **searches** for the best scale and min that minimize reconstruction error. This is where `make_qkx2_quants` enters the picture.

---

### Step 2 — First Quantization Guess (Laux)

Using a candidate scale, create an initial quantization:

```
q = round((x - min) / scale)    →    values in 0…15
```

This gives us `Laux` — a candidate set of integer levels. Think of it as: *"if the scale were this value, here's what the quantized weights would look like."*

---

### Step 3 — Fit the Best Line (Weighted Least Squares)

Here's the clever part. Given a specific `Laux`, GGML asks:

> *"Okay, assuming these are my integer levels, what's the optimal scale and min to reconstruct the original values?"*

We want to fit a line through the data:

```
x ≈ scale × L + min
```

This is just `y = mx + c` from school — but here `L` is the integer level (0–15) and `x` is the original float value. We need the best `m` (scale) and `c` (min).

**The problem in plain English:** You have 32 pairs — each pair is (an integer level `L`, an original value `x`). You need to find the line that best connects them. That's it. That's all the regression is doing.

To solve this, the code accumulates five running sums in one pass:

```c
sum_w  += w;          // total weight
sum_l  += w * l;      // weighted sum of integer levels
sum_l2 += w * l * l;  // weighted sum of squared levels
sum_xl += w * l * x;  // weighted cross-product (how levels relate to real values)
// sum_x was already computed earlier — weighted sum of original values
```

Now here's what each piece does and **why it's there**:

**`D = sum_w × sum_l2 - sum_l × sum_l`**

This measures how *spread out* the integer levels are. If every value quantized to the same level (say everything became `7`), then `D = 0` — the levels have zero spread, and there's no line to fit. `D` being zero is GGML's way of saying "these are all the same number, skip it." When `D` is large, the levels are nicely spread across 0–15, which means we have a solid grid to work with.

**`this_scale = (sum_w × sum_xl - sum_x × sum_l) / D`**

Think about what `sum_xl` captures — it's the correlation between integer levels and original values. If high levels correspond to high original values, `sum_xl` is large. The `sum_x × sum_l` part subtracts out the "would-have-happened-anyway" baseline (the part that's just because both averages are nonzero). What's left is the *actual relationship* between levels and values. Dividing by `D` normalizes it by how spread the levels are. The result: how much the original value changes per integer step. That's your scale.

**`this_min = (sum_l2 × sum_x - sum_l × sum_xl) / D`**

Once you know the scale (how steep the line is), this finds where the line starts — the y-intercept. It strips out the part that's already explained by the scale, and what remains is the offset. Intuitively: after you've accounted for "bigger level = bigger value," this is the leftover shift needed to line up the grid with the actual data.

> The beauty of this: it all happens in **one pass** over 32 values. No iteration, no gradient descent. Just five running sums → two divisions → done. That's why GGML can do this for every subblock without being slow.

---

### Step 4 — Compute Weighted Error

Reconstruct the values and measure how far off we are:

```
x_recovered = this_scale × L + this_min
diff = x_recovered - x_original
error += weight × diff²
```

**About those weights** — you might wonder: *"why not use gradient descent or backpropagation?"*

Relax. The weights here are dead simple and that's by design:

```c
float av_x = sqrtf(sum_x2 / 32);         // RMS of the block — overall "impact"
weights[l] = av_x + fabsf(x[32*j + l]);   // bigger values → bigger weight
```

The idea: if a value of `1` gets reconstructed as `0.7`, that's ~30% error but who cares — it's tiny. But if a value of `20` drops to `14`, that same 30% error is devastating. **Larger values matter more**, so they get more weight. No fancy optimization needed — the signal is in the magnitudes.

---

### Step 5 — Try Many Laux (The Big Idea)

This is the genius move. GGML does **not** search for the best scale directly. It searches for **different Laux** — different candidate integer level assignments.

Why? Because different rounding choices = different information loss patterns. A scale of `5.0` might round a value up while `4.9` rounds it down, and those different rounding decisions cascade through the entire reconstruction.

```
For each candidate scale (e.g., 4.8, 4.9, 5.0, 5.1, 5.2):

    Generate Laux  →  Find best scale/min via regression  →  Measure error
                                                                  │
                                                            Keep the winner
```

The subtle brilliance: `Laux` is **not** the final answer. It's just a scaffold — a way to assign integers so the regression can find the best-fit line. The real scale and min come from the regression, not from the candidate that generated `Laux`.

> Think of it like this: you're trying different rulers to mark positions, then asking *"given these marks, what's the best ruler that could have made them?"* — and keeping whichever ruler reconstructs the original positions most faithfully.

---

### Step 6 — Collect the Winners

After all that searching, we now have, for every 32-value subblock:

```
scale[0], min[0]
scale[1], min[1]
...
scale[7], min[7]
```

These are still float values. The hard work of finding them is done. But we can't store them all as floats — that would eat our memory savings.

---

### Step 7 — Quantize the Scales and Mins Themselves

GGML now says: *"Too many float scales/mins — let's quantize those too."*

Both scales and mins get crushed to **6 bits** (range 0…63):

```
scale_q = round(scale × inv_scale)    →    0…63
min_q   = round(min × inv_min)        →    0…63
```

---

### Step 8 — Store Super Scales (Scale of Scales)

To reconstruct the quantized scales/mins later, we need a "scale for the scales":

```
d    = max_scale / 63     ← stored as FP16
dmin = max_min / 63       ← stored as FP16
```

So at dequant time:
```
real_scale = d × scale_q
real_min   = dmin × min_q
```

It's quantization all the way down. 🐢

---

### Step 9 — Final Weight Quantization

*Now* the actual weights get quantized. Everything before this was about finding the best parameters. This is where the rubber meets the road:

```
q = round((x + min) / scale)
clamp to 0…15
```

These become the final stored 4-bit weights.

---

### Step 10 — Bit Packing

Everything gets packed tight:

| Component | Bits | Storage |
|-----------|------|---------|
| Weights | 4-bit | Two per byte (nibble packing) |
| Scales | 6-bit | Packed across bytes |
| Mins | 6-bit | Packed across bytes |
| Super-scales | FP16 | `d` and `dmin` |

---

##  Dequantization (Unpacking)

To get values back:

**1. Recover the scale and min for each subblock:**
```
scale = d × scale_q
min   = dmin × min_q
```

**2. Recover the weight:**
```
x ≈ scale × q - min
```

That's it. The compression is lossy, but the entire Q4_K pipeline exists to make that loss as small as possible.

---

##  Why This Matters at Inference, Not Just Storage

The common assumption: quantization helps storage, but you pay a dequantization cost at runtime. GGML's design makes that cost **nearly free**.

During matrix multiplication (`mul_mat`), GGML **never** materializes a full `float32` weight matrix. Instead, it processes weights in small, fixed-size windows — typically 4–8 blocks at a time, sized to fill SIMD registers (AVX2 on x86 is 256 bits = 8 `float32`s at once). Each window gets dequantized, multiplied against the input chunk, accumulated into the output, and then thrown away.

```c
for each window of blocks:
    dequant(qs, d)           → ~128–256 floats, lives in registers
    accumulate into output   → dot product
    discard, move to next window
```

**Peak live `float32` weight data at any moment: a few KB, not GB.**

This works because of the block structure. Each block is self-contained — it carries its own scale(s) and can be dequantized independently, with no knowledge of the rest of the matrix. A global scale per layer would force you to load the whole layer first.

So the block design buys you two things from one decision:

| Benefit | How |
|---------|-----|
| **Better Accuracy** | Local scales isolate outliers |
| **Streaming Matmul** | Self-contained blocks = no full dequant needed |

---

##  The Takeaway

| Format | Scale | Min | Trick | Quality |
|--------|-------|-----|-------|---------|
| **Q4_0** | 1 per block | ❌ None | — | Baseline |
| **Q4_1** | 1 per block | ✅ 1 per block | Fits actual range | Better |
| **Q4_K** | 1 per 32 values | ✅ 1 per 32 values | Laux search + regression + quantized scales | Much better |

The progression tells a clear story:

```
Same 4-bit weights everywhere.

Q4_0:  Blunt instrument. One scale, hope for the best.
Q4_1:  Add a floor. Now we fit the actual data range.
Q4_K:  Give every neighborhood its own ruler and floor,
       search hard for the best ones,
       then compress the rulers too.
```

**The main goal never changed:** keep the same 4-bit weights, but **lose less information**.

---

##  Block Structures (From Source)

<details>
<summary><b>Q4_0 Block</b></summary>

```c
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta (scale)
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
```
**Size:** 2 bytes (scale) + 16 bytes (weights) = **18 bytes for 32 values**
</details>

<details>
<summary><b>Q4_1 Block</b></summary>

```c
#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta (scale)
            ggml_half m; // min
        };
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
```
**Size:** 4 bytes (scale + min) + 16 bytes (weights) = **20 bytes for 32 values**
</details>

<details>
<summary><b>Q4_K Block (Superblock)</b></summary>

```c
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];          // 4-bit quants
} block_q4_K;
```
**Size:** 4 bytes (super-scales) + 12 bytes (quantized scales/mins) + 128 bytes (weights) = **144 bytes for 256 values**
</details>

---

## Credits

- **[Georgi Gerganov](https://github.com/ggerganov)** — Creator of GGML / llama.cpp. The code in this repo is adapted from and references his work.
- **[GGML Source](https://github.com/ggml-org/ggml)** — The actual implementation that powers quantized inference everywhere.

---

<p align="center">
<i>Built by reading source code, not blog posts.</i>
<br>
<i>If you find something here that doesn't match what actually happens — I'd love to hear about it.</i>
</p>

<p align="center">
<a href="https://github.com/Mahendra1706/ggml-quantization">
<img src="https://img.shields.io/badge/GitHub-Mahendra1706-181717?style=for-the-badge&logo=github" alt="GitHub">
</a>
</p>

