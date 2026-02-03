# 🔄 Semantic Shift

<p align="center">
  <b>A framework for detecting semantic drift in text streams using embedding-based methods</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/pytorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/transformers-🤗-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

## 📁 Project Structure

```
semantic_shift/
├── 📂 data/
│   ├── 📂 raw/                     # Raw input data
│   │   └── 📂 IMDB_reviews/        # IMDb reviews dataset
│   └── 📂 processed/               # Processed data files
│       ├── imdb_clean.pkl          # Cleaned reviews
│       ├── reference_set.pkl       # Reference period data
│       ├── stream_set.pkl          # Stream period data
│       ├── ref_embeddings.pt       # Reference embeddings
│       └── stream_embeddings.pt    # Stream embeddings
├── 📂 src/                         # Source code
│   ├── preprocess.py               # Data cleaning
│   ├── make_splits.py              # Temporal splitting
│   ├── extract_features.py         # Embedding extraction
│   ├── detector.py                 # Drift detection
│   ├── mmd.py                      # MMD implementation
│   └── data_loader.py              # Data utilities
├── 📂 experiments/                 # Experiment scripts
├── 📂 notebooks/                   # Jupyter notebooks
└── requirements.txt                # Dependencies
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/semantic_shift.git
cd semantic_shift

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.12+
- PyTorch 2.0+
- Transformers (Hugging Face)
- pandas, numpy, tqdm

---

## 📊 Data Pipeline

The pipeline transforms raw IMDb reviews into embedding representations for drift detection.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Reviews   │───▶│  Preprocessed   │───▶│   Embeddings    │
│    (JSON)       │    │    (Pickle)     │    │   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     573,913              386,833              50K + 165K
```

### 1️⃣ Preprocessing (`src/preprocess.py`)

Cleans the raw IMDb reviews data and prepares it for analysis.

| Parameter | Value |
|-----------|-------|
| **Input** | `data/raw/IMDB_reviews/IMDB_reviews.json` |
| **Output** | `data/processed/imdb_clean.pkl` |
| **Records** | 386,833 reviews |

**Processing steps:**
- ✅ Extracts fields: `review_date`, `review_summary`, `review_text`, `rating`, `movie_id`
- ✅ Creates `full_text` = `review_summary` + `review_text`
- ✅ Converts `review_date` to datetime
- ✅ Filters to years 2005-2023
- ✅ Drops missing ratings
- ✅ Creates binary labels: **Positive** (>6) | **Negative** (<5)

```bash
python src/preprocess.py
```

---

### 2️⃣ Temporal Splitting (`src/make_splits.py`)

Divides data into reference and stream sets with an intentional temporal gap.

```
        Reference              Gap              Stream
    ┌───────────────┐    ┌───────────┐    ┌───────────────┐
    │   2005-2010   │    │ 2011-2012 │    │   2013-2018   │
    │   (50,000)    │    │ (excluded)│    │  (165,790)    │
    └───────────────┘    └───────────┘    └───────────────┘
         "Gold Standard"                    "Deployment Era"
```

| Output File | Samples | Year Range | Purpose |
|-------------|---------|------------|---------|
| `reference_set.pkl` | 50,000 | 2005-2010 | Training baseline |
| `stream_set.pkl` | 165,790 | 2013-2018 | Drift testing |

```bash
python src/make_splits.py
```

---

### 3️⃣ Feature Extraction (`src/extract_features.py`)

Generates sentence embeddings using DistilBERT's `[CLS]` token.

| Parameter | Value |
|-----------|-------|
| **Model** | `distilbert-base-uncased` |
| **Embedding Dim** | 768 |
| **Max Length** | 128 tokens |
| **Batch Size** | 32 |

**Device Support:**
- 🟢 CUDA (NVIDIA GPU)
- 🟢 MPS (Apple Silicon)
- 🟢 CPU (Fallback)

| Output File | Shape | Contents |
|-------------|-------|----------|
| `ref_embeddings.pt` | (50,000 × 768) | Reference embeddings |
| `stream_embeddings.pt` | (165,790 × 768) | Stream embeddings |

**Saved structure:**
```python
{
    'embeddings': torch.Tensor,  # (N, 768)
    'labels': torch.Tensor,      # (N,)
    'dates': np.array            # datetime64
}
```

```bash
python src/extract_features.py
```

---

## 🎯 Scientific Background: The Marginal Stability Paradox

### The Core Hypothesis

This research investigates a critical blind spot in drift detection: **the Marginal Stability Paradox**.

> **Hypothesis:** The geometric shape of data $P(E)$ can remain perfectly stable (Global Detector sees nothing), while the semantic meaning $P(Y|E)$ drifts silently (Conditional Detector detects the change).

In production ML systems, this paradox manifests when:
- Input features look statistically identical over time
- But the relationship between features and labels has fundamentally changed
- Standard drift detectors remain "asleep" while model performance degrades

### Two Types of Drift

| Drift Type | Mathematical Definition | What Changes | Detection Method |
|------------|------------------------|--------------|------------------|
| **Covariate Drift** | $P(E)_{t} \neq P(E)_{t_0}$ | Feature distribution | Global MMD |
| **Concept Drift** | $P(Y|E)_{t} \neq P(Y|E)_{t_0}$ | Feature-label relationship | Conditional MMD |

### The Experimental Challenge

To prove this paradox exists, we need an experiment where:
- ✅ **Global Drift = 0** (geometry frozen)
- ✅ **Conditional Drift > 0** (semantics changed)

This is surprisingly difficult to construct because most transformations that change meaning also change geometry.

### Experiment Design Analysis

#### ❌ Experiment A: Rotation (The Failure)

**Design:** Rotate "Positive" class embeddings toward "Negative" center.

```
Before:                          After Rotation:
    ● ● ●                            ● ● ●
  ● ● ● ● ●    ──rotate──▶         ● ● ● ● ●  ←── Cluster MOVED
    ● ● ●                              ↘ ↘ ↘
```

**Why it failed:**
- Moving points changed the **location** in 768-dimensional space
- Global distribution $P(E)$ was altered
- **Result:** Both detectors fired (Global > 0, Conditional > 0)
- **Verdict:** Cannot prove the paradox—just showed both detectors agree

#### ✅ Experiment B: Concept Swap (The Success)

**Design:** Clone embeddings exactly, then swap labels randomly.

```
Reference:                       Stream (Labels Swapped):
┌─────────────────┐              ┌─────────────────┐
│ Positive: ● ● ● │              │ Positive: ● ● ○ │  ← Same vectors,
│           ● ●   │    clone     │           ● ○   │    corrupted labels
├─────────────────┤   ======▶    ├─────────────────┤
│ Negative: ○ ○ ○ │              │ Negative: ○ ● ○ │  ← "Stolen" from
│           ○ ○   │              │           ○ ●   │    wrong class
└─────────────────┘              └─────────────────┘

     Embeddings: IDENTICAL (bit-for-bit clone)
     Labels: CORRUPTED (50% swapped)
```

**Why it worked:**

| Detector | What It Sees | Result |
|----------|--------------|--------|
| **Global** | Pile of vectors (no labels) → identical pile | MMD = 0.000000 ✅ |
| **Conditional** | "Positive" class now mixed with negative vectors | MMD = 0.011643 ✅ |

**Verdict:** Successfully created a scenario where **Geometry was Frozen but Semantics Melted**—the exact definition of the paradox.

### Comparison Summary

| Feature | Experiment A (Rotation) | Experiment B (Swap) |
|---------|------------------------|---------------------|
| **Action** | Moved the points | Swapped the labels |
| **Global Geometry $P(E)$** | Changed ❌ | Identical ✅ |
| **Semantic Mapping $P(Y|E)$** | Changed | Changed |
| **Global Detector Result** | Detected Drift | **Saw Nothing (0.000)** |
| **Conditional Detector Result** | Detected Drift | **Detected Drift (0.011)** |
| **Scientific Conclusion** | "Data moved" (Ambiguous) | "Only Meaning changed" (Proof) |

### Mathematical Proof

In Experiment B, since $X_{stream} = X_{ref}$ (exact clone):

$$P(E_{stream}) = P(E_{ref}) \Rightarrow MMD_{global} = 0$$

But since labels are corrupted:

$$P(E|Y=1)_{stream} \neq P(E|Y=1)_{ref}$$

The "Positive" class in the stream now contains vectors that were originally negative, causing class-conditional distributions to diverge.

### Research Contribution

This work validates a detector that identifies **concept drift** (relationship between features and labels changes) even when **covariate drift** (feature distribution) is completely absent—**a critical blind spot in production ML systems**.

---

## 🧪 Experiments

### Experiment 1: IMDb Stream Drift Detection (`experiments/02_imdb_stream.py`)

Analyzes semantic drift in real IMDb reviews using sliding window MMD comparison.

```
Reference (2005-2010)  ──────────────────────────────────────────────▶
                              ┌────────┐ ┌────────┐ ┌────────┐
Stream (2013-2018)     ──────│Window 1│─│Window 2│─│Window 3│─ ... ──▶
                              └────────┘ └────────┘ └────────┘
                                  ↓          ↓          ↓
                              MMD scores over time
```

| Parameter | Value |
|-----------|-------|
| **Window Size** | 1,000 reviews |
| **Step Size** | 500 (overlapping) |
| **Total Windows** | 330 |
| **Reference Samples** | 500 (subsampled) |

**Metrics computed:**
- **Global MMD** - Compares overall embedding distributions
- **Conditional MMD** - Compares class-specific distributions (Positive/Negative)

**Results:**
| Metric | Mean | Std |
|--------|------|-----|
| Global MMD | 0.0927 | 0.0628 |
| Conditional MMD | 0.0984 | 0.0669 |

```bash
python experiments/02_imdb_stream.py
```

**Output:** `results/imdb_drift_results.csv`

---

### Experiment 2: Concept Swap Validation (`experiments/03_concept_swap.py`)

**Goal:** Prove that Conditional MMD detects semantic drift when Global MMD is completely blind.

**Key Insight:** By keeping embeddings identical but swapping labels, we create pure semantic drift where P(E) is stable but P(E|Y) changes.

```
                 Reference                    Stream (with swapped labels)
            ┌─────────────────┐            ┌─────────────────┐
 Positive   │  ● ● ● ● ● ●    │            │  ● ● ○ ● ○ ●    │  ← Some labels flipped
            │      ● ● ●      │     =      │      ● ○ ●      │
            ├─────────────────┤            ├─────────────────┤
 Negative   │  ○ ○ ○ ○ ○      │            │  ○ ● ○ ○ ●      │  ← Some labels flipped
            └─────────────────┘            └─────────────────┘
                 Embeddings are IDENTICAL (P(E) stable)
                 Labels are CORRUPTED (P(E|Y) drifts)
```

**Results:**

| Swap % | Global MMD | Conditional MMD |
|--------|------------|-----------------|
| 0%     | 0.000000   | 0.0054          |
| 10%    | 0.000000   | 0.0057          |
| 20%    | 0.000000   | 0.0076          |
| 30%    | 0.000000   | 0.0079          |
| 40%    | 0.000000   | 0.0097          |
| 50%    | 0.000000   | 0.0116          |

**✅ HYPOTHESIS VALIDATED:**
- Global MMD = 0 (perfect marginal stability)
- Conditional MMD increases +117% as labels are corrupted

```bash
python experiments/03_concept_swap.py
```

**Output:** `results/concept_swap_plot.png`

---

### Experiment 3: Synthetic Injection (`experiments/03_synthetic_injection.py`)

**Goal:** Demonstrate semantic drift detection when positive reviews shift toward negative semantic space.

**Method:** Gradually rotate positive class embeddings toward the negative class center.

```bash
python experiments/03_synthetic_injection.py
```

**Output:** `results/synthetic_drift_plot.png`

---

## 📊 Visualization (`src/plot_results.py`)

Generates longitudinal drift plots from experiment results.

```bash
python src/plot_results.py
```

**Output:** `results/drift_plot.png`

| Line | Color | Style | Meaning |
|------|-------|-------|---------|
| Global MMD | Blue | Dashed | Marginal distribution drift |
| Conditional MMD | Red | Solid | Semantic (class-conditional) drift |

---

## 🔬 Core Algorithm: MMD (`src/mmd.py`)

**Maximum Mean Discrepancy (MMD)** measures the distance between two probability distributions in a reproducing kernel Hilbert space (RKHS).

$$MMD^2(P,Q) = \mathbb{E}[k(x,x')] - 2\mathbb{E}[k(x,y)] + \mathbb{E}[k(y,y')]$$

**Features:**
- Multi-scale Gaussian (RBF) kernel
- Memory-efficient chunked computation for large datasets
- Automatic bandwidth selection

| Function | Use Case |
|----------|----------|
| `mmd_rbf()` | Standard MMD (small datasets) |
| `mmd_rbf_chunked()` | Memory-efficient MMD (large datasets) |

---

## 🚀 Quick Start

```bash
# Step 1: Preprocess raw data
python src/preprocess.py

# Step 2: Create temporal splits
python src/make_splits.py

# Step 3: Extract embeddings
python src/extract_features.py

# Step 4: Run experiments
python experiments/02_imdb_stream.py      # Real data drift detection
python experiments/03_concept_swap.py     # Validation experiment

# Step 5: Visualize results
python src/plot_results.py
```

---

## 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Raw Reviews** | 573,913 |
| **After Cleaning** | 386,833 |
| **Reference Set** | 50,000 |
| **Stream Set** | 165,790 |
| **Temporal Gap** | 2011-2012 |

### Label Distribution

| Set | Positive (1) | Negative (0) | Ratio |
|-----|--------------|--------------|-------|
| Reference | 36,863 | 13,137 | 2.8:1 |
| Stream | 123,138 | 42,652 | 2.9:1 |

---

## 📁 Results

| File | Description |
|------|-------------|
| `results/imdb_drift_results.csv` | Longitudinal MMD scores from stream experiment |
| `results/drift_plot.png` | Visualization of drift over time |
| `results/concept_swap_plot.png` | Validation experiment results |
| `results/synthetic_drift_plot.png` | Synthetic injection results |

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for semantic drift research
</p>