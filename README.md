# Gazelle Optimization Algorithm (GOA) - CEC2017 Research

This repository contains the source code, benchmark data, and analysis tools for researching improvements to the **Gazelle Optimization Algorithm (GOA)** using Chaotic Mapping.

## Repository Structure

```text
GOA-PAPER/
├── src/                  # C Source files
│   ├── goa.c             # Original GOA (Baseline)
│   ├── cgoa1.c           # Variant 1: Chaotic Updates
│   ├── cgoa2.c           # Variant 2: Chaotic Initialization
│   └── cec17_test_func.c # Official CEC2017 Benchmark Suite
├── input_data/           # REQUIRED: Data files for CEC17 functions
├── results/              # Generated .csv data files
├── plots/                # Generated comparison .png charts
├── scripts/              # Python visualization tools
│   └── plot_results.py
├── docs/                 # Original Research Paper (PDF)
└── .gitignore            # Excludes binaries and venv
```

---

## Getting Started

### Prerequisites

Ensure your system has a C compiler and Python 3 environment ready:

- **Linux/Ubuntu:** `sudo apt install build-essential python3-pip`
- **Python Libraries:** `pip install pandas matplotlib numpy`

### Setup

Clone the repository to your local machine.

> **CRITICAL:** The CEC2017 benchmark requires the `input_data` folder to be present in the root directory. If this folder is missing or moved, the C programs will encounter a **Segmentation Fault** during execution.

---

## Running the Experiments

All benchmarks must be compiled with the math library flag (`-lm`) and the benchmark suite file (`cec17_test_func.c`).

### Step 1: Establish the Baseline (Original GOA)

Generate baseline results first — the comparison script depends on this output.

```bash
cd src
gcc goa.c cec17_test_func.c -o goa_baseline -lm
./goa_baseline
```

Output: creates `../results/goa_cec17_results.csv`.

### Step 2: Test the Chaotic Variants

**Variant 1 — Chaotic Updates (CGOA1)**

```bash
gcc cgoa1.c cec17_test_func.c -o cgoa1_bench -lm
./cgoa1_bench
```

**Variant 2 — Chaotic Initialization (CGOA2)**

```bash
gcc cgoa2.c cec17_test_func.c -o cgoa2_bench -lm
./cgoa2_bench
```

---

## Visualization & Comparison

The `plot_results.py` script performs comparative analysis. It automatically loads `goa_cec17_results.csv` as the baseline and plots the variant side-by-side on a logarithmic scale.

Navigate to the `scripts` folder and provide the variant filename and desired output image name.

**Compare CGOA1 vs Baseline:**

```bash
cd ../scripts
python3 plot_results.py cgoa1_cec17_results.csv cgoa1_vs_goa.png
```

**Compare CGOA2 vs Baseline:**

```bash
python3 plot_results.py cgoa2_cec17_results.csv cgoa2_vs_goa.png
```

---

## Key Research Findings

- **CGOA1 (Chaotic Updates):** Replaces the linear pseudo-random parameter *R* with a non-linear **Logistic Map**. Experimental evidence shows this significantly reduces error rates in high-dimensional, deceptive landscapes (F12 and F30) by maintaining population diversity.
- **CGOA2 (Chaotic Initialization):** Uses the Logistic Map to ensure a perfectly uniform distribution of the gazelle herd at Iteration 0.
- **F2 Notice:** In accordance with official IEEE CEC2017 guidelines, Function F2 is excluded from all benchmarks due to mathematical instability in the original source code.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Segmentation Fault | Ensure `input_data/` is present in the root directory |
| Path errors | Run C binaries from `/src` and Python scripts from `/scripts` |
| Permission denied | Run `chmod +x <binary_name>` on Linux |
| Git conflicts | Always `git pull` before a new benchmark run to sync `results/` |
