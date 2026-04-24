# The Resource Divide: Data Scarcity, Compute Inequity, and Global LLM Governance

**Course:** FunML Term Project  
**Authors:** Connor Sempf (csempf3@gatech.edu) and Bo Pang (bpang42@gatech.edu)  
**Institution:** Georgia Institute of Technology, School of Electrical and Computer Engineering  
**Date:** April 2025

---

## Project Overview

This project presents a comprehensive quantitative analysis of how AI resources---compute, data, and tokenization---are distributed unevenly across the world's languages. We combine real tokenizer benchmarks, memory simulation, serving-system analysis, and a multi-dimensional language ecosystem framework to demonstrate and quantify the compounding nature of the AI divide.

**Key Findings:**
- Hindi pays a 7.99x attention cost penalty vs English's 1.32x (6x multiplier)
- Consumer GPUs (RTX 4060) OOM at 14K tokens; H100 sustains 544K (40x gap)
- In serving scenarios with 32 concurrent users, Hindi OOMs after 1 turn on RTX 4060
- Decode dominates 95-99.9% of end-to-end latency across all workloads
- Composite Resource Divide Index: Swahili 0.800 vs English 0.099 (8.1x structural disadvantage)

---

## Repository Structure

```
project/
├── README.md                           # This file
├── experiments/
│   ├── exp1_tokenization_benchmark.py  # Real tokenizer fertility analysis
│   ├── exp2_memory_wall_simulation.py  # KV cache growth simulation
│   ├── exp3_combined_analysis.py      # End-to-end latency + RDI
│   ├── exp4_multiturn_serving.py       # Multi-turn conversation simulation
│   ├── exp5_prefill_decode_disaggregation.py  # Phase-specific hardware analysis
│   └── exp6_language_ecosystem.py     # CRDI + multi-metric analysis
├── figures/                            # 13 publication-quality figures (300 DPI)
├── data/                               # Raw JSON results for all experiments
└── report/
    ├── full_paper.md                   # Source markdown
    └── Resource_Divide_Paper.docx    # Final IEEE-format paper
```

---

## Setup Instructions

### Requirements
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install transformers numpy matplotlib scipy
```

**Note:** `transformers` requires HuggingFace authentication for Llama-3 tokenizer. The script will prompt for a token if needed. Alternatively, use any public Llama-family tokenizer as fallback.

---

## Running Experiments

### Experiment 1: Tokenization Fertility Benchmark
```bash
python experiments/exp1_tokenization_benchmark.py
```
**Output:** `figures/fig1_tokenization_fertility.png`, `data/tokenization_results.json`  
Measures tokens-per-word for 6 languages using Llama-3 tokenizer.

### Experiment 2: Memory Wall Simulation
```bash
python experiments/exp2_memory_wall_simulation.py
```
**Output:** `figures/fig2a-d` (4 figures), `data/memory_simulation_results.json`  
Simulates KV cache growth, batch size limits, throughput degradation, and compute distribution.

### Experiment 3: Combined Analysis
```bash
python experiments/exp3_combined_analysis.py
```
**Output:** `figures/fig3a-d` (4 figures), `data/combined_analysis_results.json`  
End-to-end latency, Resource Divide Index, cumulative disadvantage, and language support gap.

### Experiment 4: Multi-Turn Conversational Serving
```bash
python experiments/exp4_multiturn_serving.py
```
**Output:** `data/experiment_4_results.json`  
Simulates agentic multi-turn conversations with batch_size=32 serving scenario. Tracks KV cache accumulation and OOM thresholds across hardware tiers and languages.

**Key Parameters:**
- `batch_size`: Concurrent users (default: 32)
- `num_turns`: Conversation length (default: 20)
- `words_per_turn`: Tokens per conversation turn (default: 80)

### Experiment 5: Prefill-Decode Disaggregation
```bash
python experiments/exp5_prefill_decode_disaggregation.py
```
**Output:** Console output with latency breakdowns  
Analyzes the distinct hardware requirements of prefill (compute-bound) vs decode (memory-bound) phases. Evaluates heterogeneous cluster configurations and Pareto-optimal cost-latency tradeoffs.

**Workloads analyzed:**
- Short Q&A (50 input, 30 output tokens)
- Medium Essay (500 input, 300 output tokens)
- Long Document (4000 input, 500 output tokens)
- Code Generation (200 input, 800 output tokens)

### Experiment 6: Language Ecosystem Analysis
```bash
python experiments/exp6_language_ecosystem.py
```
**Output:** `data/experiment_6_results.json`, console CRDI ranking  
Combines speaker population, internet content, digital literacy, LLM accuracy, fertility, and Wikipedia activity into a Composite Resource Divide Index (CRDI). Computes Pearson correlations across all metrics.

---

## Reproducing All Results

To reproduce all experiments and generate all figures:

```bash
# 1. Install dependencies
pip install transformers numpy matplotlib scipy

# 2. Run all experiments sequentially
python experiments/exp1_tokenization_benchmark.py
python experiments/exp2_memory_wall_simulation.py
python experiments/exp3_combined_analysis.py
python experiments/exp4_multiturn_serving.py
python experiments/exp5_prefill_decode_disaggregation.py
python experiments/exp6_language_ecosystem.py

# 3. Verify outputs
ls figures/
ls data/
```

Expected outputs:
- **13 figures** in `figures/`
- **5 JSON data files** in `data/`

---

## Datasets Used

### Tokenizer
- `meta-llama/Llama-3.2-1B` tokenizer via HuggingFace Transformers
- Fallback: Any public Llama-family tokenizer

### Language Sample Texts
- English, Chinese, Spanish, Arabic, Swahili, Hindi
- Synthetic multilingual text passages (embedded in scripts)
- ~200-500 characters per language

### Hardware Specifications
- NVIDIA H100 SXM5 (80GB HBM3, 3.35 TB/s)
- NVIDIA A100 80GB (80GB HBM2e, 2.04 TB/s)
- NVIDIA RTX 3090 (24GB GDDR6X, 936 GB/s)
- NVIDIA T4 (16GB GDDR6, 320 GB/s)
- NVIDIA RTX 4060 (8GB GDDR6, 272 GB/s)

### Ecosystem Data Sources
- W3Techs / Intelpoint: Internet content by language (2024)
- ITU / Statista: Digital literacy and internet penetration (2024-2025)
- Ethnologue / Wikipedia: Speaker populations and contributor data
- Academic literature: LLM multilingual benchmark accuracies

---

## Hardware Requirements

All experiments are **CPU-only simulations** designed for reproducibility on standard academic hardware:

- No GPU required for reproduction
- All calculations are mathematical simulations (not actual inference)
- Total runtime: ~5 minutes for all 6 experiments
- RAM: ~2 GB

A GPU (NVIDIA RTX 3090 or equivalent) was used for tokenizer validation only.

---

## Figures Summary

| Figure | Description | Source Experiment |
|--------|-------------|-------------------|
| Fig 1 | Tokenization fertility & attention cost by language | Exp 1 |
| Fig 2a | KV cache growth vs sequence length | Exp 2 |
| Fig 2b | Max batch size heatmap (OOM thresholds) | Exp 2 |
| Fig 2c | Throughput degradation | Exp 2 |
| Fig 2d | Global compute distribution | Exp 2 |
| Fig 3a | End-to-end latency comparison | Exp 3 |
| Fig 3b | Resource Divide Index heatmap | Exp 3 |
| Fig 3c | Cumulative disadvantage trajectory | Exp 3 |
| Fig 3d | Language support gap | Exp 3 |
| Fig 4a | Multi-turn KV cache accumulation & concurrent users | Exp 4 |
| Fig 4b | Throughput degradation & OOM heatmap | Exp 4 |
| Fig 4c | Serving scenario OOM heatmap (batch=32) | Exp 4 |
| Fig 5a | Prefill vs decode latency breakdown by workload | Exp 5 |
| Fig 5b | Disaggregated serving Pareto frontier | Exp 5 |
| Fig 6 | Comprehensive language ecosystem dashboard (6 panels) | Exp 6 |

---

## Report

The full paper (`report/Resource_Divide_Paper.docx`) follows IEEE conference format:
- **Main body:** ~2,700 words (Sections A-E, ~5 pages)
- **References:** 21 IEEE-format citations
- **Appendix:** Compute and AI usage disclosure

Sections:
- **A:** Introduction (AI divide, policy context, scope)
- **B:** Literature Review (compute, tokenization, data, memory)
- **C:** Technical Experiments (6 subsections, 4 tables, 6 figures)
- **D:** Dual Viewpoint Analysis (Bo Pang vs Connor Sempf)
- **E:** Conclusion (joint assessment, best practices)

---

## Citation

If you use this work, please cite:
```
Connor Sempf and Bo Pang, "The Resource Divide: Data Scarcity, Compute Inequity, 
and the Future of Global LLM Governance," FunML Term Project, 
Georgia Institute of Technology, 2025.
```

---

## License

This project is released for academic use. All code, data, and figures are provided as-is for educational and research purposes.
