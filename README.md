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
- Consumer GPUs (RTX 4060) OOM at 26K tokens; H100 sustains 557K (21x gap)
- In serving with 32 concurrent users, Hindi OOMs at turn 2 on RTX 4060; English at turn 5
- Decode dominates 85.6--99.6% of end-to-end latency (memory bandwidth is bottleneck)
- CRDI: Swahili 0.807 vs English 0.100 (8.1x structural disadvantage)
- Disaggregated serving (H100 prefill + RTX 4060 decode) halves cost at 13x latency penalty
- Reducing model constriction and allowing more data during training increases accuracy notably

---

## Repository Structure

```
project/
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
├── experiments/
│   ├── config.py                                   # UNIFIED CONFIG: all shared constants
│   ├── exp1_tokenization_benchmark.py              # Real tokenizer fertility (6 languages)
│   ├── exp2_memory_wall_simulation.py              # KV cache growth + OOM simulation (5 GPUs)
│   ├── exp3_combined_analysis.py                   # End-to-end latency + RDI index
│   ├── exp4_multiturn_serving.py                   # Multi-turn conversation + serving capacity
│   ├── exp5_prefill_decode_disaggregation.py       # Phase-specific hardware + Pareto analysis
│   └── exp6_language_ecosystem.py                  # CRDI + 6-metric ecosystem dashboard
│   └── exp7_data_constriction_and_accuracy.ipynb   # CRDI + 6-metric ecosystem dashboard
├── figures/                                        # 18 quality figures
├── data/                                           # 7 JSON result files from experiments
└── report/
    └── FinalReport_TheResourceDivide.pdf           # Final Report in IEEE-Format
```

---

## Setup Instructions

### Requirements
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** `transformers` requires HuggingFace authentication for Llama-3 tokenizer. The script will prompt for a token if needed, or use a public fallback tokenizer.

---

## Running Experiments

All experiments import from `config.py`---the single source of truth for model specs, GPU hardware, language fertility, and ecosystem data. This ensures cross-experiment numerical consistency.

### Quick Start: Run All Experiments
```bash
cd experiments/
python exp1_tokenization_benchmark.py           # ~2 min (needs HF auth)
python exp2_memory_wall_simulation.py           # ~10 sec
python exp3_combined_analysis.py                # ~10 sec
python exp4_multiturn_serving.py                # ~10 sec
python exp5_prefill_decode_disaggregation.py    # ~10 sec
python exp6_language_ecosystem.py               # ~10 sec
## manual python notebook execution for exp7_data_constriction_and_accuracy.ipynb   # ~30 min
```

### Individual Experiments

| Exp | Script | What It Does | Output |
|-----|--------|-------------|--------|
| 1 | `exp1_tokenization_benchmark.py` | Real Llama-3 tokenizer on 6 languages | `fig1_tokenization_fertility.png` |
| 2 | `exp2_memory_wall_simulation.py` | KV cache growth, batch limits, throughput, compute distribution | `fig2a-2d` (4 figures) |
| 3 | `exp3_combined_analysis.py` | End-to-end latency, RDI heatmap, cumulative disadvantage, language gap | `fig3a-3d` (4 figures) |
| 4 | `exp4_multiturn_serving.py` | Agentic multi-turn serving (batch=32), OOM thresholds | `fig4a-4b` (2 figures) |
| 5 | `exp5_prefill_decode_disaggregation.py` | Prefill vs decode breakdown, Pareto frontier | `fig5a-5b` (2 figures) |
| 6 | `exp6_language_ecosystem.py` | CRDI, correlations, radar charts, tier distribution | `fig6_language_ecosystem.png` |
| 7 | `exp7_data_constriction_and_accuracy.ipynb` | model accuracy against dataset constriction | `fig7a-7b` (2 figures) |

### Experiment Details

**Experiment 1: Tokenization Fertility**
- Uses real `meta-llama/Llama-3.2-1B` tokenizer via HuggingFace
- Tests: English, Chinese, Spanish, Arabic, Swahili, Hindi
- Metrics: tokens-per-word, attention cost multiplier (fertility^2)

**Experiment 2: Memory Wall Simulation**
- Model: Llama-3-8B AWQ-4bit (4.01 GB weights, 8.03B params)
- KV cache: 131,072 bytes/token = 0.125 MB/token (FP16)
- GPUs: H100 (3.35 TB/s), A100 (2.04 TB/s), RTX 3090 (936 GB/s), T4 (320 GB/s), RTX 4060 (272 GB/s)
- Simulates: KV growth curves, max batch size heatmap, throughput degradation, compute distribution

**Experiment 3: Combined Analysis**
- End-to-end latency = prefill (compute-bound) + decode (memory-bound)
- Task: 100-word input, 50-word output
- Resource Divide Index: RDI = fertility^2 / (bandwidth * VRAM)
- Cumulative disadvantage: 5-turn agentic conversation

**Experiment 4: Multi-Turn Serving**
- Scenario: 32 concurrent users, 80 words/turn, 20 max turns
- Tracks: KV cache accumulation, throughput decay, OOM turn
- Key insight: Hindi OOMs 2.5x faster than English on same hardware

**Experiment 5: Prefill-Decode Disaggregation**
- Prefill: parallel processing, compute-bound, 85% GPU util
- Decode: autoregressive, memory-bound, 20-40% GPU util
- 4 workload types: Short Q&A, Medium Essay, Long Document, Code Generation
- Pareto analysis: cost-latency tradeoff for heterogeneous clusters

**Experiment 6: Language Ecosystem**
- 6 metrics: speakers, content%, literacy, LLM accuracy, fertility, Wikipedia
- CRDI weights: content (25%), accuracy (20%), literacy/fertility/wiki (15% each), speakers (10%)
- Pearson correlation matrix across all metrics

**Experiment 7: Model Accuracy Against Dataset Constriction**
- Constrict the civil comments subset of WILDS dataset into 10%, 20%, and 50% groups
- Train AutoModelForSequenceClassification with tokenization under the above dataset constrictions
- Track the accuracy scores as the model is given more data each training set

---

## Reproducing All Results

```bash
# 1. Clone and setup
git clone https://github.com/Pb314314/Fund_ML.git
cd Fund_ML
pip install -r requirements.txt

# 2. Run all experiments
cd experiments/
python exp1_tokenization_benchmark.py
python exp2_memory_wall_simulation.py
python exp3_combined_analysis.py
python exp4_multiturn_serving.py
python exp5_prefill_decode_disaggregation.py
python exp6_language_ecosystem.py
## manual python notebook execution for exp7_data_constriction_and_accuracy.ipynb

# 3. Verify outputs
ls ../figures/   # PNG Figure Files
ls ../data/      # JSON Data Files
```

Expected outputs:
- **18 figures** in `figures/`
- **7 JSON data files** in `data/`

---

## Datasets and Sources

### Tokenizer and Models
- `meta-llama/Llama-3.2-1B` via HuggingFace Transformers
- AutoTokenizer and AutoModelForSequenceClassification Transformers
- Fallback: any public Llama-family tokenizer

### GPU Hardware Specifications
All sourced from NVIDIA official datasheets:
- For Experiment 7, "first available" GPU is fine
- H100 SXM5: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
- A100 80GB: https://www.nvidia.com/en-us/data-center/a100/
- RTX 3090: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/
- T4: https://www.nvidia.com/en-us/data-center/tesla-t4/
- RTX 4060: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060/

### Datasets and Language Ecosystem Data
- WILDS Dataset and Civil Comments Subset for Experiment 7
- Speaker populations: Ethnologue 2024
- Internet content %: W3Techs 2024 survey (top 10M websites)
- Digital literacy: ITU Digital Development Report 2024
- LLM accuracy: Approximated from multilingual benchmarks (Ahuja et al. ACL 2024; Masakhane)
- Wikipedia: Wikimedia Foundation 2024 active editor stats

---

## Hardware Requirements

All experiments other than Experiment 7 are **CPU-only simulations** designed for reproducibility:

- For Experiment 7, "first available" GPU is fine
- All calculations are mathematical simulations (not actual model inference)
- Total runtime: ~5 minutes for all 6 experiments
- RAM: ~2 GB
- Storage: ~50 MB for outputs

A GPU (RTX 3090) was used only for tokenizer validation in Experiment 1.

---

## Figures Summary

| Figure | Description | Experiment |
|--------|-------------|------------|
| Fig 1 | Tokenization fertility + attention cost (6 languages) | Exp 1 |
| Fig 2a | KV cache growth vs sequence length (batch sizes 1-32) | Exp 2 |
| Fig 2b | Max batch size heatmap (OOM boundaries, 5 GPUs x 8 seq lengths) | Exp 2 |
| Fig 2c | Throughput degradation: English vs Swahili | Exp 2 |
| Fig 2d | Global AI compute distribution pie chart | Exp 2 |
| Fig 3a | End-to-end latency by hardware and language (100-word task) | Exp 3 |
| Fig 3b | Resource Divide Index heatmap (6 languages x 4 GPUs) | Exp 3 |
| Fig 3c | Cumulative disadvantage over 5-turn conversation | Exp 3 |
| Fig 3d | Language support gap + digital content distribution | Exp 3 |
| Fig 4a | Multi-turn KV accumulation + max concurrent users | Exp 4 |
| Fig 4b | OOM heatmap: serving scenario (batch=32, 4 GPUs x 6 languages) | Exp 4 |
| Fig 5a | Prefill vs decode latency breakdown (4 workloads x 4 GPUs) | Exp 5 |
| Fig 5b | Pareto frontier: disaggregated serving cost-latency tradeoff | Exp 5 |
| Fig 6 | 6-panel ecosystem dashboard (CRDI, correlations, radar, tiers) | Exp 6 |
| Fig 7a | Model Accuracy vs. Dataset Constriction Results | Exp 7 |
| Fig 7a | Model F1 Score vs. Dataset Constriction Results | Exp 7 |

---

## Report

The full paper can be found in `report/FinalReport_TheResourceDivide.pdf`:

Sections:
- **A:** Introduction (AI divide, policy context, scope, contributions)
- **B:** Literature Review (compute, tokenization, data, memory)
- **C:** Technical Experiments (7 subsections, 1 for each experiment)
- **D:** Dual Viewpoint Analysis (Bo Pang vs. Connor Sempf)
- **E:** Conclusion (assessment, best practices)
- **References:** Sources from Literature
- **Appendix:** Compute and AI Usage

---

## Notes

If you use this work, please cite:
```
Connor Sempf and Bo Pang, "The Resource Divide: Data Scarcity, Compute Inequity,
and the Future of Global LLM Governance," FunML Term Project,
Georgia Institute of Technology, 2025.
```

This project is released for academic use. All code, data, and figures are provided as-is for educational and research purposes.
