# The Resource Divide: Data Scarcity, Compute Inequity, and Global LLM Governance

**Connor Sempf** and **Bo Pang**  
School of Electrical and Computer Engineering, Georgia Institute of Technology  
{csempf3, bpang42}@gatech.edu

---

## Abstract

Large language models are positioned as universal tools, yet the resources required to train and deploy them are distributed unevenly across languages. We quantify compounding disadvantages across tokenization efficiency, memory scalability, and hardware access. Using the Llama-3 tokenizer, Hindi requires 2.83 tokens per word versus 1.15 for English, resulting in a 7.99× attention cost penalty. Memory simulation across five hardware tiers shows a 40× gap: RTX 4060 OOMs at 14K tokens while H100 sustains 544K. In serving scenarios with 32 concurrent users, Hindi OOMs after one turn on consumer GPUs versus three turns for English. Our Composite Resource Divide Index reveals 8.1× structural disparity between Swahili (0.800) and English (0.099).

---

## I. Introduction

The rapid proliferation of large language models has centralized AI capabilities despite open-weight releases [1]. The United States controls approximately 74% of global high-end AI compute, with 160 nations classified as "compute deserts" entirely dependent on foreign infrastructure [2][3]. This concentration is institutionalized: the CHIPS Act appropriates $52.7 billion while prohibiting semiconductor expansion in China [4], and export controls on H100/H200 GPUs create structural access asymmetries.

The AI divide extends to data. English dominates 49.7% of global web content, while Swahili---spoken by 200 million people---constitutes merely 0.0025% [5]. This creates a self-reinforcing cycle: less digital content leads to poorer language models, which reduces adoption, which discourages content creation. This paper quantifies how compute and data disparities manifest as compounding, measurable performance gaps in LLM deployment.

---

## II. Related Work

GovAI identifies compute as uniquely excludable, making it the primary lever for AI regulation [1]. Nations without territorial compute clusters lack "computational sovereignty" [2]. Lundin et al. demonstrate that each additional token per word reduces downstream accuracy by 8--18 percentage points, coining the "tokenization tax" [6]. Of approximately 2,000 African languages, only 42 have meaningful LLM support [7][8]. Hooper et al. identify KV cache as the fundamental bottleneck for long-context inference [9]. Recent work establishes that prefill (compute-bound) and decode (memory-bound) phases have distinct hardware requirements, motivating disaggregated serving [10][11].

---

## III. Methodology and Experiments

We employ the Llama-3 tokenizer for real benchmarks and a KV cache simulation framework modeling Llama-3-8B with 4-bit quantization across five hardware tiers: H100 (80GB HBM3), A100 (80GB HBM2e), RTX 3090 (24GB), T4 (16GB), and RTX 4060 (8GB). KV cache per token is 0.125 MB for FP16 precision.

### A. Tokenization Fertility Results

Table I presents tokenization fertility (tokens per word) and attention cost multiplier (fertility²). Hindi requires 2.83 tokens per word---2.46× more than English's 1.15. Because attention scales as O(n²), Hindi incurs 7.99× attention cost versus English's 1.32×, a 6× penalty.

| Language | Script | Fertility | Attn. Cost |
|----------|--------|-----------|------------|
| Chinese  | Han    | 0.73      | 0.53       |
| English  | Latin  | 1.15      | 1.32       |
| Spanish  | Latin  | 1.68      | 2.81       |
| Swahili  | Latin  | 2.34      | 5.47       |
| Arabic   | Arabic | 2.40      | 5.76       |
| Hindi    | Devanagari | 2.83  | 7.99       |

*Table I: Tokenization fertility and attention cost by language.*

### B. Memory Wall Simulation

KV cache grows linearly with sequence length. Table II shows maximum sustainable sequence lengths before OOM. The consumer RTX 4060 hits the memory wall at just 435 tokens for batch size 32---less than one page of text. The H100 sustains 40× more concurrent tokens.

| Hardware | VRAM | Max Seq (B=1) | Max Seq (B=32) |
|----------|------|---------------|----------------|
| H100     | 80GB | 544,768       | 17,024         |
| A100     | 80GB | 544,768       | 17,024         |
| RTX 3090 | 24GB | 131,891       | 4,121          |
| T4       | 16GB | 72,908        | 2,278          |
| RTX 4060 | 8GB  | 13,926        | 435            |

*Table II: Maximum sequence length before OOM.*

### C. End-to-End Latency and Serving

For an identical 100-word task, English on H100 completes in 0.7s; Hindi on T4 requires 18.8s---a 26× gap. In multi-turn serving with 32 concurrent users, the RTX 4060 OOMs after one turn for Hindi versus three for English. The T4 OOMs at turn 6 for Hindi versus turn 13 for English.

### D. Language Ecosystem Analysis

We synthesize six metrics into a Composite Resource Divide Index (CRDI): speaker population, internet content, digital literacy, LLM accuracy, tokenization fertility, and Wikipedia contributors. Swahili scores 0.800 versus English's 0.099---an 8.1× structural disadvantage.

| Language | CRDI  | Tier     |
|----------|-------|----------|
| Swahili  | 0.800 | Low      |
| Bengali  | 0.779 | Low      |
| Arabic   | 0.674 | Moderate |
| Hindi    | 0.668 | Moderate |
| Spanish  | 0.607 | High     |
| English  | 0.099 | High     |

*Table III: Composite Resource Divide Index.*

---

## IV. Dual Viewpoint Analysis

### Viewpoint A (Bo Pang): Structural Governance Imperative

The experimental evidence supports urgent structural intervention. The compute gap is not self-correcting: a T4's GDDR6 cannot upgrade to HBM3e, and export controls prevent frontier hardware acquisition [1][4]. The tokenization tax is equally structural---Hindi's 2.83 tokens per word stems from English-centric BPE training. Market incentives do not fund inclusive tokenizer redesign because Hindi-speaking markets generate less revenue. Policy recommendations include: (1) Sovereign AI funds for regional compute clusters; (2) International Compute Commons agreements; (3) Mandatory inclusive tokenizer design for large models.

### Viewpoint B (Connor Sempf): Data-Driven Self-Correction

While Viewpoint A identifies compute barriers correctly, it risks overemphasizing hardware at the expense of data---the root cause. CRDI analysis shows LLM accuracy strongly correlates with internet content (r=0.66) and negatively with fertility (r=-0.86). Without representative corpora, no compute quantity bridges the gap. Market mechanisms can work: targeted investment in data commons breaks the cyclic disadvantage. The cost of curating 10 billion Swahili tokens is orders of magnitude less than building H100 clusters.

### Synthesis

Both viewpoints identify real, non-overlapping barriers. English benefits across all dimensions simultaneously---a structural advantage no single intervention can replicate. However, compounding disadvantage implies compounding returns from coordinated action: inclusive tokenizers reduce fertility, data commons improve model quality, and compute sharing enables deployment where markets fail. Neither governance redistribution nor data investment alone suffices.

---

## V. Conclusion

This paper demonstrates that the AI divide is real, quantifiable, and compounding. Key findings: (1) Hindi pays 7.99× attention cost versus English's 1.32×; (2) Consumer GPUs OOM at 14K tokens while frontier GPUs sustain 544K; (3) In serving scenarios, Hindi OOMs after one conversation turn on consumer hardware; (4) CRDI reveals 8.1× structural disparity between Swahili and English. Moving forward requires coordinated action: international data commons for low-resource languages, regional compute-sharing agreements with CRDI-based metrics, and mandatory inclusive tokenizer design.

---

## References

[1] GovAI, "Computing Power and the Governance of AI," arXiv:2409.02888, 2024.
[2] V. Lehdonvirta et al., "Computational Sovereignty and the AI Divide," Proc. AIES, 2024.
[3] Tony Blair Institute, "State of Compute Access," Policy Report, 2024.
[4] U.S. Congress, "CHIPS and Science Act of 2022," Public Law 117-167, 2022.
[5] W3Techs, "English accounts for 49.40% of internet content," 2024.
[6] J. Lundin et al., "The Token Tax: Systematic Bias in Multilingual Tokenization," arXiv:2509.05486, 2025.
[7] M. Abbott et al., "Masakhane---Machine Translation for Africa," arXiv:2003.11529, 2020.
[8] A. Adebara et al., "The State of LLMs for African Languages," arXiv:2506.02280, 2025.
[9] C. Hooper et al., "KVQuant: Towards 10 Million Context Length," UC Berkeley EECS, 2024.
[10] WVA, "A Global Optimization Control Plane for LLM Inference," arXiv:2603.09730, 2026.
[11] DUET, "Disaggregated Hybrid Mamba-Transformer LLMs," 2025.

---

## Appendix: Compute and AI Usage Disclosure

**Compute Usage:** Georgia Tech AI Makerspace, ~12 hours across 3 sessions. All experiments are CPU-runnable simulations. RTX 3090 used for validation only. Total compute time: ~20 hours.

**AI Usage Disclosure:** Code developed with AI assistance for boilerplate and matplotlib styling. Literature research AI-assisted but manually verified. Writing AI-assisted for organization and grammar. CRDI, RDI metrics, and simulation frameworks are original contributions.

**Code Repository:** https://github.com/Pb314314/Fund_ML
