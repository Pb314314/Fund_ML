# The Resource Divide: Data Scarcity, Compute Inequity, and the Future of Global LLM Governance

**Connor Sempf** and **Bo Pang**  
School of Electrical and Computer Engineering  
Georgia Institute of Technology  
Atlanta, GA, USA  
csempf3@gatech.edu, bpang42@gatech.edu

---

## Abstract

While large language models (LLMs) are positioned as universal tools, the resources required to train and deploy them are distributed highly unevenly across the world's languages. This paper quantifies the compounding disadvantages faced by low-resource languages across three critical dimensions: tokenization efficiency, memory scalability, and hardware access. Using the Llama-3 tokenizer, we benchmark six languages and find extreme fertility disparities---Hindi requires 2.83 tokens per word and Swahili 2.34, versus 1.15 for English---resulting in a 7.99$\times$ attention cost multiplier for Hindi compared to 1.32$\times$ for English. Through KV cache memory simulation of Llama-3-8B (4-bit AWQ, 4.01 GB) across five hardware tiers from NVIDIA H100 to RTX 4060, we demonstrate that the RTX 4060 hits an out-of-memory wall at 26,091 tokens while the H100 sustains 556,933 tokens, a 21$\times$ gap. In a realistic serving scenario with 32 concurrent users, Hindi conversations on an RTX 4060 out-of-memory at turn 2; English sustains 5 turns; on a T4, Hindi OOMs at turn 6 versus English at turn 15. Prefill-decode disaggregation analysis reveals that decode dominates 85.6--99.6% of end-to-end latency across all workload types, and heterogeneous clusters can halve serving costs by offloading decode to commodity GPUs---at a 13$\times$ latency penalty. End-to-end inference latency reveals that processing Hindi on a T4 GPU is 26$\times$ slower than English on an H100 for an identical 100-word task. A comprehensive language ecosystem analysis combining speaker population, digital content volume, digital literacy proxy, LLM benchmark accuracy, and tokenization fertility produces a Composite Resource Divide Index (CRDI) of 0.807 for Swahili versus 0.100 for English---an 8.1$\times$ structural disadvantage. We contextualize these findings against the reality that only 42 of approximately 7,150 world languages (0.6%) enjoy meaningful LLM support. The analysis offers a dual-viewpoint framework contrasting structural governance interventions with data-driven self-correction pathways, and delivers concrete policy recommendations spanning compute commons, targeted data investment, and inclusive tokenizer design.

---

## Section A: Introduction

### A.1 The AI Divide as Global Risk

The rapid proliferation of large language models (LLMs) has centralized AI capabilities in a handful of nations and corporations despite open-weight releases [1]. The United States controls approximately 74% of global high-end AI compute, with China at 14% and the European Union at merely 5% [2]. An estimated 160 nations are classified as "compute deserts," entirely dependent on foreign cloud infrastructure [3]. Within Africa, only 38% of the population used the internet in 2024, with rural penetration at just 23% versus 57% in urban areas [4]. Sub-Saharan Africa accounts for 52% of the global population facing digital exclusion [4]. This concentration is not accidental: the U.S. CHIPS Act appropriates $52.7 billion while explicitly prohibiting semiconductor expansion in China for ten years [5], and export controls on NVIDIA H100/H200 GPUs institutionalize access asymmetries.

The AI divide extends beyond compute to data. English dominates 49.7% of global web content, while Swahili---spoken by 200 million people---constitutes merely 0.0025% [6]. Hindi, with 600 million speakers, accounts for only 3.77% of internet content. This data imbalance creates a self-reinforcing cycle: less digital content leads to poorer language models, which reduces adoption, which discourages content creation [7].

### A.2 Scope and Objectives

This paper investigates how compute and data disparities manifest as compounding, quantifiable performance gaps in LLM deployment. We quantify the "tokenization tax" (higher tokens-per-word ratios for low-resource languages), the "memory wall" (KV cache limits on consumer hardware), and their interaction in realistic serving scenarios. We further analyze the distinct hardware requirements of prefill (compute-bound) versus decode (memory-bound) phases, evaluate heterogeneous cluster scheduling, and contextualize technical findings within a comprehensive language ecosystem framework combining speaker population, digital content, literacy rates, and model performance.

### A.3 Contribution Statement

This paper makes three contributions: (1) the first empirical quantification of compounding resource divide using real tokenizer benchmarks and memory simulation across hardware tiers; (2) a serving-systems analysis of multi-turn conversational KV cache accumulation and prefill-decode disaggregation; and (3) a Composite Resource Divide Index (CRDI) synthesizing six metrics to rank global language AI readiness.

---

## Section B: Literature Review

### B.1 Geopolitics of Compute

GovAI identifies compute as uniquely excludable and detectable, making it the primary lever for AI regulation [1]. Lehdonvirta et al. argue that nations without territorial compute clusters lack "computational sovereignty" and remain perpetually dependent [2]. The Tony Blair Institute and UC Berkeley AI Policy Hub document how supply-chain concentration creates structural barriers that market mechanisms cannot self-correct [3][8].

### B.2 Linguistic Inequity: The Tokenization Tax

Lundin et al. demonstrate that each additional token per word reduces downstream accuracy by 8--18 percentage points, coining the "tokenization tax" [7]. Ahuja et al. show that low-resource languages face systematic cost-performance disadvantages in LLM inference [9]. Our analysis extends this by showing that attention costs scale as O(n$^2$) with token count, amplifying disparities quadratically.

### B.3 Data Imbalance and Model Bias

Joshi et al. and the Masakhane project document that of approximately 2,000 African languages, only about 42 have meaningful LLM support [10][11]. Swahili's 200 million speakers have access to only 700 million digital tokens---orders of magnitude less than English's trillions. The "Stochastic Parrots" framework establishes that biases in training data encode systemic privileging of dominant languages and cultures [12]. African Languages Lab's recent work on 40 languages with 19 billion tokens provides a foundation, yet remains minuscule compared to English corpora [11].

### B.4 The Memory Wall and Serving Systems

Hooper et al. identify KV cache as the fundamental bottleneck for long-context inference [13]. vLLM's PagedAttention reduces memory waste from 60--80% to under 4%, enabling 2--4$\times$ throughput improvements [14][15]. Recent work by WVA and DUET establishes that prefill and decode phases have fundamentally different hardware requirements---compute-bound versus memory-bound---motivating phase-specific hardware disaggregation [16][17]. HBM3e bandwidth at 3.35 TB/s versus GDDR6 at 272 GB/s creates a 12$\times$ gap that directly governs decode throughput [18].

---

## Section C: Technical Experiments

### C.1 Experimental Design

Our experiments employ the Llama-3 tokenizer (via HuggingFace Transformers) for real tokenization benchmarks, and a KV cache memory simulation framework modeling Llama-3-8B with 4-bit AWQ quantization (4.01 GB weights, standard for production serving) across five hardware tiers: H100 (80GB HBM3, 3.35 TB/s), A100 (80GB HBM2e, 2.04 TB/s), RTX 3090 (24GB GDDR6X, 936 GB/s), T4 (16GB GDDR6, 320 GB/s), and RTX 4060 (8GB GDDR6, 272 GB/s). All GPU specifications are sourced from NVIDIA official datasheets.

KV cache per token is 131,072 bytes (0.125 MB) in FP16 precision:
$$\text{KV} = 2 \times L \times H_{\text{KV}} \times d \times \text{dtype\_size} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes}$$

### C.2 Tokenization Fertility Results

Table I presents tokenization fertility for six languages. Hindi requires 2.83 tokens per word---2.46$\times$ more than English's 1.15. Because attention scales quadratically with token count, the "attention cost multiplier" (fertility$^2$) shows Hindi at 7.99$\times$ versus English at 1.32$\times$---a 6$\times$ penalty. Swahili at 2.34 tokens/word incurs a 5.47$\times$ attention cost.

| Language | Script | Tokens | Words | Fertility | Attn. Cost |
|----------|--------|--------|-------|-----------|------------|
| Chinese  | Han    | 76     | 104   | 0.73      | 0.53       |
| English  | Latin  | 61     | 53    | 1.15      | 1.32       |
| Spanish  | Latin  | 104    | 62    | 1.68      | 2.81       |
| Swahili  | Latin  | 131    | 56    | 2.34      | 5.47       |
| Arabic   | Arabic | 108    | 45    | 2.40      | 5.76       |
| Hindi    | Devanagari | 212 | 75   | 2.83      | 7.99       |

*Table I: Tokenization fertility and attention cost multiplier by language (Llama-3 tokenizer).*

### C.3 Memory Wall Simulation Results

KV cache grows linearly with sequence length. Table II shows maximum sustainable sequence lengths before OOM for Llama-3-8B 4-bit (4.01 GB weights) at various batch sizes.

| Hardware | VRAM | Avail. KV | Max Seq (B=1) | Max Seq (B=32) |
|----------|------|-----------|---------------|----------------|
| H100     | 80GB | 67.98 GB  | 556,933 tok   | 17,404 tok     |
| A100     | 80GB | 67.98 GB  | 556,933 tok   | 17,404 tok     |
| RTX 3090 | 24GB | 17.59 GB  | 144,056 tok   | 4,502 tok      |
| T4       | 16GB | 10.38 GB  | 85,073 tok    | 2,658 tok      |
| RTX 4060 | 8GB  | 3.19 GB   | 26,091 tok    | 815 tok        |

*Table II: Maximum sequence length before OOM (Llama-3-8B 4-bit AWQ, 10% overhead).*

The consumer GPU (RTX 4060) hits the memory wall at just 815 tokens for batch size 32---less than one page of text. The H100 sustains 21$\times$ more concurrent tokens.

### C.4 Combined End-to-End Latency Analysis

For an identical 100-word semantic task (50-word output), end-to-end latency reveals compounding disadvantage. Table III shows the breakdown into prefill (time-to-first-token, TTFT) and decode phases.

| Language | GPU | Prefill (ms) | Decode (ms) | Total (ms) | Decode % |
|----------|-----|--------------|-------------|------------|----------|
| English  | H100| 4.4          | 98.0        | 102.4      | 95.7     |
| English  | T4  | 33.2         | 1026.1      | 1059.3     | 96.9     |
| Hindi    | H100| 10.9         | 247.7       | 258.6      | 95.8     |
| Hindi    | T4  | 82.6         | 2593.5      | 2676.1     | 96.9     |

*Table III: End-to-end latency for 100-word task, 50-word output (Llama-3-8B 4-bit).*

English on H100: 102.4 ms. Hindi on T4: 2,676.1 ms---a 26$\times$ gap. The Resource Divide Index (RDI = fertility$^2$ / (bandwidth $\times$ VRAM)) reaches 1,010$\times$ between extremes.

### C.5 Multi-Turn Conversational Serving Simulation

Real-world LLM applications---customer service bots, medical diagnostic assistants, educational tutors---operate as multi-turn conversations where each turn appends to the KV cache. We simulate a serving scenario with 32 concurrent users, each engaging in a multi-turn conversation with 80 words per turn. Each turn adds user input plus model response to the KV cache. Table IV shows the turn at which OOM occurs for each (hardware, language) combination.

| Hardware | English | Hindi | Swahili | Chinese | Spanish | Arabic |
|----------|---------|-------|---------|---------|---------|--------|
| H100     | 20+     | 20+   | 20+     | 20+     | 20+     | 20+    |
| A100     | 20+     | 20+   | 20+     | 20+     | 20+     | 20+    |
| T4       | 15      | 6     | 6       | 20+     | 10      | 7      |
| RTX 4060 | 5       | 2     | 2       | 20+     | 4       | 2      |

*Table IV: Maximum turns before OOM (batch size = 32, 80 words/turn).*

The RTX 4060 OOMs after just 2 turns for Hindi (batch=32) versus 5 turns for English. The T4 OOMs at turn 6 for Hindi versus turn 15 for English. Only frontier GPUs (H100, A100) sustain all 20 turns for all languages. This demonstrates that low-resource languages not only cost more per token but also reduce service capacity---a Hindi chatbot can serve 2.5$\times$ fewer concurrent users than an English equivalent before memory exhaustion.

### C.6 Prefill-Decode Disaggregation Analysis

LLM inference comprises two fundamentally different phases [16][17]. Prefill processes all input tokens in parallel via large matrix multiplications---compute-bound, achieving 85% GPU utilization on H100. Decode generates one token at a time, reading the entire KV cache from memory each step---memory-bound, with only 20--40% utilization.

Table V breaks down latency by workload type. Decode dominates 85.6--99.6% of end-to-end latency across all workloads. For code generation (200 input, 800 output tokens), decode consumes 99.4% of time on H100.

| Workload | GPU | Prefill (ms) | Decode (ms) | Total (ms) | Decode % |
|----------|-----|--------------|-------------|------------|----------|
| Short Q&A| H100| 1.9          | 51.5        | 53.4       | 96.4     |
| Medium Essay| H100| 19.2      | 522.0       | 541.3      | 96.4     |
| Long Doc | H100| 162.6        | 967.9       | 1130.5     | 85.6     |
| Code Gen | H100| 7.7          | 1378.7      | 1386.4     | 99.4     |

*Table V: Prefill vs decode latency breakdown (Llama-3-8B 4-bit).*

Figure 5 shows the Pareto frontier for disaggregated serving. Assigning prefill to H100 and decode to RTX 4060 halves serving cost ($0.00185 vs $0.00373 per request) but increases latency by 13$\times$ (7,059 ms vs 541 ms). For cost-sensitive deployments in the Global South, this tradeoff is unavoidable---H100-class hardware is simply unavailable.

### C.7 Comprehensive Language Ecosystem Analysis

Figure 6 presents a six-panel dashboard analyzing language AI readiness across ten major languages. We synthesize six metrics into a Composite Resource Divide Index (CRDI) with explicit weights: speaker population (10%), internet content percentage (25%), digital literacy proxy (15%), estimated LLM MMLU accuracy (20%), tokenization fertility (15%), and Wikipedia contributor count (15%).

The CRDI reveals striking disparities (Table VI). Swahili scores 0.807---the worst in our sample---despite 200 million speakers, due to minuscule digital content (0.0025%), low digital literacy proxy (38%), poor LLM accuracy (45%), and high fertility (2.34). Bengali follows at 0.739. English scores 0.100---an 8.1$\times$ advantage over Swahili. The correlation heatmap (Fig. 6e) reveals that internet content percentage correlates strongly with Wikipedia activity (r=0.93) and LLM accuracy (r=0.66), while fertility shows strong negative correlation with LLM accuracy (r=-0.67)---confirming that tokenization cost directly predicts model performance.

| Language | CRDI | DOI | Tier |
|----------|------|-----|------|
| Swahili  | 0.807 | 0.193 | Low |
| Hindi    | 0.750 | 0.250 | Moderate |
| Bengali  | 0.739 | 0.261 | Low |
| Arabic   | 0.687 | 0.313 | Moderate |
| Russian  | 0.638 | 0.362 | Moderate |
| French   | 0.601 | 0.399 | High |
| Japanese | 0.601 | 0.399 | High |
| Spanish  | 0.581 | 0.419 | High |
| Chinese  | 0.423 | 0.577 | High |
| English  | 0.100 | 0.900 | High |

*Table VI: Composite Resource Divide Index (CRDI) and Digital Opportunity Index (DOI).*

---

## Section D: Dual Viewpoint Analysis

### D.1 Viewpoint A (Bo Pang): Structural Governance Imperative

The experimental evidence supports an urgent need for structural intervention. The compute gap is not self-correcting: a T4 GPU's GDDR6 memory cannot upgrade itself to HBM3e, and export controls prevent acquisition of frontier hardware [1][5]. The RTX 4060 OOMs after two turns of Hindi conversation with 32 concurrent users---not due to algorithmic inefficiency, but because 8 GB of GDDR6 is structurally insufficient for multilingual serving.

The tokenization tax is equally structural. Hindi's 2.83 tokens per word stems from English-centric BPE training on 15 trillion tokens---a corpus where Hindi constitutes a negligible fraction [7]. Vocabulary extension experiments show Hindi reducible from 2.61 to 1.19 tokens per word with targeted tokenizer redesign [19], but market incentives do not fund such redesign because Hindi-speaking markets generate less revenue per user.

The CRDI data reinforces this argument. Swahili's CRDI of 0.807 versus English's 0.100 reflects a systemic failure across all six measured dimensions---not merely one correctable deficiency. Policy recommendations include: (1) Sovereign AI funds financing regional compute clusters; (2) International Compute Commons agreements modeled on CERN; (3) Mandatory inclusive tokenizer design for models above threshold parameter counts; (4) UNESCO-expanded governance monitoring CRDI-like indices.

### D.2 Viewpoint B (Connor Sempf): Data-Driven Self-Correction

While Viewpoint A correctly identifies compute barriers, it risks overemphasizing hardware at the expense of data---the root cause of poor LLM performance. The CRDI correlation analysis shows LLM accuracy strongly correlates with internet content (r=0.66) and negatively with fertility (r=-0.67). Without representative training corpora, no compute quantity bridges the gap: a Swahili model trained on 700 million tokens will underperform regardless of whether inference runs on H100 or T4.

The data argument is empirically grounded. Masakhane and African Languages Lab demonstrate that targeted data collection---even modest-scale community efforts---improves performance substantially [10][11]. XBridge and similar projects show that vocabulary extension and multilingual pre-training reduce fertility premiums by 40--60% [19]. Market mechanisms can work: India's 600 million Hindi speakers represent an addressable market; as demand for Hindi LLM applications grows, commercial incentives will drive data collection.

The cyclic disadvantage can be broken by targeted investment in data commons. The cost of curating 10 billion Swahili tokens is orders of magnitude less than building H100 clusters---and data, unlike compute, is non-excludable once released. Technology solutions including 4-bit quantization, KV cache compression, and speculative decoding reduce hardware requirements by 2--4$\times$ [14][15], enabling adequate performance on commodity hardware if models are sufficiently trained.

### D.3 Synthesis

Both viewpoints identify real, non-overlapping barriers. The CRDI framework shows that English benefits across all six dimensions simultaneously---a structural advantage that no single intervention can replicate. However, the compounding nature of disadvantage also implies compounding returns from coordinated action: inclusive tokenizers reduce fertility (improving compute efficiency), data commons improve model quality (reducing need for frontier hardware), and compute sharing agreements enable deployment where markets fail. Neither governance redistribution nor data investment alone suffices; a dual strategy addressing both data scarcity and compute inequity in parallel is required.

---

## Section E: Conclusion

### E.1 Joint Assessment

This paper demonstrates that the AI divide is real, quantifiable, and compounding across tokenization, memory, hardware, and data ecosystems. Key findings include: (1) Hindi pays a 7.99$\times$ attention cost penalty versus English's 1.32$\times$; (2) consumer GPUs OOM at 26K tokens while frontier GPUs sustain 557K---a 21$\times$ gap; (3) in serving scenarios with 32 concurrent users, Hindi on RTX 4060 OOMs at turn 2 versus turn 5 for English; (4) decode dominates 85.6--99.6% of latency, making memory bandwidth the binding constraint; (5) the CRDI shows 8.1$\times$ structural disparity between Swahili (0.807) and English (0.100).

### E.2 Best Practices Forward

Moving forward requires coordinated action across three fronts. First, establish international data commons for low-resource languages, modeled on Masakhane but scaled to billion-token corpora. Second, implement regional compute-sharing agreements and sovereign AI funds, with CRDI-based metrics tracking progress. Third, mandate inclusive tokenizer design as standard practice---not afterthought---for models above threshold sizes. The dual-viewpoint analysis shows neither markets nor governance alone suffices; only parallel investment in data equity and compute access can break the cyclic exclusion of the majority of the world's languages from the LLM revolution.

---

## References

[1] GovAI, "Computing Power and the Governance of Artificial Intelligence," arXiv:2409.02888, 2024.

[2] V. Lehdonvirta et al., "Compute North vs. Compute South: Computational Sovereignty and the AI Divide," Proc. AIES, 2024.

[3] Tony Blair Institute, "State of Compute Access: Measuring Global AI Infrastructure," Policy Report, 2024.

[4] Nature, "Bridging the Digital Divide: Digital Literacy and Access in Rural Communities in China and Nigeria," 2026.

[5] U.S. Congress, "CHIPS and Science Act of 2022," Public Law 117-167, Aug. 2022.

[6] W3Techs / Intelpoint, "English accounts for 49.40% of internet content," 2024.

[7] J. Lundin et al., "The Token Tax: Systematic Bias in Multilingual Tokenization," arXiv:2509.05486, 2025.

[8] UC Berkeley AI Policy Hub, "An Evolving AI Supply Chain," Policy Report, 2024.

[9] K. Ahuja et al., "Cost-Performance Optimization for Low-Resource Languages," ACL, 2024.

[10] M. Abbott et al., "Masakhane---Machine Translation for Africa," arXiv:2003.11529, 2020.

[11] A. Adebara et al., "The State of LLMs for African Languages," arXiv:2506.02280, 2025.

[12] E. Bender et al., "On the Dangers of Stochastic Parrots," Proc. FAccT, 2021.

[13] C. Hooper et al., "KVQuant: Towards 10 Million Context Length with Quantized Key-Value Caches," UC Berkeley EECS, 2024.

[14] W. Kwon et al., "vLLM: Efficient Memory Management for LLM Serving with PagedAttention," SOSP, 2023.

[15] Introl, "KV Cache Optimization: Memory Efficiency for Production LLMs," 2026.

[16] WVA, "A Global Optimization Control Plane for LLM Inference," arXiv:2603.09730, 2026.

[17] DUET, "Disaggregated Hybrid Mamba-Transformer LLMs with Prefill and Decode-Specific Packages," 2025.

[18] Towards Data Science, "Prefill Is Compute-Bound, Decode Is Memory-Bound," 2026.

[19] A. Nanda et al., "Reducing Tokenization Premiums for Low-Resource Languages," arXiv:2601.13328, 2026.

[20] UNESCO, "Recommendation on the Ethics of Artificial Intelligence," 2021.

[21] European Parliament, "Regulation (EU) 2024/1689---AI Act," 2024.

---

## Appendix: Compute and AI Usage Disclosure

### Compute Usage
This project used the Georgia Tech AI Makerspace and local development. All experiments are CPU-runnable simulations designed to be reproducible on standard academic hardware.

| Resource | Usage |
|----------|-------|
| AI Makerspace Hours | ~12 hours across 3 sessions |
| GPU Nodes | 0 (CPU-only reproduction) |
| GPUs Used | RTX 3090 for tokenizer validation only |
| Total Compute Time | ~20 hours (development + reproduction) |

### AI Usage Disclosure
In accordance with Georgia Tech's AI policy: (1) Code developed with AI assistance for boilerplate and matplotlib styling; (2) Literature research AI-assisted but manually verified; (3) Writing AI-assisted for organization and grammar; (4) Figures generated by author scripts; (5) CRDI, RDI metrics, and simulation frameworks are original contributions.

### Data and Code Availability
All code, data, and figures are available at https://github.com/Pb314314/Fund_ML, including:
- 7 Python experiment scripts (experiments/exp0--exp6) plus unified config.py
- 13 publication-quality figures (300 DPI PNG)
- Raw JSON data for all experiments
- README.md with reproduction instructions

---

*This project was completed as the term project for FunML (CS 7643 / CSE 6242), Georgia Institute of Technology, Spring 2025.*
