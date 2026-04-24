# Paper Outline: The Resource Divide — IEEE Conference Format

## Title
The Resource Divide: Data Scarcity, Compute Inequity, and the Future of Global LLM Governance

## Abstract (~150 words)
- Background: AI divide as socio-technical risk
- Problem: compute and data disparities create compounding performance gaps
- Method: tokenization benchmark + memory-wall simulation across 5 hardware tiers
- Key results: Swahili 2.0× more tokens than English; Hindi on T4 is 26× slower than English on H100 for same semantic task; consumer GPUs OOM at ~14K tokens
- Significance: demonstrates structural barriers requiring policy intervention

## Section A: Introduction (~500 words, ~15%)
### A.1 The AI Divide as Global Risk
- Rapid LLM proliferation centralizes power
- UNESCO, EU AI Act, CHIPS Act frameworks
- 74% US compute share; 160 nations as "compute deserts"

### A.2 Scope and Objectives
- Investigate how compute + data disparities manifest as performance gaps
- Quantify tokenization tax and memory wall
- Dual-viewpoint analysis of governance vs market solutions

### A.3 Contribution Statement
- First empirical quantification of compounding resource divide
- Reproducible simulation framework for hardware-software co-design inequity

## Section B: Literature Review (~700 words, ~20%)
### B.1 Geopolitics of Compute
- GovAI: compute as excludable, detectable regulation lever
- Lehdonvirta et al.: "computational sovereignty" concept
- CHIPS Act and export controls as structural barriers

### B.2 Linguistic Inequity: The Tokenization Tax
- Lundin et al.: fertility predicts 8-18pp accuracy drop per +1 token
- Ahuja et al.: cost-performance optimization for low-resource languages
- O(n²) attention scaling amplifies disparities

### B.3 Data Imbalance and Model Bias
- 2,000+ African languages; ~42 with LLM support
- Swahili: 200M speakers, 700M tokens (vs English trillions)
- Cyclic disadvantage: less data → worse models → no adoption → no investment

### B.4 The Memory Wall
- Hooper et al.: KVQuant and long-context challenges
- HBM3e vs GDDR6: 10× bandwidth gap
- KV cache linear scaling creates hard ceiling

## Section C: Technical Experiments (~800 words, ~25%)
### C.1 Experimental Design
- Tokenization: Llama-3 tokenizer, 6 languages
- Memory simulation: Llama-3-8B 4-bit, 5 hardware tiers
- Metrics: fertility, attention cost, latency, OOM thresholds

### C.2 Tokenization Fertility Results
- English: 1.15 tok/word; Swahili: 2.34; Hindi: 2.83
- Attention cost multiplier: Hindi 8× vs Chinese 0.5× vs English
- Fig. 1: Fertility and attention cost by language

### C.3 Memory Wall Simulation Results
- KV cache formula: 0.125 MB/token for Llama-3-8B
- Consumer GPU (RTX 4060, 8GB): OOM at ~14K tokens (batch=1)
- Frontier GPU (H100, 80GB): sustains 544K tokens (batch=1)
- Fig. 2: KV cache growth and throughput degradation

### C.4 Combined End-to-End Analysis
- Same 100-word task: Hindi on T4 = 26× slower than English on H100
- Resource Divide Index (RDI): fertility² / (bandwidth × VRAM)
- Fig. 3: Latency comparison and cumulative disadvantage

## Section D: Dual Viewpoint Analysis (~700 words, ~20%)
### D.1 Viewpoint A (Bo Pang): Structural Governance Imperative
- Compute geography is structural, not market-correctable
- Memory bandwidth bottleneck: GDDR6 cannot self-upgrade to HBM3e
- Tokenization tax is embedded in tokenizer training data
- Policy recommendations: Sovereign AI funding, Compute Commons, multilateral governance

### D.2 Viewpoint B (Connor Sempf): Data-Driven Self-Correction
- Data inequality is root cause; compute alone cannot fix
- Without data, no amount of compute bridges gap
- Market incentives:demand in Global South will drive data collection
- Technology: vocabulary extension, multilingual pre-training reduce fertility
- Cyclic consequences can be broken by targeted data investment

### D.3 Synthesis
- Both viewpoints valid; compounding nature requires dual intervention
- Fig. 4: Language support gap pyramid

## Section E: Conclusion (~300 words, ~10%)
### E.1 Joint Assessment
- AI divide is real, quantifiable, and compounding
- Neither governance nor markets alone suffice

### E.2 Best Practices Forward
- Data commons + compute sharing agreements
- Inclusive tokenizer design as standard practice
- Regional capacity building with metrics tracking

## References
- 15-20 IEEE-format references

## Appendix: Compute and AI Usage
- AI Makerspace hours: [to be disclosed]
- GPU nodes used: [to be disclosed]
- AI assistance disclosure: [to be disclosed]
