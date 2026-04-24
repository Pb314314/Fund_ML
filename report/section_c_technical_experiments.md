## Section C: Technical Experiments

### C.1 Experimental Design

We design three complementary experiments to quantify the resource divide across the NLP pipeline. First, we measure tokenization fertility using the Llama-3 tokenizer on six languages spanning five scripts. Second, we simulate the memory wall by computing KV-cache consumption for Llama-3-8B under 4-bit AWQ quantization (~5.5 GB weights) across five hardware tiers. Third, we benchmark end-to-end latency for a controlled 100-word semantic task. Metrics include fertility (tokens per word), attention-cost multiplier (fertility²), maximum sequence length before out-of-memory (OOM), and a composite Resource Divide Index: RDI = fertility² / (bandwidth_GB/s × VRAM_GB).

### C.2 Tokenization Fertility Results

Tokenization fertility—the ratio of subword tokens to source words—determines compute consumption per unit of semantic content. Figure 1 presents fertility and its squared attention-cost multiplier for six languages using the Llama-3 tokenizer. English (Latin script) requires 1.15 tokens per word, yielding a multiplier of 1.32. Chinese (Han script) achieves the most efficient encoding at 0.73 tokens per word (multiplier 0.53). Hindi (Devanagari) demands 2.83 tokens per word, producing a multiplier of 7.99. Swahili and Arabic fall at intermediate-high positions with multipliers of 5.47 and 5.76, respectively. The disparity between Hindi and English is striking: for the same semantic content, Hindi incurs ~6× the attention cost, meaning each self-attention layer performs roughly six times more floating-point operations per word. This overhead is an architectural artifact of how byte-pair encoding splits non-Latin grapheme sequences—not a user-level choice.

![Fig. 1: Tokenization fertility and attention-cost multiplier across six languages.](/mnt/agents/output/project/figures/fig1_tokenization_fertility.png)

### C.3 Memory Wall Simulation Results

Even when tokenization overhead is held constant, hardware constraints create a second tier of inequity. Figure 2 reports KV-cache capacity and maximum sustainable sequence length for Llama-3-8B (4-bit AWQ) on five hardware configurations. At 0.125 MB of FP16 KV cache per token, an H100 or A100 (80 GB VRAM) retains 66.5 GB available after weight loading, supporting 544,768 tokens at batch size 1 or 17,024 at batch size 32. A consumer RTX 4060 (8 GB) is left with only 1.7 GB, capping single-user contexts at 13,926 tokens and collapsing to 435 under batch size 32. The frontier-to-consumer gap spans nearly 40× in single-sequence capacity. For multilingual deployments where fertility inflates token counts, the RTX 4060 threshold of ~435 tokens under batching is below the length of many real-world documents. A Swahili or Hindi user on consumer hardware thus faces double jeopardy: more tokens per word and fewer tokens total before OOM.

![Fig. 2: KV-cache capacity and maximum sequence length across five hardware tiers for Llama-3-8B 4-bit AWQ.](/mnt/agents/output/project/figures/fig2a_kv_cache_growth.png)

### C.4 Combined End-to-End Analysis

Figure 3 integrates tokenization overhead with hardware latency for the same 100-word semantic task. On an H100, English (115 tokens) completes in 0.7 s, Swahili (234 tokens) in 1.5 s, and Hindi (283 tokens) in 1.8 s. On a T4, the same tasks take 7.6 s, 15.6 s, and 18.8 s, respectively. The compounded disadvantage is severe: Hindi on a T4 is 26.9× slower than English on an H100. On CPU-only inference, the gap widens further—Hindi requires 119.8 s versus English at 48.5 s. We summarize cumulative inequity via the Resource Divide Index: RDI = fertility² / (memory bandwidth in GB/s × VRAM in GB). Comparing extreme endpoints, Hindi on CPU-only versus English on H100 yields an RDI ratio of 1,010×—more than three orders of magnitude in effective resource access per unit of semantic work. This metric shows that language equity cannot be solved by translation or post-processing alone; the inequity is baked into token counts, memory geometry, and hardware availability at every layer of the stack.

![Fig. 3: End-to-end latency comparison for a 100-word semantic task across languages and hardware tiers.](/mnt/agents/output/project/figures/fig3a_latency_comparison.png)
