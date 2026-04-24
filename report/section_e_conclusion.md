## E. Conclusion

### E.1 Joint Assessment

This study demonstrates that the AI resource divide is not a hypothetical concern but a quantifiable, compounding reality. Our experiments reveal that Hindi requires 2.83 tokens per word versus 1.15 for English—imposing a 6× attention cost penalty and a 26× end-to-end latency gap for equivalent semantic tasks. The memory wall is equally stark: consumer GPUs (RTX 4060) exhaust memory at 14K tokens, while datacenter hardware (H100) sustains 544K—a 40× disparity. Aggregated across all dimensions, our Resource Disparity Index measures a 1010× effective resource access gap. Meanwhile, only 42 of approximately 7,000 world languages (0.6%) enjoy meaningful LLM support. Neither market self-correction nor isolated governance intervention alone can address this structural inequity.

### E.2 Best Practices Forward

Drawing on the dual analysis presented in the preceding section, we identify four concrete, actionable interventions. First, establish **international data commons** for low-resource languages—targeted data investment, as argued in Viewpoint B, can break compounding cycles at their origin. Second, pursue **regional compute-sharing agreements** and Sovereign AI funds to redress the geographic hardware imbalance that Viewpoint A identifies as non-self-correcting. Third, mandate **inclusive tokenizer design** as standard practice, not an afterthought, to eliminate the 6× computational penalty that low-resource languages currently bear. Fourth, institute **RDI-like indices** as longitudinal benchmarks to track progress and expose regress.

### E.3 Future Directions

We urge multilateral governance frameworks—building on UNESCO's ethical recommendations [5] and expanding the EU AI Act's accessibility principles [16]—to codify these interventions. Future work should extend RDI measurement to additional language families, evaluate the efficacy of federated compute-sharing pilots, and develop open multilingual tokenizers that explicitly optimize for parity in tokenization efficiency. The compounding nature of the divide demands dual intervention now; delay only widens the 1010× gap.
