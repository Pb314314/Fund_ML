#!/usr/bin/env python3
"""
config.py — Unified Configuration for The Resource Divide Project
===============================================================

All experiments import from this module to ensure numerical consistency.
This file is the single source of truth for:
  - Model architecture and size
  - GPU hardware specifications
  - Language fertility rates (from real tokenizer benchmark)
  - Ecosystem statistics (with sources)

Author: Student Researcher
Date: 2025-04-25
"""

from dataclasses import dataclass
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. MODEL CONFIGURATION: Llama-3-8B with 4-bit AWQ Quantization
# ---------------------------------------------------------------------------
# Source: https://huggingface.co/meta-llama/Llama-3-8B
# Quantized size from AutoAWQ / llm-awq community benchmarks.
# We use 4-bit quantization because it is the de-facto standard for
# production LLM serving (vLLM, TensorRT-LLM, TGI all default to 4-bit).

@dataclass(frozen=True)
class ModelConfig:
    name: str = "Llama-3-8B"
    params_b: float = 8.03               # Billion parameters
    num_layers: int = 32
    hidden_dim: int = 4096
    intermediate_dim: int = 14336       # FFN intermediate = 3.5 * hidden_dim
    num_attention_heads: int = 32
    num_kv_heads: int = 8               # GQA: 4 query heads share 1 KV head
    head_dim: int = 128
    bytes_per_param_quantized: float = 0.5   # 4-bit = 0.5 bytes/param
    bytes_per_param_kv: int = 2       # KV cache kept in FP16 for stability
    quantization: str = "AWQ-4bit"

    @property
    def model_weights_gb(self) -> float:
        """4-bit quantized model weights in GB."""
        return self.params_b * 1e9 * self.bytes_per_param_quantized / 1e9

    @property
    def kv_cache_per_token_bytes(self) -> int:
        """
        KV cache size per token (bytes).
        Formula: 2 (K+V) * layers * kv_heads * head_dim * dtype_size
        = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KiB = 0.125 MiB
        """
        return (2 * self.num_layers * self.num_kv_heads *
                self.head_dim * self.bytes_per_param_kv)

    @property
    def kv_cache_per_token_mb(self) -> float:
        return self.kv_cache_per_token_bytes / (1024.0 * 1024.0)

    @property
    def kv_cache_per_token_kb(self) -> float:
        return self.kv_cache_per_token_bytes / 1024.0


# Global model instance
MODEL = ModelConfig()

# ---------------------------------------------------------------------------
# 2. GPU HARDWARE SPECIFICATIONS
# ---------------------------------------------------------------------------
# Sources:
#   - H100: NVIDIA H100 SXM5 datasheet (HBM3, 80GB, 3.35 TB/s)
#     https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
#   - A100: NVIDIA A100 80GB datasheet (HBM2e, 80GB, 2.039 TB/s)
#     https://www.nvidia.com/en-us/data-center/a100/
#   - RTX 3090: NVIDIA GeForce RTX 3090 specs (GDDR6X, 24GB, 936 GB/s)
#     https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/
#   - T4: NVIDIA T4 datasheet (GDDR6, 16GB, 320 GB/s)
#     https://www.nvidia.com/en-us/data-center/tesla-t4/
#   - RTX 4060: NVIDIA RTX 4060 specs (GDDR6, 8GB, 272 GB/s)
#     https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060/
# Compute FLOPS (FP16 Tensor Core):
#   - H100 SXM5: 989 TFLOPS (with sparsity) / 495 TFLOPS (dense)
#     We use dense (conservative) for prefill compute estimates
#   - A100 80GB: 312 TFLOPS (dense)
#   - RTX 3090: 71 TFLOPS (FP16 Tensor, not RT Core)
#   - T4: 65 TFLOPS
#   - RTX 4060: 15.11 TFLOPS (FP16 Tensor)

@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float
    bandwidth_gb_s: float          # Memory bandwidth GB/s
    compute_tflops_dense: float    # FP16 dense TFLOPS (conservative)
    compute_tflops_sparse: float   # FP16 with sparsity (peak)
    power_w: float
    cost_per_hour_usd: float
    tier_label: str


GPU_SPECS: Dict[str, GPU] = {
    "H100": GPU(
        name="NVIDIA H100 SXM5",
        vram_gb=80.0,
        bandwidth_gb_s=3350.0,
        compute_tflops_dense=495.0,   # conservative: no sparsity
        compute_tflops_sparse=989.0,
        power_w=700.0,
        cost_per_hour_usd=4.0,
        tier_label="Frontier",
    ),
    "A100": GPU(
        name="NVIDIA A100 80GB PCIe/SXM",
        vram_gb=80.0,
        bandwidth_gb_s=2039.0,
        compute_tflops_dense=312.0,
        compute_tflops_sparse=624.0,
        power_w=400.0,
        cost_per_hour_usd=2.5,
        tier_label="High-end",
    ),
    "RTX3090": GPU(
        name="NVIDIA RTX 3090",
        vram_gb=24.0,
        bandwidth_gb_s=936.0,
        compute_tflops_dense=71.0,
        compute_tflops_sparse=71.0,   # no sparsity on consumer
        power_w=350.0,
        cost_per_hour_usd=0.80,
        tier_label="Workstation",
    ),
    "T4": GPU(
        name="NVIDIA T4",
        vram_gb=16.0,
        bandwidth_gb_s=320.0,
        compute_tflops_dense=65.0,
        compute_tflops_sparse=65.0,
        power_w=70.0,
        cost_per_hour_usd=0.80,
        tier_label="Commodity",
    ),
    "RTX4060": GPU(
        name="NVIDIA RTX 4060",
        vram_gb=8.0,
        bandwidth_gb_s=272.0,
        compute_tflops_dense=15.11,
        compute_tflops_sparse=15.11,
        power_w=115.0,
        cost_per_hour_usd=0.30,
        tier_label="Consumer",
    ),
    "CPU": GPU(
        name="CPU Server (2x Xeon Platinum)",
        vram_gb=128.0,               # System RAM proxy
        bandwidth_gb_s=200.0,        # DDR4-3200 quad-channel ~200 GB/s
        compute_tflops_dense=3.0,
        compute_tflops_sparse=3.0,
        power_w=400.0,
        cost_per_hour_usd=0.20,
        tier_label="CPU",
    ),
}

# ---------------------------------------------------------------------------
# 3. LANGUAGE FERTILITY (from Experiment 1 — real tokenizer benchmark)
# ---------------------------------------------------------------------------
# These values come from the Llama-3 tokenizer (meta-llama/Llama-3.2-1B).
# Fertility = tokens / word (space-separated words for Latin-script).
# For Chinese (no spaces), fertility = tokens / character ≈ 0.73.

LANGUAGE_FERTILITY: Dict[str, float] = {
    "English": 1.15,
    "Chinese": 0.73,
    "Spanish": 1.68,
    "Arabic": 2.40,
    "Swahili": 2.34,
    "Hindi": 2.83,
}

# ---------------------------------------------------------------------------
# 4. LANGUAGE ECOSYSTEM STATISTICS (with sources)
# ---------------------------------------------------------------------------
# Sources:
#   - Speaker data: Ethnologue 2024 (L1+L2 speakers, millions)
#   - Internet content %: W3Techs / Intelpoint 2024 survey of top 10M websites
#     https://w3techs.com/technologies/overview/content_language
#   - Digital literacy proxy: ITU Digital Development Report 2024
#     Composite proxy: % of population with basic digital skills
#   - LLM MMLU accuracy: Approximated from multilingual LLM benchmarks
#     (Ahuja et al. ACL 2024; Masakhane papers)
#   - Wikipedia users: Wikimedia Foundation stats, active editors 2024

@dataclass
class LanguageEcosystem:
    speakers_millions: float
    internet_content_pct: float
    digital_literacy_proxy: float
    llm_mmlu_accuracy: float
    wikipedia_users: int
    resource_tier: str
    sources: List[str]


LANGUAGE_ECOSYSTEM: Dict[str, LanguageEcosystem] = {
    "English": LanguageEcosystem(
        speakers_millions=1500.0,
        internet_content_pct=49.7,
        digital_literacy_proxy=92.0,
        llm_mmlu_accuracy=85.0,
        wikipedia_users=122038,
        resource_tier="High",
        sources=["Ethnologue 2024", "W3Techs 2024", "Wikimedia 2024"],
    ),
    "Chinese": LanguageEcosystem(
        speakers_millions=1200.0,
        internet_content_pct=19.04,
        digital_literacy_proxy=82.0,
        llm_mmlu_accuracy=78.0,
        wikipedia_users=6994,
        resource_tier="High",
        sources=["Ethnologue 2024", "W3Techs 2024", "Wikimedia 2024"],
    ),
    "Spanish": LanguageEcosystem(
        speakers_millions=600.0,
        internet_content_pct=7.70,
        digital_literacy_proxy=75.0,
        llm_mmlu_accuracy=72.0,
        wikipedia_users=14385,
        resource_tier="High",
        sources=["Ethnologue 2024", "W3Techs 2024"],
    ),
    "Arabic": LanguageEcosystem(
        speakers_millions=420.0,
        internet_content_pct=3.65,
        digital_literacy_proxy=68.0,
        llm_mmlu_accuracy=65.0,
        wikipedia_users=3930,
        resource_tier="Moderate",
        sources=["Ethnologue 2024", "W3Techs 2024"],
    ),
    "Hindi": LanguageEcosystem(
        speakers_millions=600.0,
        internet_content_pct=3.77,
        digital_literacy_proxy=48.0,   # India's digital literacy varies; 48% is rural+urban blended
        llm_mmlu_accuracy=55.0,
        wikipedia_users=0,
        resource_tier="Moderate",
        sources=["Ethnologue 2024", "W3Techs 2024", "ITU India Report 2024"],
    ),
    "Swahili": LanguageEcosystem(
        speakers_millions=200.0,
        internet_content_pct=0.0025,
        digital_literacy_proxy=38.0,   # Sub-Saharan Africa average
        llm_mmlu_accuracy=45.0,
        wikipedia_users=0,
        resource_tier="Low",
        sources=["Ethnologue 2024", "W3Techs 2024", "ITU Africa Report 2024"],
    ),
    "Bengali": LanguageEcosystem(
        speakers_millions=270.0,
        internet_content_pct=0.20,
        digital_literacy_proxy=45.0,
        llm_mmlu_accuracy=50.0,
        wikipedia_users=0,
        resource_tier="Low",
        sources=["Ethnologue 2024", "W3Techs 2024"],
    ),
    "Russian": LanguageEcosystem(
        speakers_millions=260.0,
        internet_content_pct=3.75,
        digital_literacy_proxy=69.0,
        llm_mmlu_accuracy=68.0,
        wikipedia_users=9243,
        resource_tier="Moderate",
        sources=["Ethnologue 2024", "W3Techs 2024"],
    ),
    "Japanese": LanguageEcosystem(
        speakers_millions=125.0,
        internet_content_pct=2.23,
        digital_literacy_proxy=93.0,
        llm_mmlu_accuracy=75.0,
        wikipedia_users=12409,
        resource_tier="High",
        sources=["Ethnologue 2024", "W3Techs 2024"],
    ),
    "French": LanguageEcosystem(
        speakers_millions=300.0,
        internet_content_pct=3.42,
        digital_literacy_proxy=78.0,
        llm_mmlu_accuracy=74.0,
        wikipedia_users=17717,
        resource_tier="High",
        sources=["Ethnologue 2024", "W3Techs 2024"],
    ),
}

# ---------------------------------------------------------------------------
# 5. GLOBAL COMPUTE DISTRIBUTION
# ---------------------------------------------------------------------------
# Source: Tony Blair Institute "State of Compute Access" 2024
#         GovAI "Computing Power and the Governance of AI" 2024

COMPUTE_DISTRIBUTION = {
    "United States": 74.0,
    "China": 14.0,
    "European Union": 5.0,
    "Other": 7.0,
}

# ---------------------------------------------------------------------------
# 6. LANGUAGE SUPPORT GAP DATA
# ---------------------------------------------------------------------------
# Sources:
#   - 7,000 world languages: Ethnologue 2024 (living languages)
#   - 100 with NLP: Joshi et al. "State of NLP for Low-Resource Languages" 2020
#   - 42 African LLM: Adebara et al. "State of LLMs for African Languages" 2025
#   - Masakhane: https://www.masakhane.io/

LANGUAGE_SUPPORT_GAP = {
    "total_world_languages": 7151,       # Ethnologue 2024
    "languages_with_nlp_resources": 100,  # Joshi et al. 2020
    "languages_with_llm_training_data": 42,  # Adebara et al. 2025 (African)
    "african_languages_with_any_nlp": 55,
}

# ---------------------------------------------------------------------------
# 7. SIMULATION CONSTANTS
# ---------------------------------------------------------------------------
MEMORY_OVERHEAD_PCT = 0.10    # 10% activation/workspace overhead
BANDWIDTH_UTILIZATION = 0.70  # Effective HBM bandwidth utilization for inference
COMPUTE_UTILIZATION = 0.85   # Effective compute utilization for prefill


def available_kv_memory_gb(vram_gb: float, model_weights_gb: float, overhead_pct: float = MEMORY_OVERHEAD_PCT) -> float:
    """Calculate available memory for KV cache after model weights and overhead."""
    overhead_gb = vram_gb * overhead_pct
    available = vram_gb - model_weights_gb - overhead_gb
    return max(0.0, available)


def kv_cache_size_gb(seq_len: int, batch_size: int, kv_per_token_mb: float) -> float:
    """Total KV cache memory in GB."""
    total_mb = seq_len * batch_size * kv_per_token_mb
    return total_mb / 1024.0
