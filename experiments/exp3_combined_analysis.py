#!/usr/bin/env python3
"""
Experiment 3: Combined Resource Divide Analysis
=================================================
Combines tokenization data (Experiment 1) with memory/compute simulation
(Experiment 2) to demonstrate the COMPOUNDING effect of compute inequity
and data scarcity on LLM inference latency.

Author: Machine Learning Systems Engineer
Date: 2025
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# PUBLICATION-QUALITY MATPLOTLIB SETTINGS
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/mnt/agents/output/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

for d in [DATA_DIR, FIGURES_DIR, EXPERIMENTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# HARDWARE SPECIFICATIONS (representing global compute inequality)
# ---------------------------------------------------------------------------
# Values sourced from NVIDIA datasheets and peer-reviewed ML systems papers.
# Compute FLOPS in FP16/BF16 (the precision used for LLM inference).
# Memory bandwidth in GB/s. VRAM in GB.
HARDWARE_TIERS = {
    "H100_SXM5": {
        "name": "NVIDIA H100 SXM5",
        "compute_tflops": 989.0,      # FP16 Tensor Core peak
        "memory_bw_gbps": 3350.0,     # HBM3 bandwidth
        "vram_gb": 80.0,
        "typical_access": "Cloud / Research lab / Tech company",
        "cost_per_hour_usd": 4.0,     # Approx. cloud rental
    },
    "A100_80GB": {
        "name": "NVIDIA A100 80GB",
        "compute_tflops": 312.0,      # FP16 Tensor Core peak
        "memory_bw_gbps": 2039.0,     # HBM2e bandwidth
        "vram_gb": 80.0,
        "typical_access": "University / Mid-tier cloud",
        "cost_per_hour_usd": 2.5,
    },
    "T4": {
        "name": "NVIDIA T4",
        "compute_tflops": 65.0,       # FP16 peak
        "memory_bw_gbps": 320.0,      # GDDR6 bandwidth
        "vram_gb": 16.0,
        "typical_access": "Entry cloud / Edge server / Africa regional",
        "cost_per_hour_usd": 0.80,
    },
    "CPU_Server": {
        "name": "CPU-only Server (2x Xeon)",
        "compute_tflops": 3.0,        # ~AVX-512 FP16 throughput
        "memory_bw_gbps": 50.0,       # DDR4-3200 dual channel
        "vram_gb": 32.0,              # System RAM used as proxy
        "typical_access": "On-premise / Low-resource institution",
        "cost_per_hour_usd": 0.20,
    },
}

# ---------------------------------------------------------------------------
# MODEL SPECIFICATIONS (Llama-3-8B class architecture)
# ---------------------------------------------------------------------------
MODEL_SPEC = {
    "name": "Llama-3-8B",
    "params_b": 8.03,               # Billion parameters
    "num_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_kv_heads": 8,              # GQA
    "head_dim": 128,
    "bytes_per_param": 2,          # FP16
}

# Derived model constants
MODEL_SIZE_BYTES = MODEL_SPEC["params_b"] * 1e9 * MODEL_SPEC["bytes_per_param"]
# KV cache per token: 2 (K+V) * num_layers * hidden_size * bytes_per_param
# With GQA, we use num_kv_heads * head_dim instead of hidden_size for K/V
KV_CACHE_PER_TOKEN_BYTES = (
    2
    * MODEL_SPEC["num_layers"]
    * MODEL_SPEC["num_kv_heads"]
    * MODEL_SPEC["head_dim"]
    * MODEL_SPEC["bytes_per_param"]
)

# ---------------------------------------------------------------------------
# TASK DEFINITION: Fixed semantic content (answering a 100-word question)
# ---------------------------------------------------------------------------
WORDS_PER_TASK = 100
# English answer length in tokens (empirically ~130 tokens for 100 words)
OUTPUT_TOKENS_EN = 130

# ---------------------------------------------------------------------------
# LANGUAGE SUPPORT DATA (for figure 3d)
# ---------------------------------------------------------------------------
LANGUAGE_SUPPORT_DATA = {
    "total_world_languages": 7000,
    "languages_with_nlp_support": 100,
    "languages_with_llm_support": 42,      # African languages with any LLM data
    "african_languages_with_nlp": 55,
    "digital_content_pct": {
        "English": 54.3,
        "Russian": 5.9,
        "German": 5.7,
        "Spanish": 4.6,
        "French": 4.0,
        "Japanese": 3.3,
        "Chinese": 2.8,
        "Portuguese": 2.5,
        "Italian": 1.7,
        "Polish": 1.5,
        "Turkish": 1.3,
        "Dutch": 1.2,
        "Arabic": 1.0,
        "Korean": 0.8,
        "Swahili": 0.002,
        "Hindi": 0.15,
    }
}


# ===========================================================================
# 1. LOAD TOKENIZATION DATA
# ===========================================================================

def load_tokenization_data(path: str) -> Dict[str, Any]:
    """Load fertility rates from Experiment 1."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["languages"]


# ===========================================================================
# 2. LATENCY MODELING
# ===========================================================================

def compute_latency(
    input_tokens: int,
    output_tokens: int,
    hardware: Dict[str, float],
    model_bytes: float,
    kv_cache_per_tok: float,
) -> Dict[str, float]:
    """
    Compute end-to-end latency for a single generation request.

    Prefill phase (TTFT):
        TTFT = (2 * P * N_input) / compute_flops
        Factor of 2 comes from multiply-accumulate in transformer forward pass.

    Decode phase (TPOT):
        TPOT = model_size_bytes / memory_bandwidth
        In decode, each token requires a full weight memory read (memory-bound).

    KV-cache read overhead during decode:
        As context grows, KV cache must also be read from memory each step.
        kv_read_time = (context_length * kv_cache_per_token) / memory_bw

    Total = TTFT + sum_over_output_tokens(TPOT + kv_read_for_that_step)
    """
    compute_flops = hardware["compute_tflops"] * 1e12
    mem_bw = hardware["memory_bw_gbps"] * 1e9
    vram = hardware["vram_gb"] * 1e9

    # Prefill: ~2 * params * input_tokens FLOPs
    prefill_flops = 2 * MODEL_SPEC["params_b"] * 1e9 * input_tokens
    ttft = prefill_flops / compute_flops

    # Decode: memory-bound weight loading + KV cache loading
    tpot_weights = model_bytes / mem_bw

    total_decode_time = 0.0
    max_kv_cache = 0.0
    for step in range(1, output_tokens + 1):
        context_len = input_tokens + step - 1  # KV cache accumulated so far
        kv_cache_bytes = context_len * kv_cache_per_tok
        max_kv_cache = max(max_kv_cache, kv_cache_bytes)

        # Check OOM
        total_mem = model_bytes + kv_cache_bytes
        if total_mem > vram:
            # Swapping or CPU offload; penalty factor
            tpot_kv = (kv_cache_bytes / mem_bw) * 5.0  # 5x penalty for paging
        else:
            tpot_kv = kv_cache_bytes / mem_bw

        total_decode_time += tpot_weights + tpot_kv

    total_latency = ttft + total_decode_time

    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot_weights * 1000,
        "total_decode_ms": total_decode_time * 1000,
        "total_latency_ms": total_latency * 1000,
        "max_kv_cache_mb": max_kv_cache / (1024 * 1024),
        "oom_risk": (model_bytes + max_kv_cache) > vram,
    }


# ===========================================================================
# 3. RESOURCE DIVIDE INDEX
# ===========================================================================

def compute_rdi(
    fertility: float,
    hardware: Dict[str, float],
) -> float:
    """
    Resource Divide Index (RDI):
        RDI = (fertility^2) * (1 / memory_bw_GBps) * (1 / vram_GB)

    Higher RDI = worse resource access (more tokens, slower memory, less VRAM).
    We normalize against English/H100 baseline so RDI_baseline = 1.0.
    """
    mem_bw_gb = hardware["memory_bw_gbps"]
    vram_gb = hardware["vram_gb"]
    return (fertility ** 2) * (1.0 / mem_bw_gb) * (1.0 / vram_gb)


# ===========================================================================
# 4. MULTI-TURN AGENTIC CONVERSATION SIMULATION
# ===========================================================================

def simulate_multi_turn(
    language: str,
    fertility: float,
    hardware: Dict[str, float],
    num_turns: int = 5,
) -> List[Dict[str, float]]:
    """
    Simulate a multi-turn agentic conversation where KV cache accumulates.

    Each turn:
        - User asks a 30-word question
        - Model generates a 50-word response
        - All previous context is retained (growing KV cache)
    """
    compute_flops = hardware["compute_tflops"] * 1e12
    mem_bw = hardware["memory_bw_gbps"] * 1e9
    vram = hardware["vram_gb"] * 1e9

    # Tokens per turn (scaled by fertility)
    user_tokens_per_turn = int(30 * fertility)
    response_tokens_per_turn = int(50 * fertility)

    results = []
    accumulated_context_tokens = 0

    for turn in range(1, num_turns + 1):
        # Input = user prompt + all previous context
        input_tokens = user_tokens_per_turn + accumulated_context_tokens
        output_tokens = response_tokens_per_turn

        # Prefill: process full context each turn
        prefill_flops = 2 * MODEL_SPEC["params_b"] * 1e9 * input_tokens
        ttft = prefill_flops / compute_flops

        # Decode with accumulated KV cache
        tpot_weights = MODEL_SIZE_BYTES / mem_bw
        decode_time = 0.0
        max_kv = 0.0

        for step in range(1, output_tokens + 1):
            kv_len = input_tokens + step - 1
            kv_bytes = kv_len * KV_CACHE_PER_TOKEN_BYTES
            max_kv = max(max_kv, kv_bytes)

            total_mem = MODEL_SIZE_BYTES + kv_bytes
            if total_mem > vram:
                tpot_kv = (kv_bytes / mem_bw) * 5.0
            else:
                tpot_kv = kv_bytes / mem_bw

            decode_time += tpot_weights + tpot_kv

        turn_latency = ttft + decode_time

        results.append({
            "turn": turn,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft * 1000,
            "decode_ms": decode_time * 1000,
            "total_ms": turn_latency * 1000,
            "kv_cache_mb": max_kv / (1024 * 1024),
            "oom_risk": (MODEL_SIZE_BYTES + max_kv) > vram,
        })

        # Accumulate for next turn
        accumulated_context_tokens += user_tokens_per_turn + response_tokens_per_turn

    return results


# ===========================================================================
# 5. FIGURE GENERATION
# ===========================================================================

def plot_latency_comparison(
    languages: Dict[str, Any],
    hardware: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Figure 3a: Grouped bar chart comparing English vs Swahili vs Hindi
    latency on each hardware tier for the same semantic task.
    """
    target_langs = ["English", "Swahili", "Hindi"]
    lang_colors = {"English": "#2E86AB", "Swahili": "#A23B72", "Hindi": "#F18F01"}
    lang_labels = {
        "English": "English (1.15 tok/word)",
        "Swahili": "Swahili (2.34 tok/word)",
        "Hindi": "Hindi (2.83 tok/word)",
    }

    # Compute tokens for 100-word semantic task
    tokens_map = {
        lang: int(WORDS_PER_TASK * languages[lang]["fertility"])
        for lang in target_langs
    }

    hw_names = list(hardware.keys())
    x = np.arange(len(hw_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, lang in enumerate(target_langs):
        latencies = []
        for hw_key in hw_names:
            hw = hardware[hw_key]
            lat = compute_latency(
                input_tokens=tokens_map[lang],
                output_tokens=int(OUTPUT_TOKENS_EN * languages[lang]["fertility"]),
                hardware=hw,
                model_bytes=MODEL_SIZE_BYTES,
                kv_cache_per_tok=KV_CACHE_PER_TOKEN_BYTES,
            )
            latencies.append(lat["total_latency_ms"] / 1000.0)  # seconds

        offset = (i - 1) * width
        bars = ax.bar(x + offset, latencies, width, label=lang_labels[lang],
                      color=lang_colors[lang], edgecolor="white", linewidth=0.5)

        # Annotate values
        for bar, val in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xlabel("Hardware Tier", fontweight="bold")
    ax.set_ylabel("End-to-End Latency (seconds)", fontweight="bold")
    ax.set_title(
        "Figure 3a: Latency Comparison for 100-Word Semantic Task\n"
        "(Same meaning, different token counts due to fertility disparity)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [hardware[k]["name"] for k in hw_names], rotation=15, ha="right"
    )
    ax.legend(title="Language", loc="upper left", frameon=True)
    ax.set_ylim(0, None)

    # Add annotation box explaining compounding
    ax.annotate(
        "Compounding effect:\n"
        "Hindi needs 2.5x more tokens than English,\n"
        "and T4 has 10x less bandwidth than H100.",
        xy=(2.5, ax.get_ylim()[1] * 0.6),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Saved {output_path}")


def plot_resource_divide_index(
    languages: Dict[str, Any],
    hardware: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Figure 3b: Resource Divide Index heatmap.
    RDI = fertility^2 / (memory_bw * vram)
    Normalized to English/H100 baseline = 1.0.
    """
    target_langs = list(languages.keys())
    hw_names = list(hardware.keys())

    # Compute raw RDI matrix
    rdi_matrix = np.zeros((len(target_langs), len(hw_names)))
    for i, lang in enumerate(target_langs):
        for j, hw_key in enumerate(hw_names):
            rdi_matrix[i, j] = compute_rdi(
                languages[lang]["fertility"],
                hardware[hw_key],
            )

    # Normalize: English on H100 = 1.0
    baseline_rdi = rdi_matrix[0, 0]  # English, H100
    rdi_norm = rdi_matrix / baseline_rdi

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(rdi_norm, cmap="YlOrRd", aspect="auto", vmin=1.0)

    # X ticks: hardware
    ax.set_xticks(np.arange(len(hw_names)))
    ax.set_xticklabels(
        [hardware[k]["name"] for k in hw_names], rotation=30, ha="right"
    )

    # Y ticks: languages
    ax.set_yticks(np.arange(len(target_langs)))
    ax.set_yticklabels(target_langs)

    # Annotate cells
    for i in range(len(target_langs)):
        for j in range(len(hw_names)):
            val = rdi_norm[i, j]
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.1f}x",
                    ha="center", va="center", color=text_color, fontsize=10,
                    fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Resource Divide Index (normalized to English/H100 = 1.0)",
                   rotation=270, labelpad=25, fontweight="bold")

    ax.set_xlabel("Hardware Tier", fontweight="bold")
    ax.set_ylabel("Language", fontweight="bold")
    ax.set_title(
        "Figure 3b: Resource Divide Index Heatmap\n"
        "RDI = fertility² / (memory_bw × vram)  —  higher = worse access",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Saved {output_path}")


def plot_cumulative_disadvantage(
    languages: Dict[str, Any],
    hardware: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Figure 3c: Multi-turn agentic conversation showing KV cache accumulation.
    Compare English on H100 vs Swahili on T4.
    """
    # Two scenarios: English/H100 (privileged) vs Swahili/T4 (disadvantaged)
    scenario_a = ("English", "H100_SXM5")
    scenario_b = ("Swahili", "T4")

    scenarios = [scenario_a, scenario_b]
    colors = ["#2E86AB", "#A23B72"]
    linestyles = ["-", "--"]
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: Latency per turn ---
    ax1 = axes[0]
    for (lang, hw_key), color, ls, marker in zip(scenarios, colors, linestyles, markers):
        hw = hardware[hw_key]
        fert = languages[lang]["fertility"]
        results = simulate_multi_turn(lang, fert, hw, num_turns=5)

        turns = [r["turn"] for r in results]
        latencies = [r["total_ms"] / 1000.0 for r in results]

        ax1.plot(turns, latencies, color=color, linestyle=ls, marker=marker,
                 linewidth=2.5, markersize=8,
                 label=f"{lang} on {hw['name']}")

        # Annotate final point
        ax1.annotate(f"{latencies[-1]:.1f}s",
                     xy=(turns[-1], latencies[-1]),
                     xytext=(turns[-1] + 0.2, latencies[-1] + 2),
                     fontsize=9, fontweight="bold",
                     color=color)

    ax1.set_xlabel("Conversation Turn", fontweight="bold")
    ax1.set_ylabel("Total Turn Latency (seconds)", fontweight="bold")
    ax1.set_title("Latency Per Turn (Growing KV Cache)", fontweight="bold")
    ax1.set_xticks(range(1, 6))
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_ylim(0, None)

    # --- Right panel: Cumulative time + KV cache size ---
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    for (lang, hw_key), color, ls, marker in zip(scenarios, colors, linestyles, markers):
        hw = hardware[hw_key]
        fert = languages[lang]["fertility"]
        results = simulate_multi_turn(lang, fert, hw, num_turns=5)

        turns = [r["turn"] for r in results]
        cumulative_time = np.cumsum([r["total_ms"] / 1000.0 for r in results])
        kv_sizes = [r["kv_cache_mb"] for r in results]

        # Cumulative time on left axis
        ax2.plot(turns, cumulative_time, color=color, linestyle=ls, marker=marker,
                 linewidth=2.5, markersize=8, label=f"{lang} cumulative")
        # KV cache on right axis (lighter shade)
        ax2_twin.plot(turns, kv_sizes, color=color, linestyle=":", marker="D",
                      markersize=6, alpha=0.6, label=f"{lang} KV cache")

    ax2.set_xlabel("Conversation Turn", fontweight="bold")
    ax2.set_ylabel("Cumulative Time (seconds)", fontweight="bold", color="#333")
    ax2_twin.set_ylabel("KV Cache Size (MB)", fontweight="bold", color="#666")
    ax2.set_title("Cumulative Delay & Memory Pressure", fontweight="bold")
    ax2.set_xticks(range(1, 6))

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=8)

    fig.suptitle(
        "Figure 3c: Cumulative Disadvantage Trajectory\n"
        "English on H100 vs Swahili on T4 over 5-turn agentic conversation",
        fontweight="bold", fontsize=13, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Saved {output_path}")


def plot_language_support_gap(output_path: str) -> None:
    """
    Figure 3d: Language support gap and digital content volume.
    Two subplots: (a) language counts, (b) digital content by language.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Language support pyramid ---
    ax1 = axes[0]

    categories = [
        "World Languages\n(~7,000)",
        "With NLP\nResources (~100)",
        "With LLM\nTraining Data (~42)",
    ]
    counts = [
        LANGUAGE_SUPPORT_DATA["total_world_languages"],
        LANGUAGE_SUPPORT_DATA["languages_with_nlp_support"],
        LANGUAGE_SUPPORT_DATA["languages_with_llm_support"],
    ]
    colors_pyramid = ["#E8E8E8", "#B8D4E3", "#2E86AB"]

    bars = ax1.barh(categories, counts, color=colors_pyramid, edgecolor="black",
                    height=0.6)
    ax1.invert_yaxis()

    # Annotate bars
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax1.text(width + 100, bar.get_y() + bar.get_height()/2,
                 f"{count:,}", va="center", ha="left", fontweight="bold",
                 fontsize=11)

    # Add ratio annotations
    ax1.text(3500, 0.5, "99.4% unsupported", ha="center", fontsize=10,
             style="italic", color="#555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    ax1.text(200, 1.5, "58% gap\nfrom NLP to LLM", ha="center", fontsize=9,
             style="italic", color="#555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax1.set_xlabel("Number of Languages", fontweight="bold")
    ax1.set_title("(a) Language Support Pyramid", fontweight="bold")
    ax1.set_xlim(0, 8000)

    # --- Right: Digital content volume ---
    ax2 = axes[1]

    content_data = LANGUAGE_SUPPORT_DATA["digital_content_pct"]
    # Sort descending
    sorted_content = sorted(content_data.items(), key=lambda x: x[1], reverse=True)
    langs = [item[0] for item in sorted_content]
    pcts = [item[1] for item in sorted_content]

    # Highlight English, Swahili, Hindi
    bar_colors = []
    for lang in langs:
        if lang == "English":
            bar_colors.append("#2E86AB")
        elif lang == "Swahili":
            bar_colors.append("#A23B72")
        elif lang == "Hindi":
            bar_colors.append("#F18F01")
        else:
            bar_colors.append("#CCCCCC")

    bars = ax2.barh(langs, pcts, color=bar_colors, edgecolor="black", height=0.6)
    ax2.invert_yaxis()

    # Annotate top bars
    for bar, pct in zip(bars, pcts):
        width = bar.get_width()
        if pct > 1.0:
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{pct:.1f}%", va="center", ha="left", fontsize=9)
        else:
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{pct:.3f}%", va="center", ha="left", fontsize=8)

    ax2.set_xlabel("Share of Digital Content (%)", fontweight="bold")
    ax2.set_title("(b) Digital Content Volume by Language", fontweight="bold")
    ax2.set_xlim(0, 65)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="English (high-resource)"),
        Patch(facecolor="#A23B72", label="Swahili (low-resource)"),
        Patch(facecolor="#F18F01", label="Hindi (mid-resource)"),
        Patch(facecolor="#CCCCCC", label="Other"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(
        "Figure 3d: The Language Support Gap\n"
        "Structural exclusion in the digital linguistic ecosystem",
        fontweight="bold", fontsize=13, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Saved {output_path}")


# ===========================================================================
# 6. MAIN
# ===========================================================================

def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 3: Combined Resource Divide Analysis")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load tokenization data
    # ------------------------------------------------------------------
    tokenization_path = os.path.join(DATA_DIR, "tokenization_results.json")
    print(f"\n[1] Loading tokenization data from {tokenization_path}")
    languages = load_tokenization_data(tokenization_path)
    print(f"    Loaded {len(languages)} languages: {list(languages.keys())}")

    # ------------------------------------------------------------------
    # Compute latency comparison table
    # ------------------------------------------------------------------
    print("\n[2] Computing latency comparison for 100-word semantic task")
    target_langs = ["English", "Swahili", "Hindi"]
    latency_results = {}

    for lang in target_langs:
        fertility = languages[lang]["fertility"]
        input_tokens = int(WORDS_PER_TASK * fertility)
        output_tokens = int(OUTPUT_TOKENS_EN * fertility)
        latency_results[lang] = {}

        print(f"\n    {lang}: fertility={fertility:.2f}, "
              f"input_tokens={input_tokens}, output_tokens={output_tokens}")

        for hw_key, hw in HARDWARE_TIERS.items():
            lat = compute_latency(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                hardware=hw,
                model_bytes=MODEL_SIZE_BYTES,
                kv_cache_per_tok=KV_CACHE_PER_TOKEN_BYTES,
            )
            latency_results[lang][hw_key] = lat
            print(f"      {hw['name']:25s}: "
                  f"TTFT={lat['ttft_ms']:8.1f}ms, "
                  f"TPOT={lat['tpot_ms']:8.2f}ms, "
                  f"Total={lat['total_latency_ms']/1000:8.2f}s "
                  f"{'[OOM RISK]' if lat['oom_risk'] else ''}")

    # ------------------------------------------------------------------
    # Compute Resource Divide Index
    # ------------------------------------------------------------------
    print("\n[3] Computing Resource Divide Index (RDI)")
    rdi_results = {}
    for lang_name, lang_data in languages.items():
        rdi_results[lang_name] = {}
        for hw_key, hw in HARDWARE_TIERS.items():
            rdi_val = compute_rdi(lang_data["fertility"], hw)
            rdi_results[lang_name][hw_key] = rdi_val

    # Print normalized RDI table
    baseline_rdi = rdi_results["English"]["H100_SXM5"]
    print(f"\n    Baseline (English, H100) RDI = {baseline_rdi:.6e}")
    print(f"    {'Language':<12} {'H100':>8} {'A100':>8} {'T4':>8} {'CPU':>8}")
    for lang in languages:
        vals = [rdi_results[lang][hw] / baseline_rdi for hw in HARDWARE_TIERS]
        print(f"    {lang:<12} {vals[0]:>8.1f}x {vals[1]:>8.1f}x "
              f"{vals[2]:>8.1f}x {vals[3]:>8.1f}x")

    # ------------------------------------------------------------------
    # Simulate multi-turn conversation
    # ------------------------------------------------------------------
    print("\n[4] Simulating multi-turn agentic conversation")
    multi_turn_results = {}
    for scenario_name, (lang, hw_key) in {
        "English_H100": ("English", "H100_SXM5"),
        "Swahili_T4": ("Swahili", "T4"),
    }.items():
        fert = languages[lang]["fertility"]
        hw = HARDWARE_TIERS[hw_key]
        results = simulate_multi_turn(lang, fert, hw, num_turns=5)
        multi_turn_results[scenario_name] = results

        print(f"\n    {scenario_name}:")
        for r in results:
            print(f"      Turn {r['turn']}: "
                  f"input={r['input_tokens']:4d} tok, "
                  f"output={r['output_tokens']:3d} tok, "
                  f"TTFT={r['ttft_ms']:8.1f}ms, "
                  f"total={r['total_ms']/1000:6.2f}s, "
                  f"KV={r['kv_cache_mb']:7.1f}MB "
                  f"{'[OOM]' if r['oom_risk'] else ''}")

    # ------------------------------------------------------------------
    # Save combined analysis JSON
    # ------------------------------------------------------------------
    combined_data = {
        "metadata": {
            "experiment": "Combined Resource Divide Analysis",
            "description": (
                "Demonstrates compounding effects of tokenization inefficiency "
                "(fertility) and hardware inequality on LLM inference latency."
            ),
            "model": MODEL_SPEC,
            "hardware_tiers": {k: {sk: sv for sk, sv in v.items() if sk != "name"}
                               for k, v in HARDWARE_TIERS.items()},
            "task": {
                "words_per_task": WORDS_PER_TASK,
                "output_tokens_baseline_en": OUTPUT_TOKENS_EN,
            },
            "derived_constants": {
                "model_size_bytes": MODEL_SIZE_BYTES,
                "model_size_gb": MODEL_SIZE_BYTES / 1e9,
                "kv_cache_per_token_bytes": KV_CACHE_PER_TOKEN_BYTES,
                "kv_cache_per_token_kb": KV_CACHE_PER_TOKEN_BYTES / 1024,
            },
        },
        "latency_comparison": latency_results,
        "resource_divide_index": {
            "raw": rdi_results,
            "normalized_baseline": "English_H100",
            "normalized": {
                lang: {hw: rdi_results[lang][hw] / baseline_rdi
                       for hw in HARDWARE_TIERS}
                for lang in languages
            },
        },
        "multi_turn_simulation": multi_turn_results,
        "language_support_gap": LANGUAGE_SUPPORT_DATA,
        "key_findings": {
            "hindi_t4_vs_english_h100_latency_ratio": (
                latency_results["Hindi"]["T4"]["total_latency_ms"]
                / latency_results["English"]["H100_SXM5"]["total_latency_ms"]
            ),
            "swahili_t4_vs_english_h100_latency_ratio": (
                latency_results["Swahili"]["T4"]["total_latency_ms"]
                / latency_results["English"]["H100_SXM5"]["total_latency_ms"]
            ),
            "max_rdi_ratio": (
                rdi_results["Hindi"]["CPU_Server"] / baseline_rdi
            ),
            "cumulative_time_ratio_turn5": (
                sum(r["total_ms"] for r in multi_turn_results["Swahili_T4"])
                / sum(r["total_ms"] for r in multi_turn_results["English_H100"])
            ),
        },
    }

    json_path = os.path.join(DATA_DIR, "combined_analysis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Saved combined analysis data to {json_path}")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print("\n[5] Generating figures...")

    plot_latency_comparison(
        languages, HARDWARE_TIERS,
        os.path.join(FIGURES_DIR, "fig3a_latency_comparison.png"),
    )

    plot_resource_divide_index(
        languages, HARDWARE_TIERS,
        os.path.join(FIGURES_DIR, "fig3b_resource_divide_index.png"),
    )

    plot_cumulative_disadvantage(
        languages, HARDWARE_TIERS,
        os.path.join(FIGURES_DIR, "fig3c_cumulative_disadvantage.png"),
    )

    plot_language_support_gap(
        os.path.join(FIGURES_DIR, "fig3d_language_support_gap.png"),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    en_h100 = latency_results["English"]["H100_SXM5"]["total_latency_ms"]
    sw_t4 = latency_results["Swahili"]["T4"]["total_latency_ms"]
    hi_t4 = latency_results["Hindi"]["T4"]["total_latency_ms"]
    print(f"  - English on H100: {en_h100/1000:.2f}s")
    print(f"  - Swahili on T4:   {sw_t4/1000:.2f}s  ({sw_t4/en_h100:.1f}x slower)")
    print(f"  - Hindi on T4:     {hi_t4/1000:.2f}s  ({hi_t4/en_h100:.1f}x slower)")
    print(f"  - Max RDI (Hindi/CPU): {rdi_results['Hindi']['CPU_Server']/baseline_rdi:.1f}x worse than English/H100")

    cum_en = sum(r["total_ms"] for r in multi_turn_results["English_H100"])
    cum_sw = sum(r["total_ms"] for r in multi_turn_results["Swahili_T4"])
    print(f"  - 5-turn cumulative: Swahili/T4 = {cum_sw/1000:.1f}s vs English/H100 = {cum_en/1000:.1f}s ({cum_sw/cum_en:.1f}x)")

    print("\nGenerated files:")
    print(f"  - {json_path}")
    print(f"  - {os.path.join(FIGURES_DIR, 'fig3a_latency_comparison.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'fig3b_resource_divide_index.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'fig3c_cumulative_disadvantage.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'fig3d_language_support_gap.png')}")


if __name__ == "__main__":
    main()
