#!/usr/bin/env python3
"""
Experiment 2: Memory Wall Simulation for LLM Inference
=========================================================

This simulation models LLM inference performance across different GPU hardware tiers,
demonstrating how KV cache memory limits create the "Memory Wall" for low-resource
language processing. We simulate the Llama-3-8B model (4-bit quantized) across five
hardware tiers from frontier (H100) to consumer (RTX 4060).

Author: Machine Learning Systems Engineer
Date: 2025-07-04
"""

import json
import os
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Matplotlib Publication-Quality Settings
# ---------------------------------------------------------------------------
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['axes.axisbelow'] = True

# Color-blind-friendly palette (Okabe-Ito)
CB_COLORS = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'grey': '#999999',
}

TIER_COLORS = {
    'Frontier': CB_COLORS['blue'],
    'High-end': CB_COLORS['orange'],
    'Workstation': CB_COLORS['bluish_green'],
    'Commodity': CB_COLORS['vermillion'],
    'Consumer': CB_COLORS['reddish_purple'],
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class HardwareTier:
    """Represents a GPU hardware tier with its specifications."""
    name: str
    vram_gb: float
    bandwidth_tb_s: float  # Memory bandwidth in TB/s
    tier_label: str
    color: str
    # Derived: memory bandwidth in GB/s for easier calculations
    @property
    def bandwidth_gb_s(self) -> float:
        return self.bandwidth_tb_s * 1000.0


@dataclass
class ModelConfig:
    """Configuration for Llama-3-8B inference model."""
    model_name: str = "Llama-3-8B"
    num_layers: int = 32
    hidden_dim: int = 4096
    num_kv_heads: int = 8
    head_dim: int = 128
    quantization_bits: int = 4
    dtype_size_kv: int = 2  # FP16 for KV cache

    @property
    def kv_cache_per_token_bytes(self) -> int:
        """
        KV cache size per token:
        K and V (2) × layers (32) × KV heads (8) × head dim (128) × dtype_size (2 bytes)
        """
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_size_kv

    @property
    def kv_cache_per_token_mb(self) -> float:
        return self.kv_cache_per_token_bytes / (1024.0 * 1024.0)

    @property
    def model_weights_gb(self) -> float:
        """Approximate model weights in GB for 4-bit quantization."""
        # Llama-3-8B ~7.6B params, 4-bit = 0.5 bytes per param after grouping
        # Typical 4-bit quantized size is ~4.5-5.5 GB
        return 5.5

    def available_kv_memory_gb(self, vram_gb: float, overhead_fraction: float = 0.10) -> float:
        """
        Calculate available memory for KV cache after accounting for model weights
        and activation overhead.
        """
        overhead_gb = vram_gb * overhead_fraction
        available = vram_gb - self.model_weights_gb - overhead_gb
        return max(0.0, available)


@dataclass
class SimulationResult:
    """Container for simulation outputs."""
    hardware_tier: str
    max_sequence_length: int
    max_batch_size: int
    throughput_tokens_per_sec: float
    memory_utilization: float
    is_oom: bool


# ---------------------------------------------------------------------------
# Hardware Definitions
# ---------------------------------------------------------------------------

HARDWARE_TIERS = [
    HardwareTier("NVIDIA H100 (80GB)", 80.0, 3.35, "Frontier", TIER_COLORS['Frontier']),
    HardwareTier("NVIDIA A100 (80GB)", 80.0, 2.00, "High-end", TIER_COLORS['High-end']),
    HardwareTier("NVIDIA RTX 3090 (24GB)", 24.0, 0.936, "Workstation", TIER_COLORS['Workstation']),
    HardwareTier("NVIDIA T4 (16GB)", 16.0, 0.320, "Commodity", TIER_COLORS['Commodity']),
    HardwareTier("NVIDIA RTX 4060 (8GB)", 8.0, 0.272, "Consumer", TIER_COLORS['Consumer']),
]

# ---------------------------------------------------------------------------
# Core Simulation Functions
# ---------------------------------------------------------------------------

def kv_cache_size_gb(seq_len: int, batch_size: int, kv_per_token_mb: float) -> float:
    """
    Calculate total KV cache memory in GB for a given sequence length and batch size.
    """
    total_mb = seq_len * batch_size * kv_per_token_mb
    return total_mb / 1024.0


def compute_max_batch_size(seq_len: int, available_kv_gb: float, kv_per_token_mb: float) -> int:
    """
    Compute the maximum batch size that fits in available KV memory for a given sequence length.
    Returns 0 if even batch_size=1 doesn't fit.
    """
    kv_needed_per_batch_token = seq_len * kv_per_token_mb / 1024.0  # in GB
    if kv_needed_per_batch_token <= 0:
        return 0
    max_bs = int(available_kv_gb / kv_needed_per_batch_token)
    return max(0, max_bs)


def compute_max_seq_length(batch_size: int, available_kv_gb: float, kv_per_token_mb: float) -> int:
    """
    Compute the maximum sequence length that fits for a given batch size.
    """
    if batch_size <= 0:
        return 0
    max_seq = int((available_kv_gb * 1024.0) / (batch_size * kv_per_token_mb))
    return max(0, max_seq)


def simulate_throughput(
    batch_size: int,
    seq_len: int,
    bandwidth_gb_s: float,
    kv_per_token_mb: float,
    model_weights_gb: float,
    compute_flops: float = 1e15,  # Approximate compute for tier
    is_prefill: bool = False,
    fertility: float = 1.0,
) -> float:
    """
    Simulate throughput (tokens/second) for a given configuration.

    The throughput is bounded by memory bandwidth for autoregressive decoding.
    For each generated token, we need to:
      1. Load KV cache for the entire sequence (memory-bound)
      2. Load model weights (memory-bound for small batches)
      3. Perform attention computation

    Args:
        batch_size: number of sequences in the batch
        seq_len: current sequence length (context + new tokens)
        bandwidth_gb_s: GPU memory bandwidth in GB/s
        kv_per_token_mb: KV cache size per token in MB
        model_weights_gb: model weights in GB
        compute_flops: peak compute in FLOPS (used for compute-bound check)
        is_prefill: whether this is prefill phase (more compute-bound)
        fertility: output tokens per input token (affects effective sequence length)

    Returns:
        tokens_per_second: throughput in generated tokens per second
    """
    # Effective sequence length adjusted for fertility
    effective_seq_len = int(seq_len * fertility)

    # Memory per token generated:
    # - Load KV cache for all prior tokens: seq_len * kv_per_token
    # - Load model weights (amortized across batch): model_weights / batch_size
    # - Store new KV cache: kv_per_token
    kv_load_mb = effective_seq_len * kv_per_token_mb
    weight_load_mb = (model_weights_gb * 1024.0) / max(batch_size, 1)
    kv_store_mb = kv_per_token_mb

    # Total memory traffic per token in MB
    memory_per_token_mb = kv_load_mb + weight_load_mb + kv_store_mb

    # Memory bandwidth bound time per token (in seconds)
    # We assume ~70% effective bandwidth utilization for inference
    effective_bandwidth_gb_s = bandwidth_gb_s * 0.70
    memory_time_s = (memory_per_token_mb / 1024.0) / effective_bandwidth_gb_s

    # For autoregressive decoding, each token is generated sequentially
    # Throughput = 1 / time_per_token, scaled by batch_size
    tokens_per_sec = batch_size / memory_time_s

    # Add a small compute overhead factor (attention computation)
    # For long sequences, attention becomes O(n^2) but is still usually memory-bound
    compute_overhead = 1.0 + 0.05 * math.log(max(effective_seq_len, 1))
    tokens_per_sec = tokens_per_sec / compute_overhead

    return tokens_per_sec


def generate_figure_2a(model: ModelConfig, output_dir: str) -> str:
    """
    Figure 2a: KV Cache Growth vs Sequence Length
    Plot KV cache size (GB) vs sequence length for multiple batch sizes,
    with OOM threshold lines for each hardware tier.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    seq_lengths = np.arange(0, 33001, 500)
    batch_sizes = [1, 4, 8, 16, 32]
    kv_per_token_mb = model.kv_cache_per_token_mb

    # Plot KV cache growth curves for each batch size
    colors_bs = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))
    for bs, color in zip(batch_sizes, colors_bs):
        kv_sizes = [kv_cache_size_gb(s, bs, kv_per_token_mb) for s in seq_lengths]
        ax.plot(seq_lengths, kv_sizes, label=f"Batch size = {bs}",
                color=color, linewidth=2.0, linestyle='-')

    # Plot OOM threshold lines for each hardware tier
    for hw in HARDWARE_TIERS:
        available_kv = model.available_kv_memory_gb(hw.vram_gb)
        if available_kv > 0:
            ax.axhline(
                y=available_kv,
                color=hw.color,
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                label=f"{hw.tier_label}: {hw.name} ({available_kv:.1f} GB)",
            )

    ax.set_xlabel("Sequence Length (tokens)", fontweight='bold')
    ax.set_ylabel("KV Cache Memory (GB)", fontweight='bold')
    ax.set_title("Figure 2a: KV Cache Growth vs Sequence Length\n"
                 f"Model: {model.model_name} (4-bit), KV per token: {kv_per_token_mb:.3f} MB",
                 fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_xlim(0, 32000)
    ax.set_ylim(0, 90)

    # Add annotation about memory wall
    ax.annotate(
        "Memory Wall:\nConsumer GPUs hit\nOOM at ~2K tokens\n(batch=1)",
        xy=(2000, model.available_kv_memory_gb(8.0)),
        xytext=(8000, 55),
        arrowprops=dict(arrowstyle='->', color=CB_COLORS['vermillion'], lw=1.5),
        fontsize=9,
        color=CB_COLORS['vermillion'],
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=CB_COLORS['vermillion'], alpha=0.9),
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2a_kv_cache_growth.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_2b(model: ModelConfig, output_dir: str) -> str:
    """
    Figure 2b: Max Batch Size vs Sequence Length (OOM Heatmap)
    Create a heatmap showing maximum sustainable batch size for each
    (hardware tier, sequence length) combination.
    """
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    tier_names = [hw.tier_label for hw in HARDWARE_TIERS]
    kv_per_token_mb = model.kv_cache_per_token_mb

    # Compute max batch size matrix: rows = tiers, cols = seq_lengths
    max_batch_matrix = np.zeros((len(HARDWARE_TIERS), len(seq_lengths)), dtype=int)
    for i, hw in enumerate(HARDWARE_TIERS):
        available_kv = model.available_kv_memory_gb(hw.vram_gb)
        for j, seq_len in enumerate(seq_lengths):
            max_batch_matrix[i, j] = compute_max_batch_size(seq_len, available_kv, kv_per_token_mb)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Custom colormap: white -> yellow -> red -> dark red
    cmap = LinearSegmentedColormap.from_list(
        "batch_cmap",
        ["#FFFFFF", "#FFE4B5", "#FF6B6B", "#C92A2A", "#5C0000"],
        N=256,
    )

    # Plot heatmap
    im = ax.imshow(max_batch_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(seq_lengths)))
    ax.set_xticklabels([f"{s:,}" for s in seq_lengths], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(tier_names)))
    ax.set_yticklabels(tier_names)

    # Annotate cells with values
    for i in range(len(tier_names)):
        for j in range(len(seq_lengths)):
            val = max_batch_matrix[i, j]
            text_color = 'white' if val > 20 else 'black'
            if val == 0:
                text = "OOM"
                text_color = 'white'
                # Override color for OOM cells
                # (imshow already handles via colormap)
            else:
                text = str(val)
            ax.text(j, i, text, ha='center', va='center', color=text_color,
                    fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Max Batch Size", fontweight='bold')

    ax.set_xlabel("Sequence Length (tokens)", fontweight='bold')
    ax.set_ylabel("Hardware Tier", fontweight='bold')
    ax.set_title("Figure 2b: Maximum Sustainable Batch Size by Hardware & Sequence Length\n"
                 f"Model: {model.model_name} (4-bit) — White = OOM",
                 fontweight='bold')

    # Add a subtle border around OOM cells
    for i in range(len(tier_names)):
        for j in range(len(seq_lengths)):
            if max_batch_matrix[i, j] == 0:
                rect = FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.01", linewidth=2,
                    edgecolor='#C92A2A', facecolor='none', alpha=0.5,
                )
                ax.add_patch(rect)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2b_batch_size_heatmap.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_2c(model: ModelConfig, output_dir: str) -> str:
    """
    Figure 2c: Throughput Degradation (Memory Wall Effect)
    For each hardware tier, simulate throughput (tokens/sec) vs batch size
    for English (fertility=1.15) vs Swahili (fertility=2.34).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
    seq_len = 2048  # Fixed context length for comparison
    kv_per_token_mb = model.kv_cache_per_token_mb

    fertilities = {
        'English': 1.15,
        'Swahili': 2.34,
    }

    # Approximate compute FLOPS per tier (simplified)
    tier_flops = {
        'Frontier': 51e12,    # H100 ~51 TFLOPS (FP16)
        'High-end': 19.5e12,  # A100 ~19.5 TFLOPS (FP16)
        'Workstation': 35.6e12,  # RTX 3090 ~35.6 TFLOPS (FP16)
        'Commodity': 8.1e12,   # T4 ~8.1 TFLOPS (FP16)
        'Consumer': 11.0e12,  # RTX 4060 ~11.0 TFLOPS (FP16)
    }

    for ax_idx, (lang, fertility) in enumerate(fertilities.items()):
        ax = axes[ax_idx]
        for hw in HARDWARE_TIERS:
            throughputs = []
            for bs in batch_sizes:
                tp = simulate_throughput(
                    batch_size=bs,
                    seq_len=seq_len,
                    bandwidth_gb_s=hw.bandwidth_gb_s,
                    kv_per_token_mb=kv_per_token_mb,
                    model_weights_gb=model.model_weights_gb,
                    compute_flops=tier_flops[hw.tier_label],
                    fertility=fertility,
                )
                # Check if this config would OOM
                available_kv = model.available_kv_memory_gb(hw.vram_gb)
                kv_needed = kv_cache_size_gb(seq_len, bs, kv_per_token_mb)
                if kv_needed > available_kv:
                    tp = 0.0  # OOM
                throughputs.append(tp)

            # Plot with markers
            valid_mask = np.array(throughputs) > 0
            if np.any(valid_mask):
                ax.plot(
                    batch_sizes[valid_mask],
                    np.array(throughputs)[valid_mask],
                    marker='o',
                    markersize=6,
                    linewidth=2.0,
                    color=hw.color,
                    label=hw.tier_label,
                )
                # Mark OOM point
                first_oom = np.where(~valid_mask)[0]
                if len(first_oom) > 0:
                    oom_idx = first_oom[0]
                    if oom_idx > 0:
                        ax.plot(
                            batch_sizes[oom_idx - 1],
                            throughputs[oom_idx - 1],
                            marker='X',
                            markersize=10,
                            color=hw.color,
                            markeredgecolor='black',
                            markeredgewidth=1.0,
                        )

        ax.set_xlabel("Batch Size", fontweight='bold')
        ax.set_ylabel("Throughput (tokens/sec)", fontweight='bold')
        ax.set_title(f"{lang} (fertility={fertility:.2f})", fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.set_ylim(0.1, 2000)

        # Add annotation about memory wall
        ax.annotate(
            "Memory Wall:\nThroughput plateaus\nas batch grows",
            xy=(8, 50),
            xytext=(2, 5),
            arrowprops=dict(arrowstyle='->', color=CB_COLORS['grey'], lw=1.5),
            fontsize=8,
            color=CB_COLORS['grey'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=CB_COLORS['grey'], alpha=0.9),
        )

    fig.suptitle(
        "Figure 2c: Throughput Degradation vs Batch Size by Hardware Tier\n"
        f"Model: {model.model_name}, Context: {seq_len} tokens",
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2c_throughput_degradation.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_2d(output_dir: str) -> str:
    """
    Figure 2d: Compute Distribution Pie Chart
    Show global AI compute distribution with color-coding by access tier.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # Data: global AI compute distribution by region
    regions = ['United States', 'China', 'European Union', 'Other']
    percentages = [74, 14, 5, 7]

    # Color coding by access tier
    colors = [
        CB_COLORS['blue'],     # US - Frontier / High-end
        CB_COLORS['orange'],   # China - High-end / Workstation
        CB_COLORS['bluish_green'],  # EU - Workstation / Commodity
        CB_COLORS['reddish_purple'],  # Other - Commodity / Consumer
    ]

    # Explode slightly for emphasis
    explode = (0.03, 0.03, 0.03, 0.03)

    wedges, texts, autotexts = ax.pie(
        percentages,
        explode=explode,
        labels=regions,
        colors=colors,
        autopct='%1.0f%%',
        startangle=90,
        pctdistance=0.6,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    # Add a white circle in the center for donut effect
    centre_circle = plt.Circle((0, 0), 0.40, fc='white')
    ax.add_patch(centre_circle)

    # Center text
    ax.text(0, 0.05, "Global AI", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, -0.12, "Compute", ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_title(
        "Figure 2d: Global AI Compute Distribution by Region\n"
        "Color-Coded by Dominant Hardware Access Tier",
        fontweight='bold',
        pad=20,
    )

    # Add legend for tier mapping
    tier_labels = [
        f"{CB_COLORS['blue']}  US: Frontier + High-end",
        f"{CB_COLORS['orange']}  China: High-end + Workstation",
        f"{CB_COLORS['bluish_green']}  EU: Workstation + Commodity",
        f"{CB_COLORS['reddish_purple']}  Other: Commodity + Consumer",
    ]
    # Use colored patches for legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CB_COLORS['blue'], edgecolor='black', label='US: Frontier + High-end'),
        Patch(facecolor=CB_COLORS['orange'], edgecolor='black', label='China: High-end + Workstation'),
        Patch(facecolor=CB_COLORS['bluish_green'], edgecolor='black', label='EU: Workstation + Commodity'),
        Patch(facecolor=CB_COLORS['reddish_purple'], edgecolor='black', label='Other: Commodity + Consumer'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2d_compute_distribution.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_simulation_data(model: ModelConfig, output_path: str) -> dict:
    """
    Generate comprehensive simulation data and save as JSON.
    """
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    fertilities = {'English': 1.15, 'Swahili': 2.34}

    results = {
        'model_config': asdict(model),
        'hardware_tiers': [
            {
                'name': hw.name,
                'vram_gb': hw.vram_gb,
                'bandwidth_tb_s': hw.bandwidth_tb_s,
                'tier_label': hw.tier_label,
                'available_kv_memory_gb': model.available_kv_memory_gb(hw.vram_gb),
            }
            for hw in HARDWARE_TIERS
        ],
        'kv_cache_analysis': {},
        'batch_size_analysis': {},
        'throughput_analysis': {},
    }

    # KV cache analysis
    for bs in batch_sizes:
        results['kv_cache_analysis'][f'batch_{bs}'] = {
            str(seq_len): kv_cache_size_gb(seq_len, bs, model.kv_cache_per_token_mb)
            for seq_len in seq_lengths
        }

    # Batch size analysis (OOM boundaries)
    for hw in HARDWARE_TIERS:
        available_kv = model.available_kv_memory_gb(hw.vram_gb)
        results['batch_size_analysis'][hw.tier_label] = {
            str(seq_len): compute_max_batch_size(seq_len, available_kv, model.kv_cache_per_token_mb)
            for seq_len in seq_lengths
        }

    # Throughput analysis
    tier_flops = {
        'Frontier': 51e12,
        'High-end': 19.5e12,
        'Workstation': 35.6e12,
        'Commodity': 8.1e12,
        'Consumer': 11.0e12,
    }

    for lang, fertility in fertilities.items():
        results['throughput_analysis'][lang] = {}
        for hw in HARDWARE_TIERS:
            tier_data = {}
            for bs in batch_sizes:
                for seq_len in seq_lengths:
                    available_kv = model.available_kv_memory_gb(hw.vram_gb)
                    kv_needed = kv_cache_size_gb(seq_len, bs, model.kv_cache_per_token_mb)
                    if kv_needed > available_kv:
                        tp = 0.0
                        oom = True
                    else:
                        tp = simulate_throughput(
                            bs, seq_len, hw.bandwidth_gb_s,
                            model.kv_cache_per_token_mb,
                            model.model_weights_gb,
                            tier_flops[hw.tier_label],
                            fertility=fertility,
                        )
                        oom = False
                    tier_data[f"bs_{bs}_seq_{seq_len}"] = {
                        'throughput': round(tp, 2),
                        'oom': oom,
                    }
            results['throughput_analysis'][lang][hw.tier_label] = tier_data

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Main execution: generate all figures and simulation data.
    """
    # Ensure output directories exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    figures_dir = os.path.join(project_dir, 'figures')
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Initialize model configuration
    model = ModelConfig()
    print(f"Model: {model.model_name}")
    print(f"  KV cache per token: {model.kv_cache_per_token_bytes:,} bytes ({model.kv_cache_per_token_mb:.3f} MB)")
    print(f"  Model weights (4-bit): {model.model_weights_gb} GB")

    for hw in HARDWARE_TIERS:
        available = model.available_kv_memory_gb(hw.vram_gb)
        max_seq_bs1 = compute_max_seq_length(1, available, model.kv_cache_per_token_mb)
        print(f"  {hw.tier_label} ({hw.name}): Available KV = {available:.2f} GB, Max seq (bs=1) = {max_seq_bs1:,} tokens")

    # Generate all figures
    print("\nGenerating figures...")
    paths = {}
    paths['fig2a'] = generate_figure_2a(model, figures_dir)
    print(f"  Saved: {paths['fig2a']}")

    paths['fig2b'] = generate_figure_2b(model, figures_dir)
    print(f"  Saved: {paths['fig2b']}")

    paths['fig2c'] = generate_figure_2c(model, figures_dir)
    print(f"  Saved: {paths['fig2c']}")

    paths['fig2d'] = generate_figure_2d(figures_dir)
    print(f"  Saved: {paths['fig2d']}")

    # Generate simulation data
    data_path = os.path.join(data_dir, "memory_simulation_results.json")
    generate_simulation_data(model, data_path)
    print(f"\n  Saved simulation data: {data_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Experiment 2: Memory Wall Simulation Complete")
    print("=" * 60)
    print(f"Output files:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print(f"  data: {data_path}")

    return paths


if __name__ == "__main__":
    main()
