#!/usr/bin/env python3
"""
Experiment 2: Memory Wall Simulation for LLM Inference
=========================================================

Simulates KV cache memory constraints and throughput across GPU hardware tiers.
Uses the unified config.py for all constants to ensure cross-experiment consistency.

Mathematical Model:
-------------------
KV cache per token (bytes) = 2 (K+V) * layers * kv_heads * head_dim * dtype_size
                           = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 0.125 MB

Available KV memory (GB) = VRAM * (1 - overhead) - model_weights

Max sequence length = available_kv_gb * 1024 / (batch_size * kv_per_token_mb)

Decode throughput (tok/s) = batch_size / time_per_token
  where time_per_token = (weights_bytes + kv_bytes) / (bandwidth * utilization)

Assumptions:
  - 4-bit AWQ quantized model (5.5 GB weights) — standard for production serving
  - KV cache in FP16 (2 bytes/element) — required for stable inference
  - 10% memory overhead for activations and workspace buffers
  - 70% effective bandwidth utilization (cache misses, scheduling overhead)

Author: Student Researcher
Date: 2025-04-25
"""

import json
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# Import unified configuration
from config import (
    MODEL, GPU_SPECS, LANGUAGE_FERTILITY,
    available_kv_memory_gb, kv_cache_size_gb,
    MEMORY_OVERHEAD_PCT, BANDWIDTH_UTILIZATION,
)

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
    'black': '#000000', 'orange': '#E69F00', 'sky_blue': '#56B4E9',
    'bluish_green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'reddish_purple': '#CC79A7', 'grey': '#999999',
}

TIER_COLORS = {
    'Frontier': CB_COLORS['blue'],
    'High-end': CB_COLORS['orange'],
    'Workstation': CB_COLORS['bluish_green'],
    'Commodity': CB_COLORS['vermillion'],
    'Consumer': CB_COLORS['reddish_purple'],
}

# ---------------------------------------------------------------------------
# Core Simulation Functions
# ---------------------------------------------------------------------------

def compute_max_batch_size(seq_len: int, available_kv_gb: float, kv_per_token_mb: float) -> int:
    """Maximum batch size that fits in available KV memory. Returns 0 if OOM at bs=1."""
    kv_needed_per_batch = seq_len * kv_per_token_mb / 1024.0
    if kv_needed_per_batch <= 0:
        return 0
    return max(0, int(available_kv_gb / kv_needed_per_batch))


def compute_max_seq_length(batch_size: int, available_kv_gb: float, kv_per_token_mb: float) -> int:
    """Maximum sequence length for a given batch size."""
    if batch_size <= 0:
        return 0
    return max(0, int((available_kv_gb * 1024.0) / (batch_size * kv_per_token_mb)))


def simulate_throughput(
    batch_size: int,
    seq_len: int,
    bandwidth_gb_s: float,
    kv_per_token_mb: float,
    model_weights_gb: float,
    compute_flops: float,
    fertility: float = 1.0,
) -> float:
    """
    Simulate decode throughput (tokens/second).
    
    Decode is memory-bandwidth-bound. Each token requires:
      1. Load KV cache for all prior tokens: seq_len * kv_per_token
      2. Load model weights (amortized across batch): weights / batch_size
      3. Store new KV cache: kv_per_token
    
    Effective bandwidth = peak_bandwidth * BANDWIDTH_UTILIZATION (70%)
    """
    effective_seq_len = int(seq_len * fertility)
    
    kv_load_mb = effective_seq_len * kv_per_token_mb
    weight_load_mb = (model_weights_gb * 1024.0) / max(batch_size, 1)
    kv_store_mb = kv_per_token_mb
    
    memory_per_token_mb = kv_load_mb + weight_load_mb + kv_store_mb
    effective_bw = bandwidth_gb_s * BANDWIDTH_UTILIZATION
    memory_time_s = (memory_per_token_mb / 1024.0) / effective_bw
    
    tokens_per_sec = batch_size / memory_time_s
    
    # Attention compute overhead: O(n^2) but usually still memory-bound
    compute_overhead = 1.0 + 0.05 * math.log(max(effective_seq_len, 1))
    return tokens_per_sec / compute_overhead


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------

def generate_figure_2a(output_dir: str) -> str:
    """Figure 2a: KV Cache Growth vs Sequence Length with OOM thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seq_lengths = np.arange(0, 33001, 500)
    batch_sizes = [1, 4, 8, 16, 32]
    kv_per_token_mb = MODEL.kv_cache_per_token_mb
    
    colors_bs = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))
    for bs, color in zip(batch_sizes, colors_bs):
        kv_sizes = [kv_cache_size_gb(s, bs, kv_per_token_mb) for s in seq_lengths]
        ax.plot(seq_lengths, kv_sizes, label=f"Batch = {bs}", color=color, linewidth=2.0)
    
    # OOM threshold lines
    for key, gpu in GPU_SPECS.items():
        if key in ['H100', 'A100', 'RTX3090', 'T4', 'RTX4060']:
            available_kv = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
            if available_kv > 0:
                ax.axhline(y=available_kv, color=TIER_COLORS.get(gpu.tier_label, 'gray'),
                          linestyle='--', linewidth=1.5, alpha=0.8,
                          label=f"{gpu.tier_label}: {gpu.name} ({available_kv:.1f} GB)")
    
    ax.set_xlabel("Sequence Length (tokens)", fontweight='bold')
    ax.set_ylabel("KV Cache Memory (GB)", fontweight='bold')
    ax.set_title(f"(a) KV Cache Growth vs Sequence Length\nModel: {MODEL.name} ({MODEL.quantization}), "
                 f"KV/token = {kv_per_token_mb:.3f} MB", fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_xlim(0, 32000)
    ax.set_ylim(0, 90)
    
    ax.annotate("Memory Wall:\nRTX 4060 hits\nOOM at ~14K tokens\n(batch=1)",
                xy=(14000, available_kv_memory_gb(8.0, MODEL.model_weights_gb)),
                xytext=(8000, 55),
                arrowprops=dict(arrowstyle='->', color=CB_COLORS['vermillion'], lw=1.5),
                fontsize=9, color=CB_COLORS['vermillion'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=CB_COLORS['vermillion'], alpha=0.9))
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2a_kv_cache_growth.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_2b(output_dir: str) -> str:
    """Figure 2b: Max Batch Size Heatmap (OOM boundaries)."""
    seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    gpu_list = ['H100', 'A100', 'RTX3090', 'T4', 'RTX4060']
    tier_names = [GPU_SPECS[g].tier_label for g in gpu_list]
    kv_per_token_mb = MODEL.kv_cache_per_token_mb
    
    max_batch_matrix = np.zeros((len(gpu_list), len(seq_lengths)), dtype=int)
    for i, key in enumerate(gpu_list):
        gpu = GPU_SPECS[key]
        available_kv = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
        for j, seq_len in enumerate(seq_lengths):
            max_batch_matrix[i, j] = compute_max_batch_size(seq_len, available_kv, kv_per_token_mb)
    
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = LinearSegmentedColormap.from_list("batch_cmap",
        ["#FFFFFF", "#FFE4B5", "#FF6B6B", "#C92A2A", "#5C0000"], N=256)
    
    im = ax.imshow(max_batch_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xticks(np.arange(len(seq_lengths)))
    ax.set_xticklabels([f"{s:,}" for s in seq_lengths], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(tier_names)))
    ax.set_yticklabels(tier_names)
    
    for i in range(len(tier_names)):
        for j in range(len(seq_lengths)):
            val = max_batch_matrix[i, j]
            text_color = 'white' if val > 20 else 'black'
            text = "OOM" if val == 0 else str(val)
            ax.text(j, i, text, ha='center', va='center', color=text_color,
                    fontsize=10, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Max Batch Size", fontweight='bold')
    ax.set_xlabel("Sequence Length (tokens)", fontweight='bold')
    ax.set_ylabel("Hardware Tier", fontweight='bold')
    ax.set_title("(b) Maximum Sustainable Batch Size by Hardware & Sequence\n"
                 f"Model: {MODEL.name} ({MODEL.quantization}) — White = OOM",
                 fontweight='bold')
    
    for i in range(len(tier_names)):
        for j in range(len(seq_lengths)):
            if max_batch_matrix[i, j] == 0:
                rect = FancyBboxPatch((j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.01", linewidth=2,
                    edgecolor='#C92A2A', facecolor='none', alpha=0.5)
                ax.add_patch(rect)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2b_batch_size_heatmap.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_2c(output_dir: str) -> str:
    """Figure 2c: Throughput Degradation — English vs Swahili."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
    seq_len = 2048
    kv_per_token_mb = MODEL.kv_cache_per_token_mb
    
    fertilities = {'English': LANGUAGE_FERTILITY['English'], 'Swahili': LANGUAGE_FERTILITY['Swahili']}
    gpu_list = ['H100', 'A100', 'RTX3090', 'T4', 'RTX4060']
    
    for ax_idx, (lang, fertility) in enumerate(fertilities.items()):
        ax = axes[ax_idx]
        for key in gpu_list:
            gpu = GPU_SPECS[key]
            throughputs = []
            for bs in batch_sizes:
                tp = simulate_throughput(bs, seq_len, gpu.bandwidth_gb_s,
                    kv_per_token_mb, MODEL.model_weights_gb,
                    gpu.compute_tflops_dense, fertility)
                available_kv = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
                kv_needed = kv_cache_size_gb(seq_len, bs, kv_per_token_mb)
                if kv_needed > available_kv:
                    tp = 0.0
                throughputs.append(tp)
            
            valid_mask = np.array(throughputs) > 0
            if np.any(valid_mask):
                ax.plot(batch_sizes[valid_mask], np.array(throughputs)[valid_mask],
                       marker='o', markersize=6, linewidth=2.0,
                       color=TIER_COLORS.get(gpu.tier_label, 'gray'),
                       label=gpu.tier_label)
        
        ax.set_xlabel("Batch Size", fontweight='bold')
        ax.set_ylabel("Throughput (tokens/sec)", fontweight='bold')
        ax.set_title(f"{lang} (fertility={fertility:.2f})", fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(b) for b in batch_sizes])
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.set_ylim(0.1, 2000)
    
    fig.suptitle(f"(c) Throughput Degradation vs Batch Size\nModel: {MODEL.name}, Context: {seq_len} tokens",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2c_throughput_degradation.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_2d(output_dir: str) -> str:
    """Figure 2d: Global AI Compute Distribution."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    from config import COMPUTE_DISTRIBUTION
    regions = list(COMPUTE_DISTRIBUTION.keys())
    percentages = list(COMPUTE_DISTRIBUTION.values())
    
    colors = [CB_COLORS['blue'], CB_COLORS['orange'],
              CB_COLORS['bluish_green'], CB_COLORS['reddish_purple']]
    explode = (0.03, 0.03, 0.03, 0.03)
    
    wedges, texts, autotexts = ax.pie(percentages, explode=explode, labels=regions,
                                       colors=colors, autopct='%1.0f%%', startangle=90,
                                       pctdistance=0.6,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.40, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0.05, "Global AI", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, -0.12, "Compute", ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_title("(d) Global AI Compute Distribution by Region", fontweight='bold', pad=20)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2d_compute_distribution.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# JSON Data Generation
# ---------------------------------------------------------------------------

def generate_simulation_data(output_path: str) -> dict:
    """Generate comprehensive simulation data and save as JSON."""
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    fertilities = {'English': LANGUAGE_FERTILITY['English'], 'Swahili': LANGUAGE_FERTILITY['Swahili']}
    
    results = {
        'metadata': {
            'model': {'name': MODEL.name, 'quantization': MODEL.quantization,
                      'params_b': MODEL.params_b, 'model_weights_gb': MODEL.model_weights_gb,
                      'kv_cache_per_token_mb': MODEL.kv_cache_per_token_mb,
                      'kv_cache_formula': '2 * num_layers * num_kv_heads * head_dim * 2_bytes',
                      'kv_cache_per_token_bytes': MODEL.kv_cache_per_token_bytes},
            'hardware_tiers': [
                {'name': gpu.name, 'vram_gb': gpu.vram_gb,
                 'bandwidth_gb_s': gpu.bandwidth_gb_s,
                 'tier_label': gpu.tier_label,
                 'available_kv_memory_gb': available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)}
                for gpu in GPU_SPECS.values()
            ],
            'simulation_constants': {
                'memory_overhead_pct': MEMORY_OVERHEAD_PCT,
                'bandwidth_utilization': BANDWIDTH_UTILIZATION,
            },
        },
        'kv_cache_analysis': {},
        'batch_size_analysis': {},
        'throughput_analysis': {},
    }
    
    # KV cache analysis
    for bs in batch_sizes:
        results['kv_cache_analysis'][f'batch_{bs}'] = {
            str(seq_len): kv_cache_size_gb(seq_len, bs, MODEL.kv_cache_per_token_mb)
            for seq_len in seq_lengths
        }
    
    # Batch size analysis
    for key, gpu in GPU_SPECS.items():
        available_kv = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
        results['batch_size_analysis'][gpu.tier_label] = {
            str(seq_len): compute_max_batch_size(seq_len, available_kv, MODEL.kv_cache_per_token_mb)
            for seq_len in seq_lengths
        }
    
    # Throughput analysis
    for lang, fertility in fertilities.items():
        results['throughput_analysis'][lang] = {}
        for key, gpu in GPU_SPECS.items():
            tier_data = {}
            for bs in batch_sizes:
                for seq_len in seq_lengths:
                    available_kv = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
                    kv_needed = kv_cache_size_gb(seq_len, bs, MODEL.kv_cache_per_token_mb)
                    oom = kv_needed > available_kv
                    if oom:
                        tp = 0.0
                    else:
                        tp = simulate_throughput(bs, seq_len, gpu.bandwidth_gb_s,
                            MODEL.kv_cache_per_token_mb, MODEL.model_weights_gb,
                            gpu.compute_tflops_dense, fertility)
                    tier_data[f"bs_{bs}_seq_{seq_len}"] = {'throughput': round(tp, 2), 'oom': oom}
            results['throughput_analysis'][lang][gpu.tier_label] = tier_data
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    figures_dir = os.path.join(project_dir, 'figures')
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Model: {MODEL.name} ({MODEL.quantization})")
    print(f"  KV cache per token: {MODEL.kv_cache_per_token_bytes:,} bytes ({MODEL.kv_cache_per_token_mb:.4f} MB)")
    print(f"  Model weights: {MODEL.model_weights_gb:.2f} GB")
    
    for key, gpu in GPU_SPECS.items():
        available = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
        max_seq = compute_max_seq_length(1, available, MODEL.kv_cache_per_token_mb)
        print(f"  {gpu.tier_label} ({gpu.name}): Available KV = {available:.2f} GB, Max seq (bs=1) = {max_seq:,} tokens")
    
    print("\nGenerating figures...")
    paths = {}
    paths['fig2a'] = generate_figure_2a(figures_dir)
    print(f"  Saved: {paths['fig2a']}")
    paths['fig2b'] = generate_figure_2b(figures_dir)
    print(f"  Saved: {paths['fig2b']}")
    paths['fig2c'] = generate_figure_2c(figures_dir)
    print(f"  Saved: {paths['fig2c']}")
    paths['fig2d'] = generate_figure_2d(figures_dir)
    print(f"  Saved: {paths['fig2d']}")
    
    data_path = os.path.join(data_dir, "memory_simulation_results.json")
    generate_simulation_data(data_path)
    print(f"\n  Saved simulation data: {data_path}")
    
    print("\n" + "=" * 60)
    print("Experiment 2: Memory Wall Simulation — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
