#!/usr/bin/env python3
"""
Experiment 3: Combined Analysis — End-to-End Latency, RDI, and Language Gap
=============================================================================

Integrates tokenization fertility (Exp1), memory constraints (Exp2), and
hardware specs to produce end-to-end inference metrics.

New: Uses 4-bit quantized model (5.5 GB) — consistent with Exp2/4/5.
Old version incorrectly used FP16 (16 GB), causing 3x latency error.

Key Metrics:
------------
1. Resource Divide Index (RDI) = fertility^2 / (bandwidth_GB/s * VRAM_GB)
2. Latency = Prefill + Decode = (compute_time) + (memory_time)
3. Cumulative Disadvantage Index = sum of per-turn latency ratio

Mathematical Models:
--------------------
Prefill (compute-bound):
  FLOPs ≈ 2 * params * seq_len (linear) + 2 * layers * heads * head_dim * seq_len^2 (attn)
  Time = FLOPs / (compute_TFLOPS * utilization)

Decode (memory-bound):
  Bytes per token ≈ model_weights_bytes + KV_cache_bytes(seq_len)
  Time = bytes / (bandwidth_GB/s * utilization * 1e9)

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

from config import (
    MODEL, GPU_SPECS, LANGUAGE_FERTILITY,
    available_kv_memory_gb, kv_cache_size_gb,
    MEMORY_OVERHEAD_PCT, BANDWIDTH_UTILIZATION, COMPUTE_UTILIZATION,
)

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['axes.axisbelow'] = True

CB_COLORS = {
    'black': '#000000', 'orange': '#E69F00', 'sky_blue': '#56B4E9',
    'bluish_green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'reddish_purple': '#CC79A7', 'grey': '#999999',
}

TIER_COLORS = {
    'Frontier': '#1B5E20',
    'High-end': '#2E7D32',
    'Workstation': '#F57C00',
    'Commodity': '#C62828',
    'Consumer': '#6A1B9A',
    'CPU': '#455A64',
}


# ---------------------------------------------------------------------------
# Latency Simulation Functions
# ---------------------------------------------------------------------------

def simulate_prefill_time_ms(seq_len: int, gpu_key: str) -> float:
    """
    Prefill latency in milliseconds.
    
    Prefill processes all input tokens in parallel. Dominated by:
      1. Linear projection FLOPs: 2 * params * seq_len
      2. Attention FLOPs: 2 * layers * num_heads * head_dim * seq_len^2
    
    Time = Total_FLOPs / (compute_tflops_dense * 1e12 * COMPUTE_UTILIZATION)
    
    Source: Kwon et al. "vLLM: Efficient Memory Management for LLM Serving" SOSP 2023
    """
    gpu = GPU_SPECS[gpu_key]
    
    # Linear layers: ~2 * params * seq_len (each param used ~2x per token in forward pass)
    linear_flops = 2 * MODEL.params_b * 1e9 * seq_len
    
    # Attention: 2 * layers * num_q_heads * head_dim * seq_len^2
    # For GQA, effective heads used in attention = num_kv_heads (simplified)
    attn_flops = 2 * MODEL.num_layers * MODEL.num_attention_heads * MODEL.head_dim * seq_len * seq_len
    
    total_flops = linear_flops + attn_flops
    effective_compute = gpu.compute_tflops_dense * 1e12 * COMPUTE_UTILIZATION
    time_s = total_flops / effective_compute
    
    return time_s * 1000.0  # ms


def simulate_decode_time_ms(seq_len: int, num_output_tokens: int, gpu_key: str, fertility: float = 1.0) -> float:
    """
    Decode latency in milliseconds.
    
    Decode generates one token at a time. Memory-bound:
      1. Load KV cache for all prior tokens: seq_len * kv_per_token
      2. Load model weights (4-bit quantized): model_weights_gb
      3. Store new KV cache: kv_per_token
    
    Time per token = total_bytes / (bandwidth * utilization * 1e9)
    
    Effective sequence length = seq_len * fertility (for low-resource languages)
    """
    gpu = GPU_SPECS[gpu_key]
    effective_seq_len = seq_len * fertility
    
    # Memory bytes per decode step
    kv_cache_bytes = effective_seq_len * MODEL.kv_cache_per_token_bytes
    weights_bytes = MODEL.model_weights_gb * 1e9  # 4-bit quantized weights
    new_kv_bytes = MODEL.kv_cache_per_token_bytes
    
    total_bytes_per_token = kv_cache_bytes + weights_bytes + new_kv_bytes
    effective_bw = gpu.bandwidth_gb_s * BANDWIDTH_UTILIZATION * 1e9  # bytes/s
    time_per_token_s = total_bytes_per_token / effective_bw
    
    total_time_ms = time_per_token_s * num_output_tokens * 1000.0
    
    return total_time_ms


def simulate_end_to_end_latency_ms(input_tokens: int, output_tokens: int, gpu_key: str, fertility: float = 1.0) -> dict:
    """Full end-to-end latency with prefill + decode breakdown."""
    prefill_ms = simulate_prefill_time_ms(input_tokens, gpu_key)
    decode_ms = simulate_decode_time_ms(input_tokens, output_tokens, gpu_key, fertility)
    total_ms = prefill_ms + decode_ms
    
    return {
        'prefill_ms': prefill_ms,
        'decode_ms': decode_ms,
        'total_ms': total_ms,
        'prefill_pct': (prefill_ms / total_ms) * 100 if total_ms > 0 else 0,
        'decode_pct': (decode_ms / total_ms) * 100 if total_ms > 0 else 0,
    }


def compute_rdi(fertility: float, gpu_key: str) -> float:
    """
    Resource Divide Index.
    RDI = fertility^2 / (bandwidth_GB/s * VRAM_GB)
    
    Higher RDI = worse resource access.
    Captures the compounding effect of:
      - More tokens per word (fertility)
      - Lower memory bandwidth (decode bottleneck)
      - Smaller VRAM (shorter max context)
    """
    gpu = GPU_SPECS[gpu_key]
    return (fertility ** 2) / (gpu.bandwidth_gb_s * gpu.vram_gb)


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------

def generate_figure_3a(output_dir: str) -> str:
    """Figure 3a: End-to-end latency comparison (same task, different hardware × language)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    task_words = 100
    # Same semantic task: 100 words input, 50 words output
    input_words = task_words
    output_words = 50
    
    languages = ['English', 'Chinese', 'Spanish', 'Arabic', 'Swahili', 'Hindi']
    gpu_keys = ['H100', 'A100', 'T4', 'RTX4060', 'CPU']
    gpu_labels = ['H100\n(Frontier)', 'A100\n(High-end)', 'T4\n(Commodity)', 'RTX 4060\n(Consumer)', 'CPU\n(Server)']
    
    x = np.arange(len(gpu_keys))
    width = 0.13
    
    for i, lang in enumerate(languages):
        fertility = LANGUAGE_FERTILITY[lang]
        input_tokens = int(input_words * fertility)
        output_tokens = int(output_words * fertility)
        
        latencies = []
        for gpu_key in gpu_keys:
            res = simulate_end_to_end_latency_ms(input_tokens, output_tokens, gpu_key, fertility)
            latencies.append(res['total_ms'])
        
        offset = (i - len(languages)/2 + 0.5) * width
        bars = ax.bar(x + offset, latencies, width, label=lang)
        
        # Color by fertility level
        if fertility < 1.0:
            color = '#1B5E20'
        elif fertility < 1.5:
            color = '#2E7D32'
        elif fertility < 2.0:
            color = '#F57C00'
        else:
            color = '#C62828'
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(0.7)
    
    ax.set_ylabel("End-to-End Latency (ms, log scale)", fontweight='bold')
    ax.set_xlabel("Hardware Tier", fontweight='bold')
    ax.set_title(f"(a) Same {task_words}-Word Task: Latency by Hardware and Language\n"
                 f"Input={input_words} words, Output={output_words} words, Model: {MODEL.name}",
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_labels, fontsize=9)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9, title="Language", title_fontsize=10)
    ax.set_ylim(1, 500000)
    
    # Add annotation for the worst case
    worst_latency = simulate_end_to_end_latency_ms(
        int(100 * LANGUAGE_FERTILITY['Hindi']), int(50 * LANGUAGE_FERTILITY['Hindi']), 'CPU', LANGUAGE_FERTILITY['Hindi'])
    ax.annotate(f"Worst case:\nHindi on CPU = {worst_latency['total_ms']:.0f} ms\n({worst_latency['total_ms']/1000:.1f} s)",
                xy=(4, worst_latency['total_ms']), xytext=(3.2, 100000),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3a_latency_comparison.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_3b(output_dir: str) -> str:
    """Figure 3b: Resource Divide Index heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    languages = ['English', 'Chinese', 'Spanish', 'Arabic', 'Swahili', 'Hindi']
    gpu_keys = ['H100', 'A100', 'T4', 'RTX4060']
    
    rdi_matrix = np.zeros((len(languages), len(gpu_keys)))
    for i, lang in enumerate(languages):
        for j, gpu_key in enumerate(gpu_keys):
            rdi_matrix[i, j] = compute_rdi(LANGUAGE_FERTILITY[lang], gpu_key)
    
    # Normalize to English on H100 = 1.0
    baseline = rdi_matrix[0, 0]
    rdi_normalized = rdi_matrix / baseline
    
    im = ax.imshow(rdi_normalized, cmap='YlOrRd', aspect='auto', vmin=1, vmax=1000)
    
    ax.set_xticks(np.arange(len(gpu_keys)))
    ax.set_xticklabels(gpu_keys)
    ax.set_yticks(np.arange(len(languages)))
    ax.set_yticklabels(languages)
    
    for i in range(len(languages)):
        for j in range(len(gpu_keys)):
            val = rdi_normalized[i, j]
            text_color = 'white' if val > 100 else 'black'
            ax.text(j, i, f"{val:.1f}x", ha='center', va='center',
                    color=text_color, fontsize=10, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("RDI (normalized to English/H100 = 1.0)", fontweight='bold')
    ax.set_xlabel("GPU Hardware", fontweight='bold')
    ax.set_ylabel("Language", fontweight='bold')
    ax.set_title("(b) Resource Divide Index (RDI) Heatmap\nHigher = Worse Resource Access",
                 fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3b_resource_divide_index.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_3c(output_dir: str) -> str:
    """Figure 3c: Cumulative disadvantage over multi-turn conversation."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    num_turns = 5
    words_per_turn = 80
    batch_size = 1
    
    comparisons = [
        ('H100', 'English', 'English / H100'),
        ('H100', 'Hindi', 'Hindi / H100'),
        ('T4', 'English', 'English / T4'),
        ('T4', 'Hindi', 'Hindi / T4'),
    ]
    
    turn_numbers = np.arange(1, num_turns + 1)
    
    for gpu_key, lang, label in comparisons:
        fertility = LANGUAGE_FERTILITY[lang]
        cumulative_latency_ms = 0
        latencies = []
        
        for turn in range(1, num_turns + 1):
            # Input accumulates all previous turns
            input_tokens = int(words_per_turn * fertility * (turn - 1) * 2)
            output_tokens = int(words_per_turn * fertility)
            
            res = simulate_end_to_end_latency_ms(input_tokens, output_tokens, gpu_key, fertility)
            cumulative_latency_ms += res['total_ms']
            latencies.append(cumulative_latency_ms)
        
        color = '#1565C0' if lang == 'English' else '#C62828'
        linestyle = '-' if gpu_key == 'H100' else '--'
        ax.plot(turn_numbers, latencies, color=color, linestyle=linestyle,
               marker='o', markersize=8, linewidth=2.5, label=label)
    
    ax.set_xlabel("Conversation Turn", fontweight='bold')
    ax.set_ylabel("Cumulative Latency (ms, log scale)", fontweight='bold')
    ax.set_title("(c) Cumulative Disadvantage in Multi-Turn Conversations\n"
                 f"({words_per_turn} words/turn, batch={batch_size}, Model: {MODEL.name})",
                 fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xticks(turn_numbers)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3c_cumulative_disadvantage.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_3d(output_dir: str) -> str:
    """Figure 3d: Language support gap visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    from config import LANGUAGE_SUPPORT_GAP
    
    # Left: Language support pyramid
    categories = ['World\nLanguages', 'With NLP\nResources', 'With LLM\nTraining', 'African Languages\nwith Any NLP']
    values = [
        LANGUAGE_SUPPORT_GAP['total_world_languages'],
        LANGUAGE_SUPPORT_GAP['languages_with_nlp_resources'],
        LANGUAGE_SUPPORT_GAP['languages_with_llm_training_data'],
        LANGUAGE_SUPPORT_GAP['african_languages_with_any_nlp'],
    ]
    colors_pyramid = ['#1565C0', '#2E7D32', '#F57C00', '#C62828']
    
    bars = ax1.barh(categories, values, color=colors_pyramid, alpha=0.85, edgecolor='black')
    ax1.set_xlabel("Number of Languages (log scale)", fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_title("(d) Language Support Gap\n(7,000+ languages, ~0.6% have LLM support)",
                  fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        ax1.text(val * 1.5, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=11, fontweight='bold')
    
    # Right: Digital content by language
    languages_content = {
        'English': 49.7, 'Chinese': 19.04, 'Spanish': 7.70,
        'Arabic': 3.65, 'Hindi': 3.77, 'Russian': 3.75,
        'Japanese': 2.23, 'French': 3.42, 'German': 2.93,
        'Portuguese': 2.50, 'Swahili': 0.0025, 'Bengali': 0.20,
    }
    langs = list(languages_content.keys())
    contents = list(languages_content.values())
    colors_content = ['#1B5E20' if c > 5 else '#F57C00' if c > 1 else '#C62828'
                       for c in contents]
    
    bars2 = ax2.barh(langs, contents, color=colors_content, alpha=0.85, edgecolor='black')
    ax2.set_xlabel("Internet Content (%)", fontweight='bold')
    ax2.set_title("(e) Digital Content Distribution by Language", fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars2, contents):
        ax2.text(val * 1.2, bar.get_y() + bar.get_height()/2,
                f"{val}%", va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3d_language_support_gap.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# JSON Data Generation
# ---------------------------------------------------------------------------

def generate_combined_data(output_path: str) -> dict:
    """Generate comprehensive combined analysis data."""
    languages = ['English', 'Chinese', 'Spanish', 'Arabic', 'Swahili', 'Hindi']
    gpu_keys = ['H100', 'A100', 'T4', 'RTX4060']
    
    results = {
        'metadata': {
            'model': {'name': MODEL.name, 'quantization': MODEL.quantization,
                      'params_b': MODEL.params_b, 'model_weights_gb': MODEL.model_weights_gb,
                      'kv_cache_per_token_mb': MODEL.kv_cache_per_token_mb},
            'simulation_constants': {
                'compute_utilization': COMPUTE_UTILIZATION,
                'bandwidth_utilization': BANDWIDTH_UTILIZATION,
                'memory_overhead_pct': MEMORY_OVERHEAD_PCT,
            },
        },
        'latency_analysis': {},
        'rdi_analysis': {},
        'cumulative_disadvantage': {},
    }
    
    # Latency analysis for same 100-word task
    input_words, output_words = 100, 50
    for lang in languages:
        fertility = LANGUAGE_FERTILITY[lang]
        input_tokens = int(input_words * fertility)
        output_tokens = int(output_words * fertility)
        
        results['latency_analysis'][lang] = {}
        for gpu_key in gpu_keys:
            res = simulate_end_to_end_latency_ms(input_tokens, output_tokens, gpu_key, fertility)
            results['latency_analysis'][lang][gpu_key] = res
    
    # RDI analysis
    for lang in languages:
        results['rdi_analysis'][lang] = {}
        for gpu_key in gpu_keys + ['CPU']:
            results['rdi_analysis'][lang][gpu_key] = compute_rdi(LANGUAGE_FERTILITY[lang], gpu_key)
    
    # Cumulative disadvantage (5 turns)
    for lang in languages:
        for gpu_key in ['H100', 'T4']:
            fertility = LANGUAGE_FERTILITY[lang]
            cumulative_ms = 0
            per_turn = []
            for turn in range(1, 6):
                input_tokens = int(80 * fertility * (turn - 1) * 2)
                output_tokens = int(80 * fertility)
                res = simulate_end_to_end_latency_ms(input_tokens, output_tokens, gpu_key, fertility)
                cumulative_ms += res['total_ms']
                per_turn.append({'turn': turn, 'latency_ms': res['total_ms'], 'cumulative_ms': cumulative_ms})
            
            key = f"{lang}_{gpu_key}"
            results['cumulative_disadvantage'][key] = per_turn
    
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
    
    print(f"Experiment 3: Combined Analysis")
    print(f"  Model: {MODEL.name} ({MODEL.quantization}, {MODEL.model_weights_gb:.2f} GB)")
    print(f"  KV cache: {MODEL.kv_cache_per_token_mb:.4f} MB/token")
    print()
    
    # Print sample latency values
    print("Sample latency for 100-word task:")
    for lang in ['English', 'Hindi']:
        fertility = LANGUAGE_FERTILITY[lang]
        print(f"\n  {lang} (fertility={fertility}):")
        for gpu_key in ['H100', 'A100', 'T4', 'RTX4060']:
            res = simulate_end_to_end_latency_ms(int(100*fertility), int(50*fertility), gpu_key, fertility)
            print(f"    {gpu_key}: Prefill={res['prefill_ms']:.1f}ms, Decode={res['decode_ms']:.1f}ms, "
                  f"Total={res['total_ms']:.1f}ms (Decode: {res['decode_pct']:.1f}%)")
    
    print("\nGenerating figures...")
    paths = {}
    paths['fig3a'] = generate_figure_3a(figures_dir)
    print(f"  Saved: {paths['fig3a']}")
    paths['fig3b'] = generate_figure_3b(figures_dir)
    print(f"  Saved: {paths['fig3b']}")
    paths['fig3c'] = generate_figure_3c(figures_dir)
    print(f"  Saved: {paths['fig3c']}")
    paths['fig3d'] = generate_figure_3d(figures_dir)
    print(f"  Saved: {paths['fig3d']}")
    
    data_path = os.path.join(data_dir, "combined_analysis_results.json")
    generate_combined_data(data_path)
    print(f"\n  Saved data: {data_path}")
    
    print("\n" + "=" * 60)
    print("Experiment 3: Combined Analysis — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
