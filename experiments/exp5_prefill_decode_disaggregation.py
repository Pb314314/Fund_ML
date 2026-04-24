#!/usr/bin/env python3
"""
Experiment 5: Prefill vs Decode Hardware Disaggregation Simulation
===================================================================

LLM inference has two fundamentally different phases with distinct hardware needs:

  Prefill Phase:
    - Processes all input tokens in parallel (matrix multiplications)
    - Compute-bound: high FLOPS utilization (85%+)
    - Reads model weights once, then heavy matrix ops
    - Time scales with input_length * model_size (roughly)

  Decode Phase:
    - Generates one token at a time (autoregressive)
    - Memory-bandwidth-bound: reads entire KV cache each step
    - Low compute utilization (20-40%)
    - Time scales with KV_cache_size + model_weights per token

Key Insight: Decode dominates 95-99.9% of end-to-end latency for most workloads.
This motivates heterogeneous clusters: H100 for prefill, commodity GPU for decode.

References:
  - WVA et al. "A Global Optimization Control Plane for LLM Inference" 2026
  - DUET: "Disaggregated Hybrid Mamba-Transformer LLMs" 2025
  - Towards Data Science: "Prefill Is Compute-Bound, Decode Is Memory-Bound" 2026

Author: Student Researcher
Date: 2025-04-25
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import (
    MODEL, GPU_SPECS, LANGUAGE_FERTILITY,
    BANDWIDTH_UTILIZATION, COMPUTE_UTILIZATION,
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


# ---------------------------------------------------------------------------
# Phase-Specific Simulation Functions
# ---------------------------------------------------------------------------

def simulate_prefill_ms(seq_len: int, gpu_key: str) -> float:
    """
    Prefill latency in milliseconds.
    
    Formula:
      Linear_FLOPs = 2 * params * seq_len (feedforward + attention projections)
      Attention_FLOPs = 2 * layers * num_q_heads * head_dim * seq_len^2
      Total_FLOPs = Linear_FLOPs + Attention_FLOPs
      Time = Total_FLOPs / (compute_TFLOPS_dense * 1e12 * utilization)
    """
    gpu = GPU_SPECS[gpu_key]
    
    # Linear layers: ~2 * params * seq_len
    linear_flops = 2 * MODEL.params_b * 1e9 * seq_len
    
    # Attention: 2 * layers * num_heads * head_dim * seq_len^2
    # For GQA, we use num_kv_heads in actual KV computation, but
    # attention computation still uses num_q_heads for query projection
    attn_flops = 2 * MODEL.num_layers * MODEL.num_attention_heads * MODEL.head_dim * seq_len * seq_len
    
    total_flops = linear_flops + attn_flops
    effective_compute = gpu.compute_tflops_dense * 1e12 * COMPUTE_UTILIZATION
    time_s = total_flops / effective_compute
    
    return time_s * 1000.0


def simulate_decode_ms(seq_len: int, num_output_tokens: int, gpu_key: str, fertility: float = 1.0) -> float:
    """
    Decode latency in milliseconds.
    
    Formula (memory-bound):
      Per-step memory = model_weights_bytes + KV_cache_bytes(seq_len)
      Time_per_token = memory / (bandwidth * utilization * 1e9)
      Total_time = Time_per_token * num_output_tokens
    
    Effective seq_len accounts for fertility (low-resource languages have more tokens).
    """
    gpu = GPU_SPECS[gpu_key]
    effective_seq_len = seq_len * fertility
    
    weights_bytes = MODEL.model_weights_gb * 1e9
    kv_bytes = effective_seq_len * MODEL.kv_cache_per_token_bytes
    total_bytes_per_token = weights_bytes + kv_bytes + MODEL.kv_cache_per_token_bytes  # + new KV
    
    effective_bw = gpu.bandwidth_gb_s * BANDWIDTH_UTILIZATION * 1e9  # bytes/s
    time_per_token_s = total_bytes_per_token / effective_bw
    
    return time_per_token_s * num_output_tokens * 1000.0


def simulate_disaggregated(prefill_gpu: str, decode_gpu: str,
                            input_tok: int, output_tok: int,
                            fertility: float = 1.0) -> dict:
    """Simulate prefill on one GPU and decode on another."""
    prefill_ms = simulate_prefill_ms(input_tok, prefill_gpu)
    decode_ms = simulate_decode_ms(input_tok, output_tok, decode_gpu, fertility)
    total_ms = prefill_ms + decode_ms
    
    prefill_cost = GPU_SPECS[prefill_gpu].cost_per_hour_usd * (prefill_ms / 3.6e6)
    decode_cost = GPU_SPECS[decode_gpu].cost_per_hour_usd * (decode_ms / 3.6e6)
    
    return {
        'prefill_gpu': prefill_gpu,
        'decode_gpu': decode_gpu,
        'prefill_ms': prefill_ms,
        'decode_ms': decode_ms,
        'total_ms': total_ms,
        'prefill_pct': (prefill_ms / total_ms) * 100 if total_ms > 0 else 0,
        'decode_pct': (decode_ms / total_ms) * 100 if total_ms > 0 else 0,
        'prefill_cost': prefill_cost,
        'decode_cost': decode_cost,
        'total_cost': prefill_cost + decode_cost,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figure_5a(output_dir: str) -> str:
    """Figure 5a: Prefill vs Decode latency breakdown by workload type."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    workloads = [
        ("Short Q&A (50in/30out)", 50, 30),
        ("Medium Essay (500in/300out)", 500, 300),
        ("Long Document (4000in/500out)", 4000, 500),
        ("Code Generation (200in/800out)", 200, 800),
    ]
    
    gpus = ['H100', 'A100', 'T4', 'RTX4060']
    colors = {'H100': '#1B5E20', 'A100': '#2E7D32', 'T4': '#C62828', 'RTX4060': '#6A1B9A'}
    
    for idx, (name, inp, out) in enumerate(workloads):
        ax = axes[idx // 2, idx % 2]
        
        prefill_times = []
        decode_times = []
        for gpu_key in gpus:
            p = simulate_prefill_ms(inp, gpu_key)
            d = simulate_decode_ms(inp, out, gpu_key)
            prefill_times.append(p)
            decode_times.append(d)
        
        x = np.arange(len(gpus))
        width = 0.35
        bars1 = ax.bar(x - width/2, prefill_times, width, label='Prefill (TTFT)', color='#1565C0', alpha=0.8)
        bars2 = ax.bar(x + width/2, decode_times, width, label='Decode', color='#C62828', alpha=0.8)
        
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title(name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(gpus, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_yscale('log')
        ax.set_ylim(1, 100000)
    
    fig.suptitle(f"Figure 5: Prefill vs Decode Phase Latency Breakdown\n"
                 f"Model: {MODEL.name} ({MODEL.quantization})",
                 fontweight='bold', y=1.02, fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig5a_prefill_decode_breakdown.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_5b(output_dir: str) -> str:
    """Figure 5b: Pareto frontier for disaggregated serving."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Medium Essay workload
    input_tok, output_tok = 500, 300
    
    # Generate all combinations
    combinations = []
    for pg in ['H100', 'A100', 'T4']:
        for dg in ['H100', 'A100', 'T4', 'RTX4060']:
            res = simulate_disaggregated(pg, dg, input_tok, output_tok)
            combinations.append(res)
    
    # Pareto-optimal: no other point is better in both latency and cost
    def is_pareto_optimal(point, all_points):
        for p in all_points:
            if p is point:
                continue
            if p['total_ms'] <= point['total_ms'] and p['total_cost'] <= point['total_cost']:
                if p['total_ms'] < point['total_ms'] or p['total_cost'] < point['total_cost']:
                    return False
        return True
    
    pareto = [c for c in combinations if is_pareto_optimal(c, combinations)]
    
    # Plot all points
    for c in combinations:
        color = '#1976D2' if c['prefill_gpu'] == 'H100' else '#388E3C' if c['prefill_gpu'] == 'A100' else '#F57C00'
        ax.scatter(c['total_ms'], c['total_cost'], s=200, color=color, alpha=0.5,
                  edgecolors='black', linewidth=0.5, zorder=3)
    
    # Plot Pareto frontier
    pareto_sorted = sorted(pareto, key=lambda x: x['total_ms'])
    ax.plot([p['total_ms'] for p in pareto_sorted], [p['total_cost'] for p in pareto_sorted],
           'r--', linewidth=3, alpha=0.8, label='Pareto Frontier', zorder=4)
    
    # Annotate key Pareto points
    for c in pareto_sorted:
        label = f"{c['prefill_gpu']}\u2192{c['decode_gpu']}"
        ax.annotate(label, (c['total_ms'], c['total_cost']),
                   textcoords="offset points", xytext=(10, 10), fontsize=9, fontweight='bold')
    
    # Baseline (same GPU for both)
    for gpu in ['H100', 'A100', 'T4', 'RTX4060']:
        res = simulate_disaggregated(gpu, gpu, input_tok, output_tok)
        ax.scatter(res['total_ms'], res['total_cost'], s=300, color='black', marker='X',
                  alpha=0.8, edgecolors='white', linewidth=2, zorder=5)
        ax.annotate(f"Baseline\n{gpu}", (res['total_ms'], res['total_cost']),
                   textcoords="offset points", xytext=(-40, -30), fontsize=8,
                   color='black', fontweight='bold')
    
    ax.set_xlabel('End-to-End Latency (ms)', fontweight='bold')
    ax.set_ylabel('Serving Cost ($ per request)', fontweight='bold')
    ax.set_title("Figure 5b: Disaggregated Serving Cost-Latency Tradeoff\n"
                 "Medium Essay: 500in/300out tokens", fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1976D2', markersize=10, label='Prefill: H100'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#388E3C', markersize=10, label='Prefill: A100'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F57C00', markersize=10, label='Prefill: T4'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=12, label='Baseline (Same GPU)'),
        Line2D([0], [0], linestyle='--', color='red', linewidth=2, label='Pareto Frontier'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig5b_pareto_disaggregated.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# JSON Data
# ---------------------------------------------------------------------------

def generate_experiment5_data(output_path: str) -> dict:
    workloads = [
        ("Short Q&A", 50, 30),
        ("Medium Essay", 500, 300),
        ("Long Document", 4000, 500),
        ("Code Generation", 200, 800),
    ]
    
    results = {'workloads': {}, 'disaggregated': {}}
    
    for name, inp, out in workloads:
        results['workloads'][name] = {}
        for gpu in ['H100', 'A100', 'T4', 'RTX4060']:
            p = simulate_prefill_ms(inp, gpu)
            d = simulate_decode_ms(inp, out, gpu)
            results['workloads'][name][gpu] = {
                'prefill_ms': round(p, 2),
                'decode_ms': round(d, 2),
                'total_ms': round(p + d, 2),
                'prefill_pct': round((p / (p + d)) * 100, 1) if p + d > 0 else 0,
                'decode_pct': round((d / (p + d)) * 100, 1) if p + d > 0 else 0,
            }
    
    # Disaggregated analysis for Medium Essay
    results['disaggregated']['Medium Essay'] = []
    for pg in ['H100', 'A100', 'T4']:
        for dg in ['H100', 'A100', 'T4', 'RTX4060']:
            res = simulate_disaggregated(pg, dg, 500, 300)
            results['disaggregated']['Medium Essay'].append(res)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    figures_dir = os.path.join(project_dir, 'figures')
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print("Experiment 5: Prefill vs Decode Disaggregation")
    print(f"  Model: {MODEL.name} ({MODEL.quantization})")
    
    # Print breakdown
    print("\nLatency Breakdown by Workload:")
    workloads = [
        ("Short Q&A", 50, 30),
        ("Medium Essay", 500, 300),
        ("Long Document", 4000, 500),
        ("Code Generation", 200, 800),
    ]
    for name, inp, out in workloads:
        print(f"\n  {name}:")
        for gpu in ['H100', 'A100', 'T4', 'RTX4060']:
            p = simulate_prefill_ms(inp, gpu)
            d = simulate_decode_ms(inp, out, gpu)
            total = p + d
            print(f"    {gpu}: TTFT={p:.1f}ms, Decode={d:.1f}ms, "
                  f"Total={total:.1f}ms (Prefill:{p/total*100:.1f}%, Decode:{d/total*100:.1f}%)")
    
    print("\nGenerating figures...")
    generate_figure_5a(figures_dir)
    print("  Saved: fig5a_prefill_decode_breakdown.png")
    generate_figure_5b(figures_dir)
    print("  Saved: fig5b_pareto_disaggregated.png")
    
    data_path = os.path.join(data_dir, "experiment_5_results.json")
    generate_experiment5_data(data_path)
    print(f"\n  Saved data: {data_path}")
    
    print("\n" + "=" * 60)
    print("Experiment 5: Prefill-Decode Disaggregation — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
