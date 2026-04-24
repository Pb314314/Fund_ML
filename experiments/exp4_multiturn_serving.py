#!/usr/bin/env python3
"""
Experiment 4: Multi-Turn Conversational KV Cache Accumulation + Serving Simulation
================================================================================

Simulates real-world agentic workloads where each conversation turn accumulates
KV cache, reducing throughput and service capacity (max concurrent users).

Scenario: 32 concurrent users per GPU, each engaging in a multi-turn conversation.
Each turn adds new user input + model response to the KV cache.

Key Finding: Low-resource languages hit OOM faster because higher fertility
             means more tokens per turn = faster KV cache growth.

Mathematical Model:
-------------------
Per-turn token addition: tokens_new = words_per_turn * fertility * 2 (input+output)
Cumulative KV per sequence: KV = cumulative_tokens * kv_per_token
Total KV for batch: KV_total = per_seq_KV * batch_size
OOM when: KV_total > available_kv_memory

Author: Student Researcher
Date: 2025-04-25
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

from config import (
    MODEL, GPU_SPECS, LANGUAGE_FERTILITY,
    available_kv_memory_gb, kv_cache_size_gb,
    MEMORY_OVERHEAD_PCT,
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


def simulate_serving_turns(gpu_key: str, language: str, num_turns: int = 20,
                            words_per_turn: int = 80, batch_size: int = 32) -> dict:
    """Simulate a serving scenario with multiple concurrent users."""
    gpu = GPU_SPECS[gpu_key]
    fertility = LANGUAGE_FERTILITY[language]
    
    available_kv = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
    if available_kv <= 0:
        return {'error': f'GPU {gpu_key} cannot fit model weights'}
    
    results = {
        'gpu': gpu_key,
        'gpu_name': gpu.name,
        'language': language,
        'fertility': fertility,
        'batch_size': batch_size,
        'available_kv_gb': round(available_kv, 4),
        'turns': [],
        'oom_turn': None,
    }
    
    cumulative_tokens_per_seq = 0
    for turn in range(1, num_turns + 1):
        new_tokens = int(words_per_turn * fertility)
        total_new_tokens = new_tokens * 2  # user input + model response
        cumulative_tokens_per_seq += total_new_tokens
        
        total_kv_gb = kv_cache_size_gb(cumulative_tokens_per_seq, batch_size, MODEL.kv_cache_per_token_mb)
        oom = total_kv_gb > available_kv
        
        # Decode throughput (memory-bound, simplified)
        weights_bytes = MODEL.model_weights_gb * 1e9
        kv_bytes = total_kv_gb * 1e9
        bytes_per_step = weights_bytes + kv_bytes
        effective_bw = gpu.bandwidth_gb_s * 0.70 * 1e9
        time_per_token_s = bytes_per_step / effective_bw
        throughput = batch_size / time_per_token_s if time_per_token_s > 0 else 0
        
        results['turns'].append({
            'turn': turn,
            'cumulative_tokens_per_seq': cumulative_tokens_per_seq,
            'total_kv_gb': round(total_kv_gb, 4),
            'throughput_tok_s': round(throughput, 2),
            'oom': oom,
        })
        
        if oom and results['oom_turn'] is None:
            results['oom_turn'] = turn
    
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figure_4a(output_dir: str) -> str:
    """Figure 4a: KV Cache accumulation + throughput degradation in serving."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    gpus_to_plot = ['H100', 'A100', 'T4', 'RTX4060']
    colors = {'English': '#1565C0', 'Hindi': '#C62828'}
    
    # Left: KV cache growth
    ax = axes[0]
    for lang, color in colors.items():
        for gpu_key in gpus_to_plot:
            res = simulate_serving_turns(gpu_key, lang, num_turns=15, batch_size=32)
            if 'error' in res:
                continue
            turns = [t['turn'] for t in res['turns'] if not t['oom'] or t['turn'] <= res['oom_turn']]
            kv = [t['total_kv_gb'] for t in res['turns'] if not t['oom'] or t['turn'] <= res['oom_turn']]
            ax.plot(turns, kv, color=color, linestyle='-' if lang == 'English' else '--',
                   marker='o', alpha=0.7, linewidth=2, label=f"{gpu_key} + {lang}")
    
    # OOM threshold lines
    for gpu_key in gpus_to_plot:
        gpu = GPU_SPECS[gpu_key]
        available = available_kv_memory_gb(gpu.vram_gb, MODEL.model_weights_gb)
        if available > 0:
            ax.axhline(y=available, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(0.5, available + 1, f"{gpu.tier_label} limit ({available:.1f} GB)",
                   fontsize=8, color='gray', alpha=0.8)
    
    ax.set_xlabel("Conversation Turn", fontweight='bold')
    ax.set_ylabel("Total KV Cache (GB)", fontweight='bold')
    ax.set_title(f"(a) KV Cache Accumulation (batch_size=32)\nModel: {MODEL.name}", fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.set_ylim(0, 80)
    
    # Right: Max concurrent users
    ax = axes[1]
    for lang, color in colors.items():
        for gpu_key in gpus_to_plot:
            res = simulate_serving_turns(gpu_key, lang, num_turns=15, batch_size=1)
            if 'error' in res:
                continue
            turns = [t['turn'] for t in res['turns']]
            available = available_kv_memory_gb(GPU_SPECS[gpu_key].vram_gb, MODEL.model_weights_gb)
            max_conc = []
            for t in res['turns']:
                kv_per_seq = kv_cache_size_gb(t['cumulative_tokens_per_seq'], 1, MODEL.kv_cache_per_token_mb)
                if kv_per_seq > 0:
                    mc = int(available / kv_per_seq)
                else:
                    mc = 999
                max_conc.append(mc)
            ax.plot(turns, max_conc, color=color, linestyle='-' if lang == 'English' else '--',
                   marker='s', alpha=0.7, linewidth=2, label=f"{gpu_key} + {lang}")
    
    ax.set_xlabel("Conversation Turn", fontweight='bold')
    ax.set_ylabel("Max Concurrent Users (log scale)", fontweight='bold')
    ax.set_title("(b) Service Capacity Degradation\n(Max users before OOM)", fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4a_multiturn_kv_concurrent.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def generate_figure_4b(output_dir: str) -> str:
    """Figure 4b: OOM Heatmap for serving scenario (batch=32)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gpus_list = ['H100', 'A100', 'T4', 'RTX4060']
    langs_list = ['English', 'Chinese', 'Spanish', 'Arabic', 'Swahili', 'Hindi']
    
    oom_matrix = np.zeros((len(gpus_list), len(langs_list)), dtype=int)
    for i, gpu_key in enumerate(gpus_list):
        for j, lang in enumerate(langs_list):
            res = simulate_serving_turns(gpu_key, lang, num_turns=20, batch_size=32)
            if 'error' in res:
                oom_matrix[i, j] = 0
            elif res['oom_turn']:
                oom_matrix[i, j] = res['oom_turn']
            else:
                oom_matrix[i, j] = 20
    
    cmap = LinearSegmentedColormap.from_list("oom_cmap",
        ["#FFFFFF", "#FFE4B5", "#FF6B6B", "#C92A2A", "#5C0000"], N=20)
    im = ax.imshow(oom_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=20)
    
    ax.set_xticks(range(len(langs_list)))
    ax.set_xticklabels(langs_list, fontsize=11)
    ax.set_yticks(range(len(gpus_list)))
    ax.set_yticklabels(gpus_list, fontsize=11)
    ax.set_title("(c) Max Turns Before OOM (Serving Scenario)\n"
                 f"batch_size=32, {MODEL.name}", fontweight='bold')
    
    for i in range(len(gpus_list)):
        for j in range(len(langs_list)):
            val = int(oom_matrix[i, j])
            text = "OOM" if val == 0 else (str(val) if val < 20 else "20+")
            color = 'white' if val < 8 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color,
                    fontsize=11, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Turns Before OOM", fontweight='bold')
    ax.set_xlabel("Language", fontweight='bold')
    ax.set_ylabel("GPU Hardware", fontweight='bold')
    
    # Highlight critical cells
    for i in range(len(gpus_list)):
        for j in range(len(langs_list)):
            if oom_matrix[i, j] <= 3 and oom_matrix[i, j] > 0:
                rect = FancyBboxPatch((j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="round,pad=0.01", linewidth=2.5,
                    edgecolor='#C92A2A', facecolor='none', alpha=0.7)
                ax.add_patch(rect)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4b_serving_oom_heatmap.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# JSON Data
# ---------------------------------------------------------------------------

def generate_experiment4_data(output_path: str) -> dict:
    all_results = {}
    for gpu_key in ['H100', 'A100', 'T4', 'RTX4060']:
        for lang in ['English', 'Chinese', 'Spanish', 'Arabic', 'Swahili', 'Hindi']:
            key = f"{gpu_key}_{lang}"
            all_results[key] = simulate_serving_turns(gpu_key, lang, num_turns=20, batch_size=32)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    figures_dir = os.path.join(project_dir, 'figures')
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Experiment 4: Multi-Turn Serving Simulation")
    print(f"  Model: {MODEL.name} ({MODEL.quantization})")
    print(f"  Batch size: 32 (concurrent users)")
    
    # Print key results
    print("\nKey Results (batch=32, 80 words/turn):")
    for gpu_key in ['H100', 'A100', 'T4', 'RTX4060']:
        for lang in ['English', 'Hindi']:
            res = simulate_serving_turns(gpu_key, lang, num_turns=20, batch_size=32)
            if 'error' not in res:
                oom = res['oom_turn']
                last = res['turns'][-1]
                print(f"  {gpu_key}+{lang}: OOM@turn{oom if oom else 'Never'}, "
                      f"Final KV={last['total_kv_gb']:.1f}GB")
    
    print("\nGenerating figures...")
    generate_figure_4a(figures_dir)
    print("  Saved: fig4a_multiturn_kv_concurrent.png")
    generate_figure_4b(figures_dir)
    print("  Saved: fig4b_serving_oom_heatmap.png")
    
    data_path = os.path.join(data_dir, "experiment_4_results.json")
    generate_experiment4_data(data_path)
    print(f"\n  Saved data: {data_path}")
    
    print("\n" + "=" * 60)
    print("Experiment 4: Multi-Turn Serving — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
