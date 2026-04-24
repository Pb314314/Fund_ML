"""
Experiment 5: Prefill vs Decode Hardware Disaggregation Simulation
Models how LLM inference has two distinct phases with different hardware needs:
- Prefill: compute-bound (matrix multiplications)
- Decode: memory-bandwidth-bound (KV cache reads)
Shows optimal workload-to-hardware assignment for heterogeneous clusters.
"""

import numpy as np
import matplotlib.pyplot as plt

# GPU specs
GPUS = {
    'H100': {'vram_gb': 80, 'bandwidth_gb_s': 3350, 'compute_tflops': 989, 'cost_per_hour': 25.0},
    'A100': {'vram_gb': 80, 'bandwidth_gb_s': 2039, 'compute_tflops': 312, 'cost_per_hour': 15.0},
    'T4': {'vram_gb': 16, 'bandwidth_gb_s': 320, 'compute_tflops': 65, 'cost_per_hour': 2.5},
    'RTX4060': {'vram_gb': 8, 'bandwidth_gb_s': 272, 'compute_tflops': 15.11, 'cost_per_hour': 1.0},
}

MODEL_WEIGHTS_GB = 5.5

def simulate_prefill_latency(gpu_name: str, seq_len: int) -> float:
    """Prefill latency in seconds (compute-bound)"""
    gpu = GPUS[gpu_name]
    num_layers, hidden_dim, num_heads, head_dim = 32, 4096, 32, 128
    linear_flops = 2 * num_layers * hidden_dim * hidden_dim * seq_len
    attn_flops = 2 * num_layers * num_heads * head_dim * seq_len * seq_len
    compute_time = (linear_flops + attn_flops) / (gpu['compute_tflops'] * 1e12)
    weights_bytes = MODEL_WEIGHTS_GB * 1024**3
    memory_time = weights_bytes / (gpu['bandwidth_gb_s'] * 1e9)
    return compute_time + memory_time

def simulate_decode_latency(gpu_name: str, seq_len: int, num_tokens: int) -> float:
    """Decode latency in seconds (memory-bound)"""
    gpu = GPUS[gpu_name]
    weights_bytes = MODEL_WEIGHTS_GB * 1024**3
    kv_bytes = seq_len * 0.125 * 1024**2  # 0.125 MB per token
    bytes_per_token = weights_bytes + kv_bytes
    time_per_token = bytes_per_token / (gpu['bandwidth_gb_s'] * 1e9)
    return time_per_token * num_tokens

def simulate_disaggregated(prefill_gpu: str, decode_gpu: str, input_tok: int, output_tok: int) -> dict:
    """Simulate prefill on one GPU and decode on another"""
    prefill_time = simulate_prefill_latency(prefill_gpu, input_tok)
    decode_time = simulate_decode_latency(decode_gpu, input_tok, output_tok)
    total = prefill_time + decode_time
    cost = (GPUS[prefill_gpu]['cost_per_hour'] * prefill_time / 3600 + 
            GPUS[decode_gpu]['cost_per_hour'] * decode_time / 3600)
    return {
        'prefill_ms': prefill_time * 1000,
        'decode_ms': decode_time * 1000,
        'total_ms': total * 1000,
        'cost': cost,
    }

if __name__ == '__main__':
    print("Experiment 5: Prefill vs Decode Disaggregation")
    print("=" * 60)
    
    workloads = [
        ("Short Q&A", 50, 30),
        ("Medium Essay", 500, 300),
        ("Long Document", 4000, 500),
        ("Code Generation", 200, 800),
    ]
    
    for name, inp, out in workloads:
        print(f"\n{name} ({inp}→{out} tokens):")
        for gpu in ['H100', 'A100', 'T4', 'RTX4060']:
            p = simulate_prefill_latency(gpu, inp) * 1000
            d = simulate_decode_latency(gpu, inp, out) * 1000
            print(f"  {gpu}: TTFT={p:7.1f}ms, Decode={d:8.1f}ms, Total={p+d:8.1f}ms")
    
    # Pareto analysis
    print("\n\nDisaggregated Serving Pareto Analysis (Medium Essay):")
    print(f"{'Prefill':<10} {'Decode':<10} {'Latency(ms)':<15} {'Cost($)':<12}")
    print("-" * 50)
    combinations = []
    for pg in ['H100', 'A100', 'T4']:
        for dg in ['H100', 'A100', 'T4', 'RTX4060']:
            r = simulate_disaggregated(pg, dg, 500, 300)
            combinations.append((pg, dg, r['total_ms'], r['cost']))
    
    combinations.sort(key=lambda x: x[2])
    for pg, dg, lat, cost in combinations[:10]:
        print(f"{pg:<10} {dg:<10} {lat:<15.1f} {cost:<12.6f}")
