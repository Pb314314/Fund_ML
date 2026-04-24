"""
Experiment 4: Multi-turn Conversational KV Cache Accumulation + Throughput/Concurrency Simulation
Simulates agentic workloads where each conversation turn accumulates KV cache,
affecting throughput and maximum concurrent users on different hardware tiers.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Dict

# Hardware specs
@dataclass
class GPU:
    name: str
    vram_gb: float
    bandwidth_gb_s: float
    compute_tflops: float

# Model config: Llama-3-8B with 4-bit AWQ
MODEL_WEIGHTS_GB = 5.5
KV_PER_TOKEN_MB = 0.125  # FP16
OVERHEAD_PCT = 0.10

LANGUAGES = {
    'English': {'fertility': 1.15, 'words_per_turn': 80},
    'Chinese': {'fertility': 0.73, 'words_per_turn': 80},
    'Spanish': {'fertility': 1.68, 'words_per_turn': 80},
    'Arabic': {'fertility': 2.40, 'words_per_turn': 80},
    'Swahili': {'fertility': 2.34, 'words_per_turn': 80},
    'Hindi': {'fertility': 2.83, 'words_per_turn': 80},
}

GPUS = {
    'H100': GPU('H100 SXM5', 80, 3350, 989),
    'A100': GPU('A100 80GB', 80, 2039, 312),
    'T4': GPU('T4 16GB', 16, 320, 65),
    'RTX4060': GPU('RTX 4060', 8, 272, 15.11),
}

def get_available_kv_memory(gpu: GPU) -> float:
    usable = gpu.vram_gb * (1 - OVERHEAD_PCT)
    return usable - MODEL_WEIGHTS_GB

def simulate_conversation_turns(gpu_name: str, language: str, num_turns: int = 10,
                                 words_per_turn: int = 80, batch_size: int = 32) -> Dict:
    """Simulate multi-turn conversation serving scenario."""
    gpu = GPUS[gpu_name]
    lang = LANGUAGES[language]
    fertility = lang['fertility']
    
    available_kv_gb = get_available_kv_memory(gpu)
    if available_kv_gb <= 0:
        return {'error': f'GPU {gpu_name} cannot fit model weights'}
    
    results = {'gpu': gpu_name, 'language': language, 'batch_size': batch_size,
               'available_kv_gb': available_kv_gb, 'turns': [], 'oom_turn': None}
    
    cumulative_tokens_per_seq = 0
    for turn in range(1, num_turns + 1):
        new_tokens = int(words_per_turn * fertility)
        total_new_tokens = new_tokens * 2  # user + response
        cumulative_tokens_per_seq += total_new_tokens
        
        total_kv_gb = cumulative_tokens_per_seq * batch_size * KV_PER_TOKEN_MB / 1024
        oom = total_kv_gb > available_kv_gb
        
        # Decode throughput (memory-bound)
        model_size_bytes = MODEL_WEIGHTS_GB * 1024**3
        weights_time = model_size_bytes / (gpu.bandwidth_gb_s * 1e9)
        kv_time = (total_kv_gb * 1024**3) / (gpu.bandwidth_gb_s * 1e9)
        time_per_token = weights_time + kv_time
        throughput = batch_size / time_per_token if time_per_token > 0 else 0
        
        results['turns'].append({
            'turn': turn,
            'cumulative_tokens_per_seq': cumulative_tokens_per_seq,
            'total_kv_gb': total_kv_gb,
            'time_per_token_ms': time_per_token * 1000,
            'throughput_tok_s': throughput,
            'oom': oom,
        })
        
        if oom and results['oom_turn'] is None:
            results['oom_turn'] = turn
    
    return results

if __name__ == '__main__':
    # Run key combinations
    print("Experiment 4: Multi-turn Conversation Simulation (batch_size=32)")
    print("=" * 60)
    for gpu in ['H100', 'A100', 'T4', 'RTX4060']:
        for lang in ['English', 'Hindi']:
            res = simulate_conversation_turns(gpu, lang, num_turns=20, batch_size=32)
            if 'error' not in res:
                oom = res['oom_turn']
                last = res['turns'][-1]
                print(f"{gpu}+{lang}: OOM@turn{oom if oom else 'Never'}, "
                      f"Final KV={last['total_kv_gb']:.1f}GB")
    
    # Save results
    all_results = {}
    for gpu in ['H100', 'A100', 'T4', 'RTX4060']:
        for lang in ['English', 'Chinese', 'Spanish', 'Arabic', 'Swahili', 'Hindi']:
            key = f"{gpu}_{lang}"
            all_results[key] = simulate_conversation_turns(gpu, lang, num_turns=20, batch_size=32)
    
    with open('data/experiment_4_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to data/experiment_4_results.json")
