"""
Experiment 6: Comprehensive Language-Resource Ecosystem Analysis
Combines speaker population, internet content, digital literacy proxy,
LLM benchmark performance, and tokenization fertility into a composite index.
"""

import numpy as np
import json
from scipy.stats import pearsonr

LANGUAGE_ECOSYSTEM = {
    'English': {
        'speakers_millions': 1500, 'internet_content_pct': 49.7,
        'digital_literacy_proxy': 92, 'llm_mmlu_accuracy': 85,
        'fertility': 1.15, 'wikipedia_users': 122038, 'resource_tier': 'High',
    },
    'Chinese': {
        'speakers_millions': 1200, 'internet_content_pct': 19.04,
        'digital_literacy_proxy': 82, 'llm_mmlu_accuracy': 78,
        'fertility': 0.73, 'wikipedia_users': 6994, 'resource_tier': 'High',
    },
    'Spanish': {
        'speakers_millions': 600, 'internet_content_pct': 7.70,
        'digital_literacy_proxy': 48, 'llm_mmlu_accuracy': 72,
        'fertility': 1.68, 'wikipedia_users': 14385, 'resource_tier': 'High',
    },
    'Arabic': {
        'speakers_millions': 420, 'internet_content_pct': 3.65,
        'digital_literacy_proxy': 68, 'llm_mmlu_accuracy': 65,
        'fertility': 2.40, 'wikipedia_users': 3930, 'resource_tier': 'Moderate',
    },
    'Hindi': {
        'speakers_millions': 600, 'internet_content_pct': 3.77,
        'digital_literacy_proxy': 90, 'llm_mmlu_accuracy': 55,
        'fertility': 2.83, 'wikipedia_users': 0, 'resource_tier': 'Moderate',
    },
    'Swahili': {
        'speakers_millions': 200, 'internet_content_pct': 0.0025,
        'digital_literacy_proxy': 38, 'llm_mmlu_accuracy': 45,
        'fertility': 2.34, 'wikipedia_users': 0, 'resource_tier': 'Low',
    },
    'Bengali': {
        'speakers_millions': 270, 'internet_content_pct': 0.2,
        'digital_literacy_proxy': 45, 'llm_mmlu_accuracy': 50,
        'fertility': 2.5, 'wikipedia_users': 0, 'resource_tier': 'Low',
    },
    'Russian': {
        'speakers_millions': 260, 'internet_content_pct': 3.75,
        'digital_literacy_proxy': 69, 'llm_mmlu_accuracy': 68,
        'fertility': 1.9, 'wikipedia_users': 9243, 'resource_tier': 'Moderate',
    },
    'Japanese': {
        'speakers_millions': 125, 'internet_content_pct': 2.23,
        'digital_literacy_proxy': 93, 'llm_mmlu_accuracy': 75,
        'fertility': 1.2, 'wikipedia_users': 12409, 'resource_tier': 'High',
    },
    'French': {
        'speakers_millions': 300, 'internet_content_pct': 3.42,
        'digital_literacy_proxy': 78, 'llm_mmlu_accuracy': 74,
        'fertility': 1.3, 'wikipedia_users': 17717, 'resource_tier': 'High',
    },
}

def calculate_crdi(data: dict) -> float:
    speakers_norm = 1 - (data['speakers_millions'] / 1500)
    content_norm = 1 - (data['internet_content_pct'] / 49.7)
    literacy_norm = 1 - (data['digital_literacy_proxy'] / 100)
    accuracy_norm = 1 - (data['llm_mmlu_accuracy'] / 100)
    fertility_norm = data['fertility'] / 3.0
    wiki_norm = 1 - min(data['wikipedia_users'] / 122038, 1.0)
    
    return (speakers_norm * 0.15 + content_norm * 0.25 + literacy_norm * 0.15 +
            accuracy_norm * 0.20 + fertility_norm * 0.15 + wiki_norm * 0.10)

if __name__ == '__main__':
    print("Experiment 6: Comprehensive Language Ecosystem Analysis")
    print("=" * 60)
    
    for lang, data in LANGUAGE_ECOSYSTEM.items():
        data['crdi'] = calculate_crdi(data)
        data['doi'] = 1 - data['crdi']
    
    print("\nComposite Resource Divide Index (CRDI) Ranking:")
    sorted_langs = sorted(LANGUAGE_ECOSYSTEM.items(), key=lambda x: x[1]['crdi'], reverse=True)
    for lang, data in sorted_langs:
        print(f"  {lang:<12} CRDI={data['crdi']:.3f}  DOI={data['doi']:.3f}  Tier={data['resource_tier']}")
    
    # Correlation analysis
    print("\n\nInter-Metric Correlations (Pearson r):")
    metrics = ['speakers_millions', 'internet_content_pct', 'digital_literacy_proxy',
               'llm_mmlu_accuracy', 'fertility', 'wikipedia_users']
    metric_labels = ['Speakers', 'Content%', 'Digital Lit.', 'LLM Acc.', 'Fertility', 'Wiki Users']
    
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i < j:
                x = [LANGUAGE_ECOSYSTEM[l][m1] for l in LANGUAGE_ECOSYSTEM]
                y = [LANGUAGE_ECOSYSTEM[l][m2] for l in LANGUAGE_ECOSYSTEM]
                r, _ = pearsonr(x, y)
                print(f"  {metric_labels[i]} vs {metric_labels[j]}: r={r:.3f}")
    
    # Save
    with open('data/experiment_6_results.json', 'w') as f:
        json.dump(LANGUAGE_ECOSYSTEM, f, indent=2)
    print("\nSaved to data/experiment_6_results.json")
