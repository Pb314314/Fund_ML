#!/usr/bin/env python3
"""
Experiment 6: Comprehensive Language-Resource Ecosystem Analysis
=================================================================

Combines six metrics into a Composite Resource Divide Index (CRDI):
  1. Speaker population (millions, L1+L2)
  2. Internet content percentage (W3Techs 2024)
  3. Digital literacy proxy (% with basic digital skills)
  4. LLM benchmark accuracy (MMLU proxy)
  5. Tokenization fertility (tokens/word, from Exp1)
  6. Wikipedia contributor activity (proxy for digital participation)

CRDI Formula:
-------------
Each metric is normalized to [0, 1] where 1 = worst (most disadvantaged).
CRDI = weighted sum:
  - Speakers: 0.10  (inverse normalized: fewer speakers = worse)
  - Content: 0.25   (inverse normalized: less content = worse)
  - Literacy: 0.15  (inverse normalized: lower literacy = worse)
  - LLM Acc: 0.20   (inverse normalized: lower accuracy = worse)
  - Fertility: 0.15 (direct normalized: higher fertility = worse)
  - Wikipedia: 0.15 (inverse normalized: fewer contributors = worse)

Weights rationale:
  - Content (0.25): Most direct predictor of LLM training data availability
  - LLM Acc (0.20): Direct performance metric
  - Literacy/Fertility/Wikipedia (0.15 each): Important but secondary
  - Speakers (0.10): Population alone doesn't guarantee resource access

DOI (Digital Opportunity Index) = 1 - CRDI

Author: Student Researcher
Date: 2025-04-25
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from config import LANGUAGE_ECOSYSTEM, LANGUAGE_FERTILITY, MODEL

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

# Weights with explicit rationale
WEIGHTS = {
    'speakers': 0.10,    # Population potential (inverse)
    'content': 0.25,     # Training data availability (inverse)
    'literacy': 0.15,    # Digital access capability (inverse)
    'accuracy': 0.20,    # Current LLM performance (inverse)
    'fertility': 0.15,   # Inference cost multiplier (direct)
    'wikipedia': 0.15,   # Community digital participation (inverse)
}


def calculate_crdi(data: dict) -> tuple:
    """
    Calculate Composite Resource Divide Index and components.
    Returns (crdi, doi, components_dict).
    """
    # Normalize each metric
    speakers_norm = 1 - (data['speakers_millions'] / 1500.0)
    content_norm = 1 - (data['internet_content_pct'] / 49.7)
    literacy_norm = 1 - (data['digital_literacy_proxy'] / 100.0)
    accuracy_norm = 1 - (data['llm_mmlu_accuracy'] / 100.0)
    fertility_norm = data.get('fertility', 1.5) / 3.0
    wiki_norm = 1 - min(data['wikipedia_users'] / 122038.0, 1.0)
    
    components = {
        'speakers': round(speakers_norm, 4),
        'content': round(content_norm, 4),
        'literacy': round(literacy_norm, 4),
        'accuracy': round(accuracy_norm, 4),
        'fertility': round(fertility_norm, 4),
        'wikipedia': round(wiki_norm, 4),
    }
    
    crdi = (components['speakers'] * WEIGHTS['speakers'] +
            components['content'] * WEIGHTS['content'] +
            components['literacy'] * WEIGHTS['literacy'] +
            components['accuracy'] * WEIGHTS['accuracy'] +
            components['fertility'] * WEIGHTS['fertility'] +
            components['wikipedia'] * WEIGHTS['wikipedia'])
    
    doi = 1 - crdi
    
    return round(crdi, 4), round(doi, 4), components


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figure_6(output_dir: str, ecosystem_with_crdi: dict) -> str:
    """Figure 6: Comprehensive 6-panel ecosystem dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    langs = list(ecosystem_with_crdi.keys())
    
    # (a) Speaker Population vs Internet Content
    ax1 = fig.add_subplot(gs[0, 0])
    for lang in langs:
        data = ecosystem_with_crdi[lang]
        tier = data['resource_tier']
        color = '#1B5E20' if tier == 'High' else '#F57C00' if tier == 'Moderate' else '#C62828'
        ax1.scatter(data['speakers_millions'], data['internet_content_pct'],
                   s=300, color=color, alpha=0.7, edgecolor='black', linewidth=1, zorder=5)
        ax1.annotate(lang, (data['speakers_millions'], data['internet_content_pct']),
                    textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Speakers (Millions, L1+L2)', fontweight='bold')
    ax1.set_ylabel('Internet Content (%)', fontweight='bold')
    ax1.set_title('(a) Speaker Population vs Digital Content', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1B5E20', label='High Resource'),
                       Patch(facecolor='#F57C00', label='Moderate Resource'),
                       Patch(facecolor='#C62828', label='Low Resource')]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # (b) LLM Performance vs Fertility
    ax2 = fig.add_subplot(gs[0, 1])
    for lang in langs:
        data = ecosystem_with_crdi[lang]
        tier = data['resource_tier']
        color = '#1B5E20' if tier == 'High' else '#F57C00' if tier == 'Moderate' else '#C62828'
        ax2.scatter(data['fertility'], data['llm_mmlu_accuracy'],
                   s=300, color=color, alpha=0.7, edgecolor='black', linewidth=1, zorder=5)
        ax2.annotate(lang, (data['fertility'], data['llm_mmlu_accuracy']),
                    textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Tokenization Fertility (Tokens/Word)', fontweight='bold')
    ax2.set_ylabel('Estimated LLM MMLU Accuracy (%)', fontweight='bold')
    ax2.set_title('(b) Tokenization Cost vs LLM Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # (c) CRDI Ranking
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_langs = sorted(ecosystem_with_crdi.items(), key=lambda x: x[1]['crdi'], reverse=True)
    crdi_vals = [data['crdi'] for _, data in sorted_langs]
    lang_names = [lang for lang, _ in sorted_langs]
    colors_bar = ['#1B5E20' if ecosystem_with_crdi[l]['resource_tier'] == 'High'
                  else '#F57C00' if ecosystem_with_crdi[l]['resource_tier'] == 'Moderate'
                  else '#C62828' for l in lang_names]
    
    bars = ax3.barh(range(len(lang_names)), crdi_vals, color=colors_bar, alpha=0.8, ec='black')
    ax3.set_yticks(range(len(lang_names)))
    ax3.set_yticklabels(lang_names, fontsize=11)
    ax3.set_xlabel('Composite Resource Divide Index (CRDI)', fontweight='bold')
    ax3.set_title('(c) CRDI Ranking (Higher = Worse)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, crdi_vals)):
        ax3.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # (d) Radar chart
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    categories = ['Content\nVolume', 'Digital\nLiteracy', 'LLM\nAccuracy', 'Low\nFertility', 'Wikipedia\nActivity']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    selected = ['English', 'Chinese', 'Hindi', 'Swahili']
    colors_radar = ['#1565C0', '#C62828', '#F57C00', '#2E7D32']
    
    for lang, color in zip(selected, colors_radar):
        data = ecosystem_with_crdi[lang]
        values = [
            1 - data['internet_content_pct'] / 49.7,
            1 - data['digital_literacy_proxy'] / 100.0,
            1 - data['llm_mmlu_accuracy'] / 100.0,
            data['fertility'] / 3.0,
            1 - min(data['wikipedia_users'] / 122038.0, 1.0),
        ]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=lang, color=color, alpha=0.7)
        ax4.fill(angles, values, alpha=0.15, color=color)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('(d) Multi-Dimensional Disparity\n(Further from center = worse)', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # (e) Correlation heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['speakers_millions', 'internet_content_pct', 'digital_literacy_proxy',
               'llm_mmlu_accuracy', 'fertility', 'wikipedia_users']
    metric_labels = ['Speakers', 'Content%', 'Digital Lit.', 'LLM Acc.', 'Fertility', 'Wiki Users']
    
    n = len(metrics)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = [ecosystem_with_crdi[l][metrics[i]] for l in langs]
            y = [ecosystem_with_crdi[l][metrics[j]] for l in langs]
            corr, _ = pearsonr(x, y)
            corr_matrix[i, j] = corr
    
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xticks(range(n))
    ax5.set_yticks(range(n))
    ax5.set_xticklabels(metric_labels, fontsize=10, rotation=45, ha='right')
    ax5.set_yticklabels(metric_labels, fontsize=10)
    ax5.set_title('(e) Inter-Metric Correlations\n(Pearson r)', fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax5.text(j, i, f'{val:.2f}', ha='center', va='center', color=color,
                    fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax5, shrink=0.8)
    
    # (f) Resource tier distribution pie
    ax6 = fig.add_subplot(gs[1, 2])
    tier_counts = {}
    for lang in langs:
        tier = ecosystem_with_crdi[lang]['resource_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    tier_colors = {'High': '#1B5E20', 'Moderate': '#F57C00', 'Low': '#C62828'}
    labels = list(tier_counts.keys())
    sizes = list(tier_counts.values())
    colors_pie = [tier_colors[l] for l in labels]
    explode = [0.05 if l == 'Low' else 0 for l in labels]
    
    wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors_pie,
                                        autopct='%1.0f%%', explode=explode,
                                        startangle=90, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax6.set_title('(f) Languages by Resource Tier\n(Sample of 10 Languages)', fontweight='bold')
    
    plt.suptitle('Figure 6: Comprehensive Language-Resource Ecosystem Analysis\n'
                 'Global AI Readiness Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    out_path = os.path.join(output_dir, "fig6_language_ecosystem.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


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
    
    print("Experiment 6: Language Ecosystem Analysis")
    print(f"  Metrics: {list(WEIGHTS.keys())}")
    print(f"  Weights: {WEIGHTS}")
    print()
    
    # Build ecosystem data with fertility and CRDI
    ecosystem_with_crdi = {}
    for lang, data in LANGUAGE_ECOSYSTEM.items():
        ecosystem_with_crdi[lang] = {
            'speakers_millions': data.speakers_millions,
            'internet_content_pct': data.internet_content_pct,
            'digital_literacy_proxy': data.digital_literacy_proxy,
            'llm_mmlu_accuracy': data.llm_mmlu_accuracy,
            'wikipedia_users': data.wikipedia_users,
            'resource_tier': data.resource_tier,
            'fertility': LANGUAGE_FERTILITY.get(lang, 1.5),
            'sources': data.sources,
        }
        
        crdi, doi, components = calculate_crdi(ecosystem_with_crdi[lang])
        ecosystem_with_crdi[lang]['crdi'] = crdi
        ecosystem_with_crdi[lang]['doi'] = doi
        ecosystem_with_crdi[lang]['components'] = components
    
    # Print CRDI ranking
    print("CRDI Ranking:")
    sorted_langs = sorted(ecosystem_with_crdi.items(), key=lambda x: x[1]['crdi'], reverse=True)
    for lang, data in sorted_langs:
        print(f"  {lang:<12} CRDI={data['crdi']:.3f}  DOI={data['doi']:.3f}  Tier={data['resource_tier']}")
    
    # Print correlations
    print("\nInter-Metric Correlations (Pearson r):")
    metrics = ['speakers_millions', 'internet_content_pct', 'digital_literacy_proxy',
               'llm_mmlu_accuracy', 'fertility', 'wikipedia_users']
    metric_labels = ['Speakers', 'Content%', 'Digital Lit.', 'LLM Acc.', 'Fertility', 'Wiki Users']
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i < j:
                x = [ecosystem_with_crdi[l][m1] for l in ecosystem_with_crdi]
                y = [ecosystem_with_crdi[l][m2] for l in ecosystem_with_crdi]
                r, _ = pearsonr(x, y)
                print(f"  {metric_labels[i]} vs {metric_labels[j]}: r={r:.3f}")
    
    print("\nGenerating figure...")
    generate_figure_6(figures_dir, ecosystem_with_crdi)
    print("  Saved: fig6_language_ecosystem.png")
    
    data_path = os.path.join(data_dir, "experiment_6_results.json")
    with open(data_path, 'w') as f:
        json.dump(ecosystem_with_crdi, f, indent=2)
    print(f"\n  Saved data: {data_path}")
    
    print("\n" + "=" * 60)
    print("Experiment 6: Ecosystem Analysis — Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
