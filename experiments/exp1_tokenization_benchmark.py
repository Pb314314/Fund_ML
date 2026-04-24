#!/usr/bin/env python3
"""
Experiment 1: Tokenization Fertility Benchmark
===============================================

Benchmarks tokenization fertility across languages using the Llama-3 tokenizer.
Fertility = tokens per word (higher = less efficient for that language).

Metrics collected per language:
- tokens: number of subword tokens produced
- words: number of space-separated words
- chars: number of characters
- fertility (tokens_per_word): tokens / words
- tokens_per_char: tokens / chars
- attention_cost_multiplier: fertility^2 (attention is O(n^2))

Output:
- Figure: fig1_tokenization_fertility.png
- Data: tokenization_results.json

Author: ML Systems Engineer
Date: 2024
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. Try to import and load tokenizer
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install -q transformers")
    from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------

# Try multiple Llama-3 tokenizer identifiers (public + gated fallbacks)
# Local cache path takes priority for environments with limited internet access
TOKENIZER_CANDIDATES: List[str] = [
    "/mnt/agents/output/project/tokenizer_cache/nous_llama3",  # local cache (preferred)
    "meta-llama/Llama-3.2-1B",     # preferred (public)
    "meta-llama/Llama-3.1-8B",     # gated, may need token
    "meta-llama/Meta-Llama-3-8B",  # gated
    "meta-llama/Llama-3.2-1B-Instruct",
]

# Define languages with metadata and sample texts
LANGUAGE_SAMPLES: Dict[str, Dict] = {
    "English": {
        "script": "Latin",
        "resource_level": "high",
        "rtl": False,
        "has_spaces": True,
        "text": (
            "The quick brown fox jumps over the lazy dog. "
            "Artificial intelligence is transforming how we work and live. "
            "Large language models require significant computational "
            "resources to train and deploy effectively. "
            "Natural language processing enables machines to understand "
            "human communication in context. "
            "Deep learning architectures have revolutionized speech "
            "recognition, computer vision, and autonomous systems."
        ),
    },
    "Spanish": {
        "script": "Latin",
        "resource_level": "moderate",
        "rtl": False,
        "has_spaces": True,
        "text": (
            "La inteligencia artificial está transformando la forma en que "
            "trabajamos y vivimos. Los modelos de lenguaje grande requieren "
            "recursos computacionales significativos para entrenarse y "
            "desplegarse eficazmente. El aprendizaje automático necesita "
            "datos de alta calidad. "
            "El procesamiento del lenguaje natural permite que las máquinas "
            "comprendan la comunicación humana. "
            "Las arquitecturas de aprendizaje profundo han revolucionado "
            "el reconocimiento de voz y la visión artificial."
        ),
    },
    "Swahili": {
        "script": "Latin",
        "resource_level": "low",
        "rtl": False,
        "has_spaces": True,
        "text": (
            "Haraka haraka haina baraka. Teknolojia ya akili ya bandia "
            "inabadilisha jinsi watu wanavyofanya kazi na kuishi. "
            "Mifano mikubwa ya lugha inahitaji rasilimali nyingi za hesabu "
            "kwa ajili ya mafunzo na utekelezaji. "
            "Akili ya mashine inahitaji data ya hali ya juu ili kufanya "
            "kazi vizuri. "
            "Lugha ya kompyuta inasaidia mashine kuelewa mawasiliano ya "
            "binadamu katika muktadha wake."
        ),
    },
    "Arabic": {
        "script": "Arabic",
        "resource_level": "moderate",
        "rtl": True,
        "has_spaces": True,
        "text": (
            "الذكاء الاصطناعي يغير طريقة عملنا وحياتنا. "
            "النماذج اللغوية الكبيرة تحتاج إلى موارد حاسوبية ضخمة. "
            "التعلم الآلي يتطلب بيانات عالية الجودة حتى يعمل بفعالية. "
            "معالجة اللغات الطبيعية تتيح للآلات فهم التواصل البشري في "
            "سياقه. "
            "تعلم الآلة العميق قد أحدث ثورة في التعرف على الكلام والرؤية "
            "الحاسوبية."
        ),
    },
    "Chinese": {
        "script": "Han (Logographic)",
        "resource_level": "high",
        "rtl": False,
        "has_spaces": False,
        "text": (
            "人工智能正在改变我们的工作和生活方式。"
            "大型语言模型需要大量计算资源来训练和部署。"
            "机器学习需要高质量的数据才能有效工作。"
            "自然语言处理使机器能够理解人类在上下文中的交流。"
            "深度学习架构彻底改变了语音识别、计算机视觉和自主系统。"
        ),
    },
    "Hindi": {
        "script": "Devanagari",
        "resource_level": "moderate",
        "rtl": False,
        "has_spaces": True,
        "text": (
            "कृत्रिम बुद्धिमत्ता हमारे काम और जीवन को बदल रही है। "
            "बड़े भाषा मॉडलों को प्रशिक्षित करने के लिए बहुत अधिक "
            "संगणन संसाधनों की आवश्यकता होती है। "
            "मशीन लर्निंग को प्रभावी रूप से काम करने के लिए उच्च गुणवत्ता "
            "वाले डेटा की आवश्यकता होती है। "
            "प्राकृतिक भाषा प्रसंस्करण मशीनों को मानव संचार को संदर्भ में "
            "समझने की अनुमति देता है। "
            "डीप लर्निंग आर्किटेक्चर ने वाक पहचान, कंप्यूटर दृष्टि और "
            "स्वायत्त प्रणालियों में क्रांति ला दी है।"
        ),
    },
}


# ---------------------------------------------------------------------------
# 3. Data class for results
# ---------------------------------------------------------------------------

@dataclass
class LanguageResult:
    language: str
    script: str
    resource_level: str
    rtl: bool
    has_spaces: bool
    text: str
    char_count: int
    word_count: int
    token_count: int
    tokens: List[int]
    token_strings: List[str]
    fertility: float          # tokens per word
    tokens_per_char: float    # tokens per character
    attention_cost_multiplier: float  # fertility^2


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def load_tokenizer(candidates: List[str]):
    """Attempt to load the first available tokenizer from candidates."""
    last_err = None
    for model_name in candidates:
        try:
            print(f"  Trying tokenizer: {model_name} ...")
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"  ✓ Loaded tokenizer from {model_name}")
            return tok, model_name
        except Exception as e:
            last_err = e
            print(f"  ✗ Failed: {e}")
    raise RuntimeError(
        f"Could not load any tokenizer. Last error: {last_err}"
    )


def count_words(text: str, has_spaces: bool, language: str) -> int:
    """
    Count words in text.
    For space-separated languages, split on whitespace.
    For Chinese (no spaces), each character is roughly a word/morpheme.
    """
    if has_spaces:
        # Strip punctuation-like whitespace handling
        words = text.split()
        return len(words)
    else:
        # Chinese: characters are semantic units; count non-space chars
        return len(text.replace(" ", "").replace("。", "").replace("，", "").replace("、", "").replace("！", "").replace("？", ""))


def tokenize_and_analyze(tokenizer, language: str, meta: Dict) -> LanguageResult:
    """Tokenize sample text and compute all metrics."""
    text = meta["text"]
    
    # Encode
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    
    # Counts
    char_count = len(text)
    word_count = count_words(text, meta["has_spaces"], language)
    token_count = len(tokens)
    
    # Metrics
    fertility = token_count / word_count if word_count > 0 else 0.0
    tokens_per_char = token_count / char_count if char_count > 0 else 0.0
    attention_cost_multiplier = fertility ** 2
    
    return LanguageResult(
        language=language,
        script=meta["script"],
        resource_level=meta["resource_level"],
        rtl=meta["rtl"],
        has_spaces=meta["has_spaces"],
        text=text,
        char_count=char_count,
        word_count=word_count,
        token_count=token_count,
        tokens=tokens,
        token_strings=token_strings,
        fertility=fertility,
        tokens_per_char=tokens_per_char,
        attention_cost_multiplier=attention_cost_multiplier,
    )


def results_to_dict(results: List[LanguageResult]) -> Dict:
    """Convert results to a JSON-serializable dictionary."""
    data = {
        "metadata": {
            "tokenizer": "",
            "description": "Tokenization fertility benchmark across languages",
            "metrics": [
                "char_count",
                "word_count",
                "token_count",
                "fertility (tokens_per_word)",
                "tokens_per_char",
                "attention_cost_multiplier (fertility^2)",
            ],
        },
        "languages": {},
    }
    for r in results:
        data["languages"][r.language] = {
            "script": r.script,
            "resource_level": r.resource_level,
            "rtl": r.rtl,
            "has_spaces": r.has_spaces,
            "char_count": r.char_count,
            "word_count": r.word_count,
            "token_count": r.token_count,
            "fertility": round(r.fertility, 4),
            "tokens_per_char": round(r.tokens_per_char, 4),
            "attention_cost_multiplier": round(r.attention_cost_multiplier, 4),
            "sample_text_preview": r.text[:120] + "...",
            "token_strings_preview": r.token_strings[:20],
        }
    return data


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_results(results: List[LanguageResult], tokenizer_name: str, save_path: str):
    """Generate publication-quality figure with two subplots."""
    
    # Sort by fertility for consistent visual ordering
    results_sorted = sorted(results, key=lambda r: r.fertility)
    languages = [r.language for r in results_sorted]
    fertilities = [r.fertility for r in results_sorted]
    attention_costs = [r.attention_cost_multiplier for r in results_sorted]
    resource_levels = [r.resource_level for r in results_sorted]
    
    # Color map by resource level
    color_map = {
        "high": "#2E7D32",      # green
        "moderate": "#F9A825",   # amber
        "low": "#C62828",        # red
    }
    colors = [color_map.get(r.resource_level, "#757575") for r in results_sorted]
    
    # Create figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("white")
    
    # --- Subplot 1: Fertility (tokens per word) ---
    ax1 = axes[0]
    bars1 = ax1.bar(languages, fertilities, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Tokens per Word (Fertility)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Tokenization Fertility by Language", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(0, max(fertilities) * 1.2)
    ax1.tick_params(axis="x", rotation=30, labelsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, val in zip(bars1, fertilities):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    # --- Subplot 2: Attention Cost Multiplier (fertility^2) ---
    ax2 = axes[1]
    bars2 = ax2.bar(languages, attention_costs, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Attention Cost Multiplier (Fertility²)", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Attention Cost Multiplier by Language", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(attention_costs) * 1.2)
    ax2.tick_params(axis="x", rotation=30, labelsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, val in zip(bars2, attention_costs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    # Legend for resource levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E7D32", edgecolor="black", label="High-resource"),
        Patch(facecolor="#F9A825", edgecolor="black", label="Moderate-resource"),
        Patch(facecolor="#C62828", edgecolor="black", label="Low-resource"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    
    # Overall title
    fig.suptitle(
        f"Tokenization Fertility Benchmark — {tokenizer_name}",
        fontsize=14, fontweight="bold", y=1.08,
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Figure saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 1: Tokenization Fertility Benchmark")
    print("=" * 70)
    
    # --- Paths ---
    base_dir = Path("/mnt/agents/output/project")
    fig_path = base_dir / "figures" / "fig1_tokenization_fertility.png"
    data_path = base_dir / "data" / "tokenization_results.json"
    script_path = base_dir / "experiments" / "exp1_tokenization_benchmark.py"
    
    # Ensure directories exist
    for d in [fig_path.parent, data_path.parent]:
        d.mkdir(parents=True, exist_ok=True)
    
    # --- Load tokenizer ---
    print("\n[1/4] Loading tokenizer...")
    tokenizer, tokenizer_name = load_tokenizer(TOKENIZER_CANDIDATES)
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size:,}")
    
    # --- Run benchmark ---
    print("\n[2/4] Benchmarking languages...")
    results: List[LanguageResult] = []
    
    for language, meta in LANGUAGE_SAMPLES.items():
        print(f"\n  → {language} ({meta['script']}, {meta['resource_level']}-resource)")
        r = tokenize_and_analyze(tokenizer, language, meta)
        results.append(r)
        print(f"    Chars: {r.char_count} | Words: {r.word_count} | Tokens: {r.token_count}")
        print(f"    Fertility: {r.fertility:.3f} | Tok/Char: {r.tokens_per_char:.3f} | Attn²: {r.attention_cost_multiplier:.3f}")
        # Show first few token strings for insight
        preview = r.token_strings[:15]
        print(f"    Tokens preview: {preview}")
    
    # --- Save JSON ---
    print("\n[3/4] Saving raw data...")
    data_dict = results_to_dict(results)
    data_dict["metadata"]["tokenizer"] = tokenizer_name
    
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Data saved to {data_path}")
    
    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Language':<12} {'Script':<18} {'Tokens':>8} {'Words':>8} {'Fertility':>10} {'Attn²':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r.language:<12} {r.script:<18} {r.token_count:>8} {r.word_count:>8} "
            f"{r.fertility:>10.3f} {r.attention_cost_multiplier:>10.3f}"
        )
    print("-" * 70)
    
    # --- Generate figure ---
    print("\n[4/4] Generating figure...")
    plot_results(results, tokenizer_name, str(fig_path))
    
    print("\n" + "=" * 70)
    print("Experiment 1 complete!")
    print(f"  Script : {script_path}")
    print(f"  Figure : {fig_path}")
    print(f"  Data   : {data_path}")
    print("=" * 70)
    
    return results, data_dict


if __name__ == "__main__":
    main()
