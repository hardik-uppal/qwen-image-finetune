#!/usr/bin/env python3
"""
Analyze training data quality for prompt generation.

Identifies issues like:
- Vague prompts (no specific values)
- Short prompts (< 50 words)
- Missing categories
- Lack of structure
- Generic language

Usage:
    python script/analyze_training_data_quality.py \
        --csv workspace/metadata-v12-recaptioned-long/train_filtered_2k.csv \
        --output results/data_analysis.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


def has_numerical_values(text: str) -> bool:
    """Check if prompt contains specific numerical values."""
    # Look for patterns like: +1.5, -20%, 150K, etc.
    patterns = [
        r'[+-]?\d+\.?\d*\s*(?:EV|%|K|degrees?|stops?)',
        r'[+-]?\d+\.?\d*',
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def has_structure(text: str) -> bool:
    """Check if prompt has clear structure (sections, steps, bullets)."""
    structure_indicators = [
        r'STEP \d+',
        r'^\d+\.',
        r'^-\s',
        r'[A-Z\s]{5,}:',  # Section headers like "COLOR GRADING:"
    ]
    return any(re.search(pattern, text, re.MULTILINE) for pattern in structure_indicators)


def mentions_tools(text: str) -> bool:
    """Check if prompt mentions specific editing tools."""
    tools = [
        'lightroom', 'photoshop', 'curves', 'hsl', 'levels',
        'brush', 'gradient', 'mask', 'layer', 'adjustment',
        'slider', 'panel', 'filter'
    ]
    text_lower = text.lower()
    return any(tool in text_lower for tool in tools)


def has_categories(text: str) -> bool:
    """Check if prompt has category tags."""
    categories = [
        'lighting', 'exposure', 'color', 'composition',
        'crop', 'white balance', 'saturation', 'contrast',
        'sharpness', 'noise', 'lens', 'distortion'
    ]
    text_lower = text.lower()
    return sum(cat in text_lower for cat in categories) >= 2


def is_generic(text: str) -> bool:
    """Check if prompt uses generic/vague language."""
    generic_phrases = [
        'make better', 'improve', 'enhance', 'fix',
        'adjust', 'change', 'modify', 'update',
        'brighten', 'darken', 'more', 'less'
    ]
    # Generic if contains these WITHOUT specific values
    text_lower = text.lower()
    generic_count = sum(phrase in text_lower for phrase in generic_phrases)
    return generic_count > 3 and not has_numerical_values(text)


def analyze_prompt(prompt: str) -> Dict:
    """Analyze a single prompt for quality indicators."""
    word_count = len(prompt.split())
    
    return {
        'word_count': word_count,
        'has_numbers': has_numerical_values(prompt),
        'has_structure': has_structure(prompt),
        'mentions_tools': mentions_tools(prompt),
        'has_categories': has_categories(prompt),
        'is_generic': is_generic(prompt),
        'is_short': word_count < 50,
        'is_long_enough': word_count >= 100,
    }


def analyze_dataset(csv_path: str) -> Dict:
    """Analyze entire dataset."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Analyzing {len(df)} prompts...")
    analyses = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row.get('prompt', '')
        if pd.isna(prompt) or not prompt:
            analyses.append({
                'word_count': 0,
                'has_numbers': False,
                'has_structure': False,
                'mentions_tools': False,
                'has_categories': False,
                'is_generic': True,
                'is_short': True,
                'is_long_enough': False,
            })
        else:
            analyses.append(analyze_prompt(prompt))
    
    # Calculate statistics
    stats = {
        'total_samples': len(df),
        'avg_word_count': sum(a['word_count'] for a in analyses) / len(analyses),
        'pct_has_numbers': sum(a['has_numbers'] for a in analyses) / len(analyses) * 100,
        'pct_has_structure': sum(a['has_structure'] for a in analyses) / len(analyses) * 100,
        'pct_mentions_tools': sum(a['mentions_tools'] for a in analyses) / len(analyses) * 100,
        'pct_has_categories': sum(a['has_categories'] for a in analyses) / len(analyses) * 100,
        'pct_is_generic': sum(a['is_generic'] for a in analyses) / len(analyses) * 100,
        'pct_is_short': sum(a['is_short'] for a in analyses) / len(analyses) * 100,
        'pct_is_long_enough': sum(a['is_long_enough'] for a in analyses) / len(analyses) * 100,
    }
    
    # Find problem examples
    problem_indices = {
        'generic': [i for i, a in enumerate(analyses) if a['is_generic']],
        'short': [i for i, a in enumerate(analyses) if a['is_short']],
        'no_numbers': [i for i, a in enumerate(analyses) if not a['has_numbers']],
        'no_structure': [i for i, a in enumerate(analyses) if not a['has_structure']],
    }
    
    return {
        'stats': stats,
        'problem_counts': {k: len(v) for k, v in problem_indices.items()},
        'problem_indices': problem_indices,
        'sample_problems': {
            'generic': df.iloc[problem_indices['generic'][:5]]['prompt'].tolist() if problem_indices['generic'] else [],
            'short': df.iloc[problem_indices['short'][:5]]['prompt'].tolist() if problem_indices['short'] else [],
        }
    }


def display_report(analysis: Dict):
    """Display analysis report."""
    stats = analysis['stats']
    problems = analysis['problem_counts']
    
    print("\n" + "=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 DATASET SIZE: {stats['total_samples']:,} samples")
    print(f"📝 AVERAGE PROMPT LENGTH: {stats['avg_word_count']:.1f} words")
    
    print("\n" + "-" * 80)
    print("QUALITY INDICATORS")
    print("-" * 80)
    
    indicators = [
        ('Has Numerical Values', stats['pct_has_numbers'], 70),
        ('Has Structure (steps/bullets)', stats['pct_has_structure'], 60),
        ('Mentions Tools', stats['pct_mentions_tools'], 30),
        ('Has Multiple Categories', stats['pct_has_categories'], 80),
        ('Long Enough (100+ words)', stats['pct_is_long_enough'], 50),
    ]
    
    for name, pct, target in indicators:
        status = "✅" if pct >= target else "❌"
        print(f"{status} {name:<35} {pct:>5.1f}% (target: {target}%)")
    
    print("\n" + "-" * 80)
    print("PROBLEMS DETECTED")
    print("-" * 80)
    
    problem_types = [
        ('Generic/Vague Language', stats['pct_is_generic'], problems['generic']),
        ('Too Short (< 50 words)', stats['pct_is_short'], problems['short']),
        ('No Numerical Values', 100 - stats['pct_has_numbers'], problems['no_numbers']),
        ('No Structure', 100 - stats['pct_has_structure'], problems['no_structure']),
    ]
    
    for name, pct, count in problem_types:
        severity = "🔴" if pct > 50 else "🟡" if pct > 20 else "🟢"
        print(f"{severity} {name:<35} {pct:>5.1f}% ({count:,} samples)")
    
    # Overall score
    print("\n" + "=" * 80)
    quality_score = (
        stats['pct_has_numbers'] * 0.3 +
        stats['pct_has_structure'] * 0.2 +
        stats['pct_mentions_tools'] * 0.1 +
        stats['pct_has_categories'] * 0.2 +
        stats['pct_is_long_enough'] * 0.2
    )
    
    if quality_score >= 70:
        grade = "🟢 EXCELLENT"
        recommendation = "Data quality is good! You may still improve metrics by fine-tuning hyperparameters."
    elif quality_score >= 50:
        grade = "🟡 MODERATE"
        recommendation = "Data needs improvement. Focus on adding numerical values and structure."
    else:
        grade = "🔴 POOR"
        recommendation = "Data quality is low. This is likely why fine-tuning isn't helping much.\nSTRONGLY RECOMMEND restructuring prompts before retraining."
    
    print(f"OVERALL DATA QUALITY: {grade} ({quality_score:.1f}/100)")
    print("\n💡 RECOMMENDATION:")
    print(f"   {recommendation}")
    print("=" * 80)
    
    # Show examples
    if analysis['sample_problems']['generic']:
        print("\n📝 SAMPLE GENERIC PROMPTS:")
        for i, prompt in enumerate(analysis['sample_problems']['generic'][:3], 1):
            print(f"\n{i}. {prompt[:200]}...")
    
    if analysis['sample_problems']['short']:
        print("\n📝 SAMPLE SHORT PROMPTS:")
        for i, prompt in enumerate(analysis['sample_problems']['short'][:3], 1):
            print(f"\n{i}. {prompt}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training data quality")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to training CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/data_analysis.json",
        help="Path to save analysis JSON"
    )
    
    args = parser.parse_args()
    
    # Analyze
    analysis = analyze_dataset(args.csv)
    
    # Display report
    display_report(analysis)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✓ Full analysis saved to: {output_path}")
    print("\nNext steps:")
    print("1. If quality score < 60: Improve training data (see docs/improving_lora_quality.md)")
    print("2. If quality score >= 60: Try better system prompts or increase LoRA rank")


if __name__ == "__main__":
    main()

