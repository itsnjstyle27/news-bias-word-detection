#!/usr/bin/env python
"""
Phase 1 Execution Script: Dataset Ingestion and Exploratory Data Analysis (EDA).
Replicates corpus statistics reported in Spinde et al. (2021).
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import setup_logger
from src.data.dataset_loader import MBICDataLoader
from src.data.eda import run_exploratory_analysis

logger = setup_logger("phase1_runner")


def main():
    logger.info("=" * 70)
    logger.info("Starting Phase 1: Ingestion, Validation & Exploratory Data Analysis")
    logger.info("=" * 70)

    loader = MBICDataLoader()
    output_dir = "outputs/eda"

    report = run_exploratory_analysis(loader, output_dir=output_dir, generate_plots=True)

    tokens = report.get("tokens", {})
    sentences = report.get("sentences", {})

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY METRICS")
    print("=" * 70)
    print(f"Total Sentences Analyzed:         {sentences.get('total_count', 'N/A')}")
    print(f"Contentious Topics Covered:       {sentences.get('unique_topics', 'N/A')}")
    print(f"Total Word Observations:          {tokens.get('total_tokens', 'N/A'):,}")
    print(f"Biased Words (threshold >= 4):    {tokens.get('biased_tokens_count', 'N/A'):,} ({tokens.get('imbalance_percentage', 'N/A')}%)")
    print(f"Unbiased Words:                   {tokens.get('unbiased_tokens_count', 'N/A'):,}")
    print("\nTop POS Tags of Biased Words (Paper Table 3):")
    for pos, data in tokens.get("pos_breakdown_of_biased_words", {}).items():
        print(f"  - {pos:<6}: {data['count']:>5} ({data['percentage']}%)")

    print("\nLexicon Overlap Precision:")
    for lex, data in tokens.get("lexicon_statistics", {}).items():
        print(f"  - {lex:<15}: {data['total_occurrences']:>5} words | Biased Matches: {data['overlap_with_biased']:>4} | Precision: {data['precision']:.2%}")

    print("\nArtifacts saved:")
    print(f"  - Summary JSON:  {os.path.join(output_dir, 'eda_summary.json')}")
    print(f"  - Report MD:     {os.path.join(output_dir, 'eda_report.md')}")
    print(f"  - Plots:         {output_dir}/*.png")
    print("=" * 70)
    logger.info("Phase 1 completed successfully!")


if __name__ == "__main__":
    main()
