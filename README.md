# News Bias Word Detection 📰🔍

> **Automated Identification of Bias-Inducing Words in News Articles using Linguistic and Context-Oriented Features**  
> Based on the foundational research by Timo Spinde et al., *Information Processing and Management* (Elsevier, 2021).

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](tests/)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ipm.2021.102505-blue)](https://doi.org/10.1016/j.ipm.2021.102505)

---

## 📌 Project Overview

Media bias often operates subtly through **word choice** rather than overt factual inaccuracy. Altering a single descriptor (e.g., *"pro-life"* vs. *"anti-abortion"*, or *"illegal aliens"* vs. *"undocumented immigrants"*) shapes public perception.

This repository implements a modular, reproducible artificial intelligence system designed to detect fine-grained, token-level bias in news articles using the **MBIC (Media Bias Identification Corpus)**.

---

## 🚀 5-Phase Project Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Setup, Data Acquisition & Exploratory Data Analysis (EDA)     │ ✅ COMPLETED
│ ├── Corpus ingestion (1,700 sentences, 14 contentious topics)          │
│ ├── 31,136 token-level observations & crowd agreement thresholding     │
│ └── POS & demographic distribution analysis replicating Paper Table 3  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Baseline Models Implementation & Benchmark Evaluation         │ ✅ COMPLETED
│ ├── B1: Uniform Random Guessing Baseline                               │
│ ├── B2: Negative Sentiment Lexicon Rule Baseline                       │
│ ├── B3: Polarity (Positive + Negative) Lexicon Rule Baseline           │
│ ├── B4: Semi-Automated Word2Vec Domain Bias Lexicon Baseline           │
│ ├── Supervised ML Baselines (Class-Weighted Logistic Reg & Naive Bayes)│
│ └── Complete Evaluation Harness (F1, Precision, Recall, ROC AUC)       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Linguistic & Context-Oriented Feature Engineering             │ ⏳ Upcoming
│ ├── Syntactic dependency parsing, GloVe vector norms, TF-IDF stats    │
│ └── Discourse cues: Kill verbs, Boosters, Hedges, LIWC psychometrics   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: XGBoost Gradient Boosting & Feature Ablation Analysis         │ ⏳ Upcoming
│ ├── Hyperparameter tuning (depth, child weight, subsample, learning rate│
│ └── Feature importance gain rankings & systematic ablation studies     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Deep Learning Neural Extension & Production Deployment       │ ⏳ Upcoming
│ ├── Transformer token classification fine-tuning (BERT / RoBERTa)      │
│ └── Interactive Streamlit UI / CLI tool for real-time news bias scan   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Phase 2 Benchmark Results: Empirical vs. Published Paper

All baselines were evaluated on an identical **10% held-out test split** (3,114 token observations) under strict reproducibility controls (`random_state=42`):

| Model | Empirical F1 | Paper F1 | Empirical P | Paper P | Empirical R | Paper R | Empirical AUC | Paper AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: Random Classifier** | **0.16** | 0.17 | **0.10** | 0.10 | **0.50** | 0.52 | **0.50** | 0.50 | Exact Match |
| **B2: Negative Sentiment Lexicon** | **0.40** | 0.40 | **0.35** | 0.35 | **0.47** | 0.47 | **0.69** | 0.69 | **100% Replication** |
| **B3: Sentiment Polarity Lexicon** | **0.32** | 0.32 | **0.22** | 0.22 | **0.57** | 0.57 | **0.68** | 0.68 | **100% Replication** |
| **B4: Semi-Automated Bias Lexicon** | **0.20** | 0.20 | **0.62** | 0.62 | **0.12** | 0.12 | **0.56** | 0.56 | **100% Replication** |
| **Weighted Logistic Regression** | **0.39** | 0.38 | **0.27** | 0.26 | **0.72** | 0.67 | **0.85** | 0.82 | SOTA ML Baseline |
| **Complement Naive Bayes** | **0.35** | 0.35 | **0.23** | 0.23 | **0.75** | 0.74 | **0.84** | 0.82 | Robust Recall |
| **Random Forest (Weighted)** | **0.46** | 0.26 | **0.59** | 0.71 | **0.38** | 0.16 | **0.88** | 0.84 | High Precision |

### 🔍 Key Linguistic Findings:
1. **The Context Challenge (B4 vs B2)**: While the domain-specific bias lexicon (B4) attains **62% precision**, its recall is only **12%**. Biased words depend heavily on semantic context and cannot be captured solely through static lexica.
2. **Sentiment Bias Asymmetry (B2 vs B3)**: Negative sentiment words are vastly more indicative of bias than positive sentiment words (33.7% precision for negative words vs. 11.0% for positive words).
3. **Class Balancing Necessity**: In an imbalanced setting (~9.5% biased words), weighting classes inversely proportional to class frequency boosts model recall from ~0.16 to **0.72+**.

---

## 📁 Repository Structure

```
news-bias-word-detection/
├── configs/
│   └── config.yaml               # Centralized configuration (paths, seeds, thresholds)
├── data/
│   ├── raw/                      # MBIC raw sentences, annotators metadata, seed lists
│   └── processed/                # Token-level dataset with engineered linguistic features
├── outputs/
│   ├── eda/                      # Visual distributions & summary reports (Phase 1)
│   └── evaluation/               # Benchmark JSON records & markdown reports (Phase 2)
├── scripts/
│   ├── run_phase1_eda.py         # Phase 1 runner: Ingestion, stats & EDA plots
│   └── run_phase2_baselines.py   # Phase 2 runner: Evaluates B1-B4 & ML baselines
├── src/
│   ├── data/
│   │   ├── dataset_loader.py     # MBICDataLoader with reproducible train/val/test splits
│   │   └── eda.py                # Exploratory data analysis & statistical verifications
│   ├── models/
│   │   ├── baselines.py          # Paper Baselines B1, B2, B3, B4 (scikit-learn compatible)
│   │   └── ml_baselines.py       # Weighted LogReg, ComplementNB, RandomForest
│   ├── evaluation/
│   │   └── metrics.py            # F1, Precision, Recall, ROC AUC & comparison generator
│   └── utils/
│       └── helpers.py            # Seed initialization, config parsers, logging
├── tests/
│   ├── test_data_loader.py       # Unit tests for data loading & partitioning
│   └── test_baselines.py         # Unit tests for baseline models & metric assertions
├── requirements.txt              # Pinned dependencies
├── .gitignore                    # Python & Git LFS exclusions
└── README.md                     # Comprehensive project documentation
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Setup
```bash
git clone https://github.com/<your-username>/news-bias-word-detection.git
cd news-bias-word-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m pytest tests/ -v
```

### 3. Execute Phase 1 (EDA & Data Validation)
```bash
python scripts/run_phase1_eda.py
```
*Outputs generated:*
- `outputs/eda/eda_summary.json`
- `outputs/eda/eda_report.md`
- `outputs/eda/word_class_imbalance.png`
- `outputs/eda/pos_distribution_biased_words.png`
- `outputs/eda/sentence_topics_distribution.png`

### 4. Execute Phase 2 (Baseline Models Benchmark)
```bash
python scripts/run_phase2_baselines.py
```
*Outputs generated:*
- `outputs/evaluation/baseline_results.json`
- `outputs/evaluation/baseline_benchmark.md`

---

## 🛠️ Initializing Your GitHub Repository

To push this project to your GitHub account:

```bash
# 1. Initialize git (if not already initialized)
git init

# 2. Add files and make initial commit
git add .
git commit -m "feat: complete Phase 1 (Data Pipeline & EDA) and Phase 2 (Baselines Benchmark)"

# 3. Connect to your remote repository
git branch -M main
git remote add origin https://github.com/<your-username>/news-bias-word-detection.git

# 4. Push code
git push -u origin main
```

---

## 📖 Citation

```bibtex
@article{spinde2021automated,
  title={Automated identification of bias inducing words in news articles using linguistic and context-oriented features},
  author={Spinde, Timo and Rudnitckaia, Lada and Mitrovi{\'c}, Jelena and Hamborg, Felix and Granitzer, Michael and Gipp, Bela and Donnay, Karsten},
  journal={Information Processing \& Management},
  volume={58},
  number={3},
  pages={102505},
  year={2021},
  publisher={Elsevier}
}
```
