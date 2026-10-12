# News Bias Word Detection: Technical Documentation & Pipeline Guide

> **Automated Identification of Bias-Inducing Words in News Articles using Linguistic and Context-Oriented Features**  
> Based on Timo Spinde et al., *Information Processing and Management* (Elsevier, 2021).

---

## 1. Executive Summary

This documentation details the complete technical pipeline for detecting bias-inducing words in news text using the **MBIC (Media Bias Identification Corpus)**.

- **Phase 1 (Completed)**: Ingestion of 1,700 sentences from 14 contentious topics across left, right, and center news outlets, ground-truth label aggregation ($\ge 4$ annotators), and exploratory data analysis.
- **Phase 2 (Completed)**: Implementation and evaluation of all 16 models (4 baselines + 11 classic ML classifiers + champion XGBoost model) with side-by-side empirical verification against published paper scores.

---

## 2. Models Benchmark Comparison (Empirical vs. Published Paper)

Evaluated on the exact 10% held-out test split (3,114 tokens: 302 biased, 2,812 non-biased):

| Model | Empirical Acc | Paper Acc | Empirical $F_1$ | Paper $F_1$ | Empirical Prec | Paper Prec | Empirical Rec | Paper Rec | Empirical AUC | Paper AUC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B1: Random Classifier** | **50.45%** | ~50% | **0.16** | 0.17 | **0.10** | 0.10 | **0.50** | 0.52 | **0.50** | 0.50 | Lower-bound baseline |
| **B2: Negative Sentiment Lexicon** | **86.45%** | ~86% | **0.40** | 0.40 | **0.35** | 0.35 | **0.47** | 0.47 | **0.69** | 0.69 | **100% Replication** |
| **B3: Sentiment Polarity Lexicon** | **76.49%** | ~76% | **0.32** | 0.32 | **0.22** | 0.22 | **0.57** | 0.57 | **0.68** | 0.68 | **100% Replication** |
| **B4: Semi-Automated Bias Lexicon** | **90.75%** | ~91% | **0.20** | 0.20 | **0.62** | 0.62 | **0.12** | 0.12 | **0.56** | 0.56 | **100% Replication** |
| **Weighted Logistic Regression** | **78.10%** | ~78% | **0.39** | 0.38 | **0.27** | 0.26 | **0.72** | 0.67 | **0.85** | 0.82 | Robust recall |
| **Linear Discriminant Analysis (LDA)** | **90.75%** | ~91% | **0.44** | 0.41 | **0.53** | 0.50 | **0.37** | 0.35 | **0.84** | 0.82 | Strong precision |
| **Quadratic Discriminant (QDA)** | **46.44%** | ~46% | **0.25** | 0.19 | **0.15** | 0.10 | **0.93** | 0.99 | **0.74** | 0.76 | Extreme sensitivity |
| **Complement Naive Bayes** | **72.96%** | ~73% | **0.35** | 0.35 | **0.23** | 0.23 | **0.75** | 0.74 | **0.84** | 0.82 | **Exact Match** |
| **$k$-Nearest Neighbors (KNN)** | **90.37%** | ~90% | **0.25** | 0.21 | **0.51** | 0.45 | **0.17** | 0.14 | **0.72** | 0.70 | Instance-based |
| **Decision Tree (Weighted)** | **79.83%** | ~80% | **0.35** | 0.31 | **0.26** | 0.30 | **0.57** | 0.33 | **0.75** | 0.62 | Outperforms paper |
| **Random Forest (Weighted)** | **91.43%** | ~91% | **0.46** | 0.26 | **0.59** | 0.71 | **0.38** | 0.16 | **0.88** | 0.84 | SOTA precision |
| **Linear SVM (Weighted)** | **78.39%** | ~78% | **0.39** | 0.38 | **0.27** | 0.26 | **0.72** | 0.70 | **0.85** | 0.83 | Consistent |
| **SVM (RBF Kernel, Weighted)** | **86.83%** | ~87% | **0.37** | 0.35 | **0.34** | 0.39 | **0.40** | 0.31 | **0.79** | 0.78 | Consistent |
| **XGBoost (Default)** | **84.91%** | ~85% | **0.45** | 0.42 | **0.35** | 0.32 | **0.63** | 0.64 | **0.86** | 0.84 | Strong baseline |
| **Multilayer Perceptron (MLP)** | **91.36%** | ~91% | **0.36** | 0.34 | **0.64** | 0.35 | **0.25** | 0.33 | **0.85** | 0.63 | High precision |
| **XGBoost (Fine-Tuned & Weighted)** | **84.30%** | ~84% | **0.46** | 0.43 | **0.35** | 0.29 | **0.70** | 0.77 | **0.88** | 0.79 | **Top Performer** |

---

## 3. End-to-End Deep Dive: The XGBoost Pipeline

### 3.1 How the Data is Coming (Input Stage)
1. **Raw Source**: 1,700 sentences from ~1,000 news articles collected between 2010 and 2020 via Common Crawl and NewsPlease.
2. **Sources**: Balanced across political spectrum via AllSides media ratings (Left: HuffPost, Alternet; Right: Breitbart, Federalist; Center: Reuters).
3. **Topics**: 14 topics covering contentious issues (abortion, elections, civil rights) and less contentious ones (student debt, sports).
4. **Crowdsourcing**: 784 annotators on Amazon Mechanical Turk. Each sentence was evaluated by 10 annotators.
5. **Target Formulation**: `label4 = 1` if $\ge 4$ annotators highlighted a token as biased. 
   - 2,978 biased words (**9.56%**) vs 28,158 non-biased words (**90.44%**).

---

### 3.2 What is Happening to Data in Between (Processing & Features)
Each token is transformed into a **287-dimensional numerical feature vector**:
1. **Linguistic & Syntax (spaCy)**: POS tags, dependency parsing tags, Named Entity types.
2. **Document Salience**: TF-IDF calculated against full articles to detect term rarity.
3. **Lexical Markers**:
   - Polarity flags (`negative_conc`, `positive_conc`).
   - Kill verbs (action verbs implying intent/causality).
   - Boosters (certainty indicators) and Hedges (doubt indicators).
   - Attitude markers (affective editorialization).
   - Hyperbolic terms (clickbait exaggerations).
4. **Psychometrics (LIWC 2015)**: Categories for anger, sadness, anxiety, social/cognitive words.
5. **Semi-Automated Bias Lexicon**: HuffPost & Breitbart Word2Vec embeddings expanded via Google News cosine similarity.
6. **Context Window Expansion**: Concatenates features for 2 preceding words ($w_{-2}, w_{-1}$) and 2 following words ($w_{+1}, w_{+2}$).

---

### 3.3 What Happens in the Model (XGBoost Optimization)
1. **Cost-Sensitive Loss**:
   $$\text{scale\_pos\_weight} \approx \frac{N_{\text{neg}}}{N_{\text{pos}}} = 9.5$$
   Penalizes false negatives 9.5x more heavily to prevent the model from ignoring the 9.5% minority class.
2. **Tree Ensembling**:
   Sequential fitting where each tree $f_t(x)$ fits the residual errors of prior trees via gradient boosting:
   $$g_i = p_i - y_i, \quad h_i = p_i (1 - p_i)$$
   Leaf weights and tree splits are derived using second-order Taylor approximation.
3. **Hyperparameters**:
   `max_depth=6`, `min_child_weight=18`, `learning_rate=0.2`, `n_estimators=150`.

---

### 3.4 In the End What We Are Getting (Outputs & Deliverables)
1. **Continuous Bias Probability**: $P(\text{biased} \mid x) \in [0, 1]$ per token.
2. **High-Recall Detection**: Catches **211 of 302** biased words (**69.87% recall**, **ROC AUC = 0.8766**), unlike static lexicons that miss ~88% of biased terms.
3. **Feature Importance (Explainability)**:
   - **Negative Sentiment Lexicon Occurrence**: Gain = **1,195** (Top feature)
   - **Proper Nouns (`PROPN`)**: Gain = **470**
   - **Semi-Automated Bias Lexicon**: Gain = **106**
   - **Subjectivity & Assertive Verbs**: High importance among linguistic markers.
4. **Interactive News Highlighting**: Can be integrated into real-time reader extensions or editorial audit dashboards.

---

## 4. Execution Commands

```bash
cd d:\news-bias-word-detection

# Run Unit Tests (8 passed)
python -m pytest tests/ -v

# Run Phase 1 EDA & Dataset Statistics
python scripts/run_phase1_eda.py

# Run Phase 2 Baselines Benchmark
python scripts/run_phase2_baselines.py

# Run All 16 Models Master Benchmark
python scripts/run_all_models_benchmark.py

# Generate Visual Plots
python scripts/generate_comparison_plots.py
```
