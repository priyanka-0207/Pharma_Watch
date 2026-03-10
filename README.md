# 💊 PharmaWatch - AI Drug Safety Intelligence Platform

I've always been curious about how drugs affect people differently, and during my Master's at Northeastern, I decided to dig into it. I used ToolUniverse — a scientific tool platform built by researchers at Harvard and MIT — to pull real adverse event data from the FDA for 20 common drugs like aspirin, metformin, and sertraline. From there, I built out the whole pipeline myself: cleaning and transforming the messy data, engineering features like TF-IDF scores to figure out which side effects actually matter for each drug, training ML models to predict which reactions are likely to be serious, and running NLP on drug labels to catch risk patterns buried in the text. I put everything into a Streamlit dashboard so anyone can explore it without touching code. The most interesting part was honestly the mistakes — my models initially showed perfect accuracy, and it took me a while to realize I had data leakage in my target variable. Fixing that taught me more about real-world ML than any classroom assignment has.

---

## How It Works

```
FDA FAERS Data ──→ ETL Pipeline ──→ ML Models ──→ Streamlit Dashboard
(via ToolUniverse)   (Feature Eng)    (XGBoost)     (Interactive UI)
                         │
                         └──→ NLP Module ──→ Risk Pattern Detection
                              (TF-IDF)       (Drug Label Analysis)
```

**1. Data Collection** — Pulled real adverse event reports, seriousness outcomes, and drug label text for 20 drugs across 12 therapeutic categories using ToolUniverse's FDA tools via MCP.

**2. ETL Pipeline** — Transforms raw pharmacovigilance data into ML-ready features including log-scaled reaction counts, TF-IDF weighted reaction importance, drug-specific reaction flags, and reaction diversity metrics.

**3. ML Pipeline** — Compares 4 models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with stratified 5-fold CV to predict reaction severity. Handles class imbalance and includes regularization to prevent overfitting.

**4. NLP Module** — Analyzes FDA drug label text using TF-IDF keyword extraction, cosine similarity between drug warnings, and regex-based risk pattern detection for phrases like "fatal," "life-threatening," and "suicidal ideation."

**5. Dashboard** — 4-page Streamlit app for interactive exploration of drug safety profiles, model performance, and NLP insights.

---

## Dashboard Preview

| Overview | Drug Explorer |
|----------|--------------|
| ![Overview](results/screenshots/overview.png) | ![Explorer](results/screenshots/drug_explorer.png) |

| ML Predictions | NLP Insights |
|----------------|-------------|
| ![ML](results/screenshots/ml_predictions.png) | ![NLP](results/screenshots/nlp_insights.png) |

> *Replace with your actual screenshots from the dashboard*

---

## Project Structure

```
drug-safety-intelligence/
├── data/
│   ├── raw_adverse_events.json        # FDA FAERS data (via ToolUniverse)
│   ├── raw_drug_labels.json           # Drug label text (via ToolUniverse)
│   └── processed_drug_safety.csv      # ETL pipeline output
├── src/
│   ├── etl_pipeline.py                # Data ingestion + feature engineering
│   ├── ml_pipeline.py                 # Model training + evaluation
│   └── nlp_analyzer.py                # TF-IDF + risk pattern analysis
├── results/
│   ├── model_results.json             # Model comparison metrics
│   ├── nlp_analysis.json              # Keywords + risk scores
│   └── label_text_similarity.csv      # Drug label similarity matrix
├── app.py                             # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- ToolUniverse (`pip install tooluniverse')
