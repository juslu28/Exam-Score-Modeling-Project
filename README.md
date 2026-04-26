# Exam Score Prediction — Kaggle Playground Series S6E1

Regression pipeline for predicting student exam scores from behavioral and environmental features. Final ensemble achieves **RMSE 8.632** on out-of-fold validation.

---

## Results

| Model | OOF RMSE |
|---|---|
| XGBoost | 8.647 |
| LightGBM | 8.742 |
| MLP (PyTorch) | 8.857 |
| **Ridge Stack** | **8.632** |

Baseline (mean prediction): RMSE ≈ 18.9 (1 SD of target).

---

## Approach

### Data
Synthetic dataset of 630,000 student records with 11 features covering study behavior (study hours, class attendance, sleep), environmental factors (facility rating, internet access), and course metadata. No missing values or duplicates.

### Feature Engineering
47 features engineered from 11 base features, including:
- Reverse-engineered formula feature — the data-generating process was recovered via Tobit model MLE and community validation, producing a single scalar that closely approximates the latent exam score before synthetic noise is added. This was the highest-impact feature in the pipeline.
- Polynomial transforms, log/sqrt transforms, and pairwise interaction terms for numeric features
- Ordinal encoding for `sleep_quality`, `facility_rating`, `exam_difficulty`
- Dual representation of base numeric features: raw `category` dtype (for XGBoost's value-specific categorical split algorithm) alongside engineered float transforms (for smooth threshold-based splits)

### Ensemble Architecture
Three structurally distinct base learners trained on identical 10-fold CV splits:
- **XGBoost** — depth-wise gradient boosted trees with native categorical support
- **LightGBM** — leaf-wise gradient boosted trees
- **MLP** — 3-layer feedforward network (256 units, BatchNorm, dropout) included for architectural diversity; as the only non-tree model it spans a fundamentally different region of the error surface

Base model OOF predictions are stacked using **Ridge regression** as the meta-learner, which finds the optimal regularized linear combination while avoiding overfitting on the meta-features.

Both XGBoost and LightGBM use early stopping (50 rounds) against the held-out fold. The original competition dataset is included in the training pool alongside the synthetic training data.

### Interpretability
SHAP analysis (TreeExplainer on the dominant XGBoost base learner) confirms that model-learned feature importance aligns with EDA findings:
- The formula feature dominates by a wide margin, consistent with its construction
- `study_hours × class_attendance` is the highest-ranked engineered feature — joint study effort is the primary behavioral driver of exam performance
- Ordinal environmental features (`sleep_quality`, `facility_rating`) contribute primarily through interaction terms with study behavior, rather than independently
- `gender` and `internet_access` rank near the bottom, consistent with their weak univariate target trends in EDA

---

## Repository Structure

```
.
├── data/                  # Kaggle competition data (not tracked)
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── Exam_Score_Prediction.csv
├── outputs/               # Generated outputs (not tracked)
├── student-test-predict-eda (3).ipynb
└── .gitignore
```

## Data

Data is sourced from the [Kaggle Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1) competition. Download and place files in `data/` before running the notebook.

```bash
kaggle competitions download -c playground-series-s6e1
unzip playground-series-s6e1.zip -d data/
```

## Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas seaborn matplotlib scikit-learn torch xgboost lightgbm shap ipykernel
```

---

*Author: Justin Lu*
