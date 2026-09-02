# Email Spam Classification — Spambase Project

A complete, reproducible ML project built on the classic **UCI Spambase**
dataset: 4,601 emails, each described by 57 numeric features (word/character
frequencies plus capital-letter run-length stats) and a binary label
(`1` = spam, `0` = not spam).

## Project structure

```
spam_classifier_project/
├── data/
│   └── spambase.csv              # source data
├── src/
│   ├── train_and_evaluate.py     # full pipeline: EDA -> train -> evaluate -> save
│   └── predict.py                # load saved model, score new emails
├── outputs/
│   ├── model/
│   │   ├── best_model.pkl        # trained Random Forest
│   │   └── scaler.pkl            # fitted StandardScaler
│   ├── plots/                    # all chart PNGs (see below)
│   ├── model_comparison.csv      # metrics table for all 5 models
│   └── results.json              # machine-readable run summary
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python3 src/train_and_evaluate.py      # trains everything, writes outputs/
python3 src/predict.py new_emails.csv  # score new data with the saved model
```

## Dataset

- **4,601 rows**, **57 features**, no missing values.
- Class balance: 2,788 non-spam (60.6%) vs. 1,813 spam (39.4%) — moderately
  imbalanced but not severe.
- Features: `word_freq_*` (frequency of specific words), `char_freq_*`
  (frequency of characters like `!`, `$`), and three `capital_run_length_*`
  stats describing runs of capital letters.

## Methodology

1. **EDA** — class balance and the features most correlated with the spam
   label (unsurprisingly: `!`, `$`, "remove", "free", "your", capital-letter
   runs).
2. **Preprocessing** — 80/20 stratified train/test split; features
   standardized with `StandardScaler` (fit on train only, to avoid leakage).
3. **Model comparison** — trained and 5-fold cross-validated five
   classifiers: Logistic Regression, Gaussian Naive Bayes, Random Forest,
   Gradient Boosting, and an RBF-kernel SVM.
4. **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC on the held-
   out test set, plus 5-fold CV accuracy for stability.
5. **Model selection** — ranked by F1 score (a balanced metric appropriate
   here, since both false positives — real mail marked spam — and false
   negatives — spam reaching the inbox — are costly).

## Results

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC | CV Acc (mean ± std) |
|----------------------|:--------:|:---------:|:------:|:-----:|:-------:|:--------------------|
| **Random Forest**    | **0.946**| **0.951** | 0.909  | **0.930** | **0.983** | 0.952 ± 0.013 |
| Gradient Boosting    | 0.939    | 0.937     | 0.906  | 0.922 | 0.983   | 0.945 ± 0.013 |
| Logistic Regression  | 0.929    | 0.921     | 0.898  | 0.909 | 0.970   | 0.924 ± 0.008 |
| SVM (RBF)            | 0.927    | 0.928     | 0.884  | 0.906 | 0.967   | 0.933 ± 0.009 |
| Naive Bayes          | 0.833    | 0.715     | 0.959  | 0.819 | 0.938   | 0.817 ± 0.009 |

**Best model: Random Forest** — 94.6% test accuracy, 0.930 F1, 0.983 ROC-AUC.
Naive Bayes has the highest recall (catches the most spam) but far more false
positives, making it the weakest overall trade-off here.

### Plots produced (in `outputs/plots/`)
- `class_distribution.png` — spam vs. non-spam counts
- `top_correlated_features.png` — features most linked to the spam label
- `model_comparison.png` — bar chart of all 5 models across all metrics
- `roc_curves_all_models.png` — ROC curves overlaid for all models
- `confusion_matrix_best_model.png` — confusion matrix for the Random Forest
- `feature_importance_best_model.png` — top 15 features driving predictions

## Key takeaways

- Character frequencies for `!` and `$`, plus word frequencies for
  "remove", "free", "your", and "000", are strong spam signals — consistent
  with typical marketing/scam language.
- Tree ensembles (Random Forest, Gradient Boosting) outperform linear models
  and SVM here, suggesting non-linear feature interactions matter.
- With 0.983 ROC-AUC, the classifier separates spam from ham very reliably;
  further gains would likely come from feature engineering (e.g., n-grams,
  header metadata) rather than more model tuning.

## Possible extensions
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) on the Random Forest
- Threshold tuning to explicitly trade off precision vs. recall for a
  production spam filter (e.g., minimize false positives)
- Try XGBoost/LightGBM or a simple neural network for comparison
- Deploy `predict.py` behind a small API for real-time scoring
