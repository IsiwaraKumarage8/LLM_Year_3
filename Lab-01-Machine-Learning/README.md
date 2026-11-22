# Lab 01: Sentiment Analysis Using Traditional Machine Learning

**Module:** PUSL 3205 – Large Language Models and Generative AI

A text-classification pipeline that predicts whether a sentence is **positive** or **negative**, using text preprocessing, Bag of Words / TF-IDF features and classic machine learning models.

## Dataset

[UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences): Amazon, IMDb and Yelp reviews, labelled `1` (positive) or `0` (negative).

| Item | Value |
|---|---|
| Sentences loaded | 3000 (1000 per source) |
| After removing duplicates | 2983 |
| Train / test split | 2386 / 597 (80/20, stratified) |
| Class balance | ~50% positive, ~50% negative |
| Vocabulary size (BOW) | 1647 words |

## Pipeline

1. **Load** the three files (`quoting=csv.QUOTE_NONE` so IMDb quotes don't merge rows)
2. **Preprocess**: lowercase, expand `n't` to `not`, remove punctuation and extra whitespace
3. **Explore**: examples, frequent words, log-odds of distinctive words, review length
4. **Split** into train and test sets
5. **Vectorise** with Bag of Words (stop words removed, negations kept)
6. **Train** Logistic Regression, Linear SVM and Naive Bayes
7. **Evaluate** with accuracy, precision, recall, F1, confusion matrices, 5-fold cross-validation
8. **Extension**: TF-IDF + bigrams + hyperparameter tuning with `GridSearchCV`
9. **Predict** on new sentences

## Results (test set, 597 sentences)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.8239 | 0.7852 | 0.8041 |
| Linear SVM | 0.7923 | 0.8063 | 0.7685 | 0.7869 |
| Naive Bayes | 0.8208 | 0.8328 | 0.8020 | 0.8171 |
| Tuned Logistic Regression (TF-IDF + bigrams) | 0.8174 | 0.8203 | 0.8121 | 0.8162 |

5-fold cross-validation F1: Logistic Regression 0.815, Linear SVM 0.801, Naive Bayes 0.818.

## Key findings

- All models score about **79–82%**. Naive Bayes and the tuned Logistic Regression are effectively tied.
- The **SVM overfits most** (98.3% training vs 79.2% test accuracy).
- The learned weights are sensible: `great`, `love`, `delicious` push positive; `bad`, `worst`, `not` push negative. Keeping `not` as a feature mattered.
- **Limitations:** Bag of Words ignores word order, so it fails on double negation ("not bad"), sarcasm ("Oh great, another delay") and mixed opinions. This is the gap embeddings, Transformers and LLMs fill.

## How to run

```powershell
# from the project root
uv pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
```

Open `notebook/sentiment_analysis.ipynb` and run all cells. The first cells download the dataset automatically if `sentiment.zip` is missing. If the download fails, save the zip from the UCI link next to the notebook as `sentiment.zip` and re-run.

## Project structure

```
Lab-01-Machine-Learning/
├── notebook/
│   ├── sentiment_analysis.ipynb   # full lab: code + explanations
│   └── data/                      # extracted dataset (git-ignored)
├── screenshots/                   # result charts
└── README.md
```

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn