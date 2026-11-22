# Laboratory Session 01: Sentiment Analysis Using Traditional Machine Learning

## Objective
Practice basic Natural Language Processing (NLP) preprocessing, Bag-of-Words (BoW) feature extraction, and supervised machine learning classification to predict sentiment from customer reviews.

## Tasks and Implementation
1. **Dataset Loading:** Downloaded the UCI Sentiment Labeled Sentences dataset and loaded `yelp_labelled.txt` (1,000 customer reviews) into a Pandas DataFrame.
2. **Text Preprocessing:** Converted all reviews to lowercase, removed punctuation and non-alphabetic characters using regular expressions, cleaned up extra spaces, and stored the output in a `clean_review` column.
3. **Data Exploration:** Checked the label split (500 positive, 500 negative), printed sample reviews from each class, and examined the most common words after removing stopwords.
4. **Train/Test Split:** Divided the dataset into 80% training (800 reviews) and 20% testing (200 reviews) using a stratified split (`random_state=42`).
5. **Feature Extraction:** Converted the cleaned text into a Bag-of-Words matrix (`(800, 1783)`) using `CountVectorizer(max_features=2500)`.
6. **Model Training:** Trained a `LogisticRegression` classifier (`random_state=42`) on the training vectors.
7. **Evaluation:** Tested the classifier on the held-out test set and calculated Accuracy, Precision, Recall, and F1-Score.
8. **Custom Sentence Testing:** Wrote a `predict_sentiment` helper function to clean, vectorize, and classify unseen review sentences.

## Results and Evidence
- **Notebook:** [`notebook/sentiment_analysis.ipynb`](notebook/sentiment_analysis.ipynb)
- **Test Set Results (`LogisticRegression` + BoW):**
  - **Accuracy:** 85.00% (`0.8500`)
  - **Precision:** `0.8365`
  - **Recall:** `0.8700`
  - **F1-Score:** `0.8529`
- **Sample Predictions on Unseen Reviews:**
  - `"The food was absolutely wonderful and the service was top notch!"` → **Positive**
  - `"Terrible experience, I will never come back here again."` → **Negative**
- **Screenshots:**
  - ![Evaluation Metrics](screenshots/evaluation_metrics.png)
  - ![Word Frequency Exploration](screenshots/word_frequency.png)

## Observations
- Lowercasing text and removing punctuation combined duplicate variations of the same word (such as `"Great"` and `"great"`), keeping the vocabulary smaller and cleaner without losing the meaning of the review.
- Removing common English stopwords was important during exploration; before filtering them out, words like `"the"`, `"and"`, and `"was"` topped the frequency list instead of useful sentiment words like `"great"`, `"good"`, and `"nasty"`.
- Although Logistic Regression with Bag-of-Words reached 85.00% accuracy on the test set, a single-word BoW model ignores word order, so it cannot capture phrase-level context or negation (like `"not good"`) as effectively as n-grams or modern language models.

## Learning Outcomes
- Built a repeatable Python text-cleaning workflow using Pandas and regular expressions.
- Turned raw text reviews into numerical feature vectors with Scikit-learn's `CountVectorizer`.
- Trained and evaluated a baseline sentiment classifier using Accuracy, Precision, Recall, and F1-Score.