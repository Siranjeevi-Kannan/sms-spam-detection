import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix,
    classification_report, make_scorer,
    f1_score, precision_score, recall_score,
)
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


path = "https://github.com/Siranjeevi-Kannan/sms-spam-detection/raw/refs/heads/main/dataset/spam.csv"
data = pd.read_csv(path, encoding="latin1")
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data.rename(columns={"v1": "label", "v2": "msg"}, inplace=True)
le = LabelEncoder()
data['label_num'] = le.fit_transform(data["label"])  # spam=1, ham=0


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' __url__ ', text)
    text = re.sub(r'£|\$|€', ' __currency__ ', text)
    text = re.sub(r'\b\d{10,}\b', ' __phone__ ', text)
    text = re.sub(r'!{2,}', ' __exclaim__ ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


data['processed'] = data['msg'].apply(preprocess)
X_all = data['processed'].to_numpy()
y_all = data['label_num'].to_numpy()

print(f"Majority class baseline (always predict ham): {(data['label_num'] == 0).mean():.4f} accuracy")
print(f"Class distribution:\n{data['label'].value_counts()}\n")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)


configs = [
    ("BoW (1,1) + MNB, sw kept", CountVectorizer(ngram_range=(1,1)), MultinomialNB()),
    ("BoW (1,1) + MNB, sw removed", CountVectorizer(ngram_range=(1,1), stop_words='english'), MultinomialNB()),
    ("BoW (1,2) + MNB, sw kept", CountVectorizer(ngram_range=(1,2)), MultinomialNB()),
    ("BoW (1,2) + MNB, sw removed", CountVectorizer(ngram_range=(1,2), stop_words='english'), MultinomialNB()),
    ("BoW (1,1) + CNB, sw kept", CountVectorizer(ngram_range=(1,1)), ComplementNB()),
    ("TF-IDF (1,1) + CNB, sw kept", TfidfVectorizer(ngram_range=(1,1)), ComplementNB()),
    ("TF-IDF (1,2) + CNB, sw kept", TfidfVectorizer(ngram_range=(1,2)), ComplementNB()),
    ("BoW (1,1) + LR, sw kept", CountVectorizer(ngram_range=(1,1)), LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("BoW (1,2) + LR, sw kept", CountVectorizer(ngram_range=(1,2)), LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("TF-IDF (1,1) + LR, sw kept", TfidfVectorizer(ngram_range=(1,1)), LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("TF-IDF (1,1) + LR, sw removed", TfidfVectorizer(ngram_range=(1,1), stop_words='english'), LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("TF-IDF (1,2) + LR, sw kept", TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("TF-IDF (1,2) + LR, sw removed", TfidfVectorizer(ngram_range=(1,2), stop_words='english'), LogisticRegression(max_iter=1000, class_weight='balanced')),
]

scorers = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Config':<45} {'Precision':>12} {'Recall':>12} {'F1':>10}")

cv_results = {}
for name, vectorizer, clf in configs:
    pipe = Pipeline([('vec', vectorizer), ('clf', clf)])
    scores = cross_validate(pipe, X_all, y_all, cv=cv, scoring=scorers)
    cv_results[name] = {
        'scores': scores,
        'vectorizer': vectorizer,
        'clf': clf
    }
    p = scores['test_precision'].mean()
    r = scores['test_recall'].mean()
    f = scores['test_f1'].mean()
    ps = scores['test_precision'].std()
    rs = scores['test_recall'].std()
    fs = scores['test_f1'].std()
    print(f"{name:<45} {p:.3f}±{ps:.3f}  {r:.3f}±{rs:.3f}  {f:.3f}±{fs:.3f}")


best_name = max(cv_results, key=lambda n: cv_results[n]['scores']['test_f1'].mean())
best_vectorizer = cv_results[best_name]['vectorizer']
best_clf = cv_results[best_name]['clf']

X_train_vec = best_vectorizer.fit_transform(X_train)
X_test_vec = best_vectorizer.transform(X_test)
best_clf.fit(X_train_vec, y_train)
y_pred = best_clf.predict(X_test_vec)

print(f"\nDetailed report — {best_name}")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
axes[0].set_title(f'Confusion matrix\n{best_name}')
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted label')

y_scores = best_clf.predict_proba(X_test_vec)[:, 1]
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores)
axes[1].plot(recall_vals, precision_vals, color='steelblue', linewidth=2)
axes[1].set_title('Precision-Recall curve')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].grid(True, alpha=0.3)

names = list(cv_results.keys())
f1_means = [cv_results[n]['scores']['test_f1'].mean() for n in names]
f1_stds = [cv_results[n]['scores']['test_f1'].std() for n in names]
axes[2].barh(names, f1_means, xerr=f1_stds, color='steelblue', alpha=0.7, capsize=4)
axes[2].set_title('F1 score — 5-fold CV (mean ± std)')
axes[2].set_xlabel('F1 score')
axes[2].set_xlim(0.7, 1.0)
axes[2].grid(True, axis='x', alpha=0.3)
axes[2].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
plt.show()


print("\nFalse Positives — ham classified as spam:")
fp_mask = (y_pred == 1) & (y_test == 0)
for msg in X_test[fp_mask][:5]:
    print(f"  {msg}")

print("\nFalse Negatives — spam classified as ham:")
fn_mask = (y_pred == 0) & (y_test == 1)
for msg in X_test[fn_mask][:5]:
    print(f"  {msg}")


print("\nTop 15 spam indicator features:")
feature_names = best_vectorizer.get_feature_names_out()
coefs = best_clf.coef_[0]
for idx in np.argsort(coefs)[-15:][::-1]:
    print(f"  {feature_names[idx]:<20} {coefs[idx]:.4f}")

print("\nTop 15 ham indicator features:")
for idx in np.argsort(coefs)[:15]:
    print(f"  {feature_names[idx]:<20} {coefs[idx]:.4f}")


def predict_text(inp):
    processed = preprocess(inp)
    vec = best_vectorizer.transform([processed])
    pred = best_clf.predict(vec)[0]
    prob = best_clf.predict_proba(vec)[0]
    label = le.inverse_transform([pred])[0]
    print(f"Input: {inp}\nProcessed: {processed}\nPrediction: {label}  (ham: {prob[0]:.3f}, spam: {prob[1]:.3f})\n")


predict_text("Congratulations! You've won a free £1000 prize. Call now!")
predict_text("Hey, are we still meeting for lunch tomorrow?")
predict_text("URGENT: Your account has been compromised. Verify now at www.google.com")