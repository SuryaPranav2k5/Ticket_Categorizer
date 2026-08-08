#!/usr/bin/env python3
"""
Auto Email / Ticket Categorizer — AI/ML Intern Assessment
Fobes Skill Itech Pvt Ltd

A lightweight NLP classifier that reads incoming support tickets
and routes them to the correct department automatically.

Categories: Billing | Technical | HR | General
Stack: Python, scikit-learn, pandas, numpy
"""

import sys
import re
import string
import warnings
import joblib
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

import os

# ============================================================
# 1. REAL DATASET LOADING (directly from dataset/ folder)
# ============================================================

def load_dataset(filepath="dataset/dataset-tickets-multi-lang-4-20k.csv"):
    """Loads real support ticket dataset directly from dataset CSV file."""
    if not os.path.exists(filepath) and os.path.exists("tickets_dataset.csv"):
        filepath = "tickets_dataset.csv"
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
        if 'language' in df.columns:
            df = df[df['language'] == 'en'].copy()
        
        if 'queue' in df.columns and 'category' not in df.columns:
            def map_cat(row):
                tags = str(row.get('tag_1', '')) + ' ' + str(row.get('tag_2', '')) + ' ' + str(row.get('tag_3', '')) + ' ' + str(row.get('tag_4', ''))
                queue = str(row.get('queue', ''))
                if queue == 'Billing and Payments' or 'Billing' in tags or 'Payment' in tags:
                    return 'Billing'
                elif queue == 'Human Resources' or 'HR' in tags or 'salary' in str(row.get('subject', '')).lower():
                    return 'HR'
                elif any(t in tags for t in ['Bug', 'Crash', 'Outage', 'DataLoss', 'SyncIssue', 'Integration', 'Performance', 'Hardware', 'API', 'Network']) or queue in ['Technical Support', 'IT Support', 'Product Support', 'Service Outages and Maintenance']:
                    return 'Technical'
                else:
                    return 'General'
            df['category'] = df.apply(map_cat, axis=1)

    df = df.dropna(subset=['category', 'subject', 'body'])
    df["text"] = df["subject"].fillna('') + " . " + df["body"].fillna('')
    return df


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "through", "during", "before", "after",
    "above", "below", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "to", "from", "also", "please", "kindly"
}


def clean_text(text):
    """
    Clean raw ticket text:
    - Lowercase
    - Remove URLs, emails, numbers, punctuation
    - Remove extra whitespace
    - Remove stopwords
    """
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)


# ============================================================
# 3. PRIORITY TAGGING (keyword-based rules)
# ============================================================

URGENT_KEYWORDS = {
    "urgent", "immediately", "asap", "critical", "down", "not working", "broken",
    "crash", "crashes", "crashed", "error", "failed", "failure", "dispute",
    "unauthorized", "fraud", "emergency", "outage", "downtime", "blocked",
    "cannot access", "unable to", "deadline", "overdue", "refund", "chargeback"
}


def tag_priority(text):
    """Returns 'Urgent' or 'Normal' based on keyword presence."""
    text_lower = str(text).lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            return "Urgent"
    return "Normal"


# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================

class TicketCategorizer:
    """
    A complete real-time support ticket categorization pipeline.
    
    MODEL CHOICE & FEATURE REPRESENTATION REASONING:
    1. Feature Representation (TF-IDF):
       - Converts raw text into numerical feature vectors.
       - TF-IDF (Term Frequency-Inverse Document Frequency) is chosen over simple Bag-of-Words
         because it penalizes commonly occurring fluff words while amplifying rare, highly
         discriminative department keywords (e.g. "invoice", "crash", "resignation").
       - Uses (1, 2) n-grams to capture key multi-word phrases like "not working" or "credit card".

    2. Model Selection (Naive Bayes & Logistic Regression):
       - Multinomial Naive Bayes: Extremely fast training and inference on high-dimensional sparse
         text data; highly robust baseline relying on Bayes' Theorem with word independence assumptions.
       - Logistic Regression: Provides well-calibrated class probability distributions required for
         confidence scoring, human review thresholding, and supports class balancing.
    """

    def __init__(self, confidence_threshold=0.60):
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95
        )
        self.nb_model = MultinomialNB(alpha=0.1)
        self.lr_model = LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced')
        self.classes_ = None

    def preprocess(self, texts):
        """Apply text cleaning to a list or single text."""
        if isinstance(texts, str):
            return clean_text(texts)
        return [clean_text(t) for t in texts]

    def fit(self, X_train, y_train):
        """Train both models on cleaned training data."""
        X_clean = self.preprocess(X_train)
        X_tfidf = self.vectorizer.fit_transform(X_clean)

        self.nb_model.fit(X_tfidf, y_train)
        self.lr_model.fit(X_tfidf, y_train)
        self.classes_ = self.nb_model.classes_
        return self

    def predict_single(self, text, model='logistic_regression'):
        """
        Predict a single ticket with confidence score and priority.
        Returns a dict with all details.
        """
        cleaned = self.preprocess(text)
        X_tfidf = self.vectorizer.transform([cleaned])

        # Choose model
        clf = self.lr_model if model == 'logistic_regression' else self.nb_model

        # Get prediction and probabilities
        pred = clf.predict(X_tfidf)[0]
        proba = clf.predict_proba(X_tfidf)[0]
        confidence = round(float(np.max(proba)) * 100, 2)

        # Human review check
        needs_review = confidence < (self.confidence_threshold * 100)

        # Priority tagging
        priority = tag_priority(text)

        # All class probabilities
        all_probs = {
            str(cls): round(float(p) * 100, 2)
            for cls, p in zip(self.classes_, proba)
        }
        all_probs = dict(sorted(all_probs.items(), key=lambda x: -x[1]))

        return {
            "input_text": text[:120] + "..." if len(text) > 120 else text,
            "predicted_category": pred,
            "confidence_percent": confidence,
            "needs_human_review": needs_review,
            "priority": priority,
            "all_probabilities": all_probs,
            "model_used": model,
            "cleaned_text": cleaned[:200]
        }

    def predict_batch(self, texts, model='logistic_regression'):
        """Predict multiple tickets at once."""
        return [self.predict_single(t, model) for t in texts]

    def evaluate(self, X_test, y_test, model='logistic_regression'):
        """Run full evaluation on test set."""
        X_clean = self.preprocess(X_test)
        X_tfidf = self.vectorizer.transform(X_clean)
        clf = self.lr_model if model == 'logistic_regression' else self.nb_model

        y_pred = clf.predict(X_tfidf)

        print("=" * 60)
        print(f"  EVALUATION REPORT — {model.upper().replace('_', ' ')}")
        print("=" * 60)
        print(f"\nAccuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 60)

    def save_model(self, filepath="model.joblib"):
        """Save the trained model pipeline to disk."""
        joblib.dump(self, filepath)
        print(f"✅ Model successfully saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath="model.joblib"):
        """Load a pre-trained model pipeline from disk."""
        model = joblib.load(filepath)
        print(f"✅ Model successfully loaded from: {filepath}")
        return model


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  AUTO EMAIL / TICKET CATEGORIZER")
    print("  AI/ML Intern Assessment — Fobes Skill Itech")
    print("=" * 60 + "\n")

    # Load data directly from real dataset file
    df = load_dataset("dataset/dataset-tickets-multi-lang-4-20k.csv")
    print(f"Real Dataset loaded from dataset folder: {len(df)} tickets")
    print(f"Category distribution:")
    print(df['category'].value_counts().to_string())
    print()

    # Train-test split (80/20)
    X = df["text"].tolist()
    y = df["category"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Initialize and train
    categorizer = TicketCategorizer(confidence_threshold=0.60)
    categorizer.fit(X_train, y_train)

    # Save trained model artifact to disk
    categorizer.save_model("model.joblib")

    # Evaluate both models
    categorizer.evaluate(X_test, y_test, model='naive_bayes')
    categorizer.evaluate(X_test, y_test, model='logistic_regression')

    # ========================================================
    # 6. PREDICT 5 NEW UNSEEN TICKETS
    # ========================================================
    new_tickets = [
        "Urgent: My credit card was charged $200 twice for the same transaction. I need a refund immediately or I will dispute.",
        "The mobile app keeps crashing whenever I open the settings page. This started after the latest update on Android 14.",
        "I would like to know how many casual leaves I have left for this quarter. Also, when is the next holiday?",
        "Does your platform support integration with Zapier? I couldn't find it in the documentation.",
        "Our production API is down and returning 503 errors. This is critical — all our customers are affected."
    ]

    print("\n" + "=" * 60)
    print("  PREDICTIONS ON 5 NEW UNSEEN TICKETS")
    print("=" * 60 + "\n")

    for i, ticket in enumerate(new_tickets, 1):
        result = categorizer.predict_single(ticket, model='logistic_regression')
        review_flag = "⚠️  HUMAN REVIEW" if result["needs_human_review"] else "✅ AUTO-ROUTED"
        priority_flag = "🔴 URGENT" if result["priority"] == "Urgent" else "🟢 Normal"

        print(f"Ticket #{i}")
        print(f"  Text      : {result['input_text']}")
        print(f"  Category  : {result['predicted_category']}")
        print(f"  Confidence: {result['confidence_percent']}%")
        print(f"  Priority  : {priority_flag}")
        print(f"  Status    : {review_flag}")
        print(f"  Probs     : {result['all_probabilities']}")
        print("-" * 60)

    # ========================================================
    # 7. EDGE CASE: LOW CONFIDENCE TICKET
    # ========================================================
    print("\n" + "=" * 60)
    print("  EDGE CASE: AMBIGUOUS / LOW-CONFIDENCE TICKET")
    print("=" * 60 + "\n")

    ambiguous_ticket = "I have a question about something related to my account and also a technical problem with the login."
    edge_result = categorizer.predict_single(ambiguous_ticket, model='logistic_regression')
    print(f"Text: {ambiguous_ticket}")
    print(f"Predicted: {edge_result['predicted_category']} ({edge_result['confidence_percent']}%)")
    print(f"Needs Human Review: {'YES' if edge_result['needs_human_review'] else 'NO'}")
    print(f"All probabilities: {edge_result['all_probabilities']}")

    # ========================================================
    # 8. REFLECTION NOTE
    # ========================================================
    print("\n" + "=" * 60)
    print("  REFLECTION: WHAT WOULD I IMPROVE WITH MORE TIME?")
    print("=" * 60)
    reflection = """
1. DATA: With more data, I'd collect thousands of real tickets instead of 50 dummy
   samples. More data = better generalization and higher confidence scores.

2. MODEL: I'd experiment with ensemble methods (VotingClassifier combining NB + LR + SVM)
   and potentially fine-tune a lightweight transformer like DistilBERT for marginal gains.

3. FEATURES: Add metadata features — ticket length, presence of attachments, sender domain,
   time-of-day — alongside TF-IDF to improve accuracy.

4. THRESHOLD: The 60% threshold is heuristic-based. With a validation set, I'd optimize
   it to minimize false auto-routes while keeping human-review queue manageable.

5. DEPLOYMENT: Package as a REST API (FastAPI) with async queue processing instead of
   a single script, enabling real-time enterprise helpdesk integration.
"""
    print(reflection)

    print("\n✅ Assessment complete. Ready for submission.")
    print("=" * 60)


if __name__ == "__main__":
    main()
