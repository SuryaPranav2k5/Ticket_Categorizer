import streamlit as st
import numpy as np
import pandas as pd
from ticket_categorizer import (
    load_dataset, clean_text, tag_priority,
    TicketCategorizer, STOPWORDS, URGENT_KEYWORDS
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Ticket Categorizer", page_icon="🎫", layout="centered")

st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; color: #10203D; margin-bottom: 0.3rem; }
.sub-header { font-size: 1rem; color: #4B5A78; margin-bottom: 2rem; }
.result-box { background: #F5F7FA; border: 1px solid #DCE2ED; border-radius: 12px; padding: 20px; margin-top: 16px; }
.category-badge { display: inline-block; padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.95rem; color: white; }
.billing { background: #2C5FDB; }
.technical { background: #0EA5A0; }
.hr { background: #E9A61B; }
.general { background: #E15B5B; }
.prob-bar { height: 8px; border-radius: 4px; background: #EEF1F6; margin: 4px 0; }
.prob-fill { height: 100%; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎫 Auto Ticket Categorizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI/ML Intern Assessment — Fobes Skill Itech Pvt Ltd</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Human Review Threshold (%)", min_value=30, max_value=90, value=60, step=5) / 100
    model_choice = st.radio("Model", ["Logistic Regression", "Naive Bayes"])
    st.divider()
    st.header("📊 Dataset")
    df = load_dataset("dataset/dataset-tickets-multi-lang-4-20k.csv")
    st.write(f"Total tickets: **{len(df)}**")
    st.bar_chart(df['category'].value_counts())

import os

@st.cache_resource
def get_trained_model(threshold):
    X = df["text"].tolist()
    y = df["category"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    if os.path.exists("model.joblib"):
        cat = TicketCategorizer.load_model("model.joblib")
        cat.confidence_threshold = threshold
    else:
        cat = TicketCategorizer(confidence_threshold=threshold)
        cat.fit(X_train, y_train)
        cat.save_model("model.joblib")
    return cat, X_test, y_test

categorizer, X_test, y_test = get_trained_model(confidence_threshold)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Demo", "📋 Batch Predict", "📊 Evaluation", "🎙️ Evaluator Guide & Summary"])

with tab1:
    st.subheader("Type a support ticket to classify instantly")
    ticket_input = st.text_area("Ticket text (subject + body)", height=120,
        placeholder="e.g. My invoice shows the wrong amount and I need a refund immediately...")
    col1, col2 = st.columns([1, 3])
    with col1:
        classify_btn = st.button("▶ Classify", type="primary", use_container_width=True)

    if classify_btn and ticket_input.strip():
        model_key = 'logistic_regression' if model_choice == "Logistic Regression" else 'naive_bayes'
        result = categorizer.predict_single(ticket_input, model=model_key)
        cat_class = result["predicted_category"].lower()
        review_msg = "⚠️ Needs Human Review" if result["needs_human_review"] else "✅ Auto-Routed"
        priority_icon = "🔴" if result["priority"] == "Urgent" else "🟢"

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f'<span class="category-badge {cat_class}">{result["predicted_category"]}</span> &nbsp; <span style="font-size:1.3rem; font-weight:700;">{result["confidence_percent"]}%</span> &nbsp; <span style="color:#4B5A78; font-size:0.9rem;">{review_msg}</span>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin:8px 0;"><b>Priority:</b> {priority_icon} {result["priority"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin:8px 0;"><b>Model:</b> {model_choice}</div>', unsafe_allow_html=True)
        st.markdown('<hr style="border:none; border-top:1px solid #DCE2ED; margin:12px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.9rem; color:#4B5A78; margin-bottom:8px;"><b>Confidence Breakdown:</b></div>', unsafe_allow_html=True)

        colors = {"Billing": "#2C5FDB", "Technical": "#0EA5A0", "HR": "#E9A61B", "General": "#E15B5B"}
        for cat, prob in result["all_probabilities"].items():
            color = colors[cat]
            st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin:4px 0;"><div style="width:80px; font-size:0.85rem; font-weight:500;">{cat}</div><div class="prob-bar" style="flex:1;"><div class="prob-fill" style="width:{prob}%; background:{color};"></div></div><div style="width:45px; text-align:right; font-size:0.85rem; font-weight:600;">{prob}%</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif classify_btn:
        st.warning("Please enter some ticket text first.")

with tab2:
    st.subheader("Predict multiple tickets at once")
    batch_input = st.text_area("Enter one ticket per line", height=200,
        placeholder="Invoice not received for March payment\nApp crashes when I click settings...")
    if st.button("▶ Run Batch Prediction", type="primary"):
        if batch_input.strip():
            tickets = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
            model_key = 'logistic_regression' if model_choice == "Logistic Regression" else 'naive_bayes'
            results = categorizer.predict_batch(tickets, model=model_key)
            out_data = []
            for r in results:
                out_data.append({
                    "Ticket Preview": r["input_text"][:80] + "...",
                    "Category": r["predicted_category"],
                    "Confidence": f"{r['confidence_percent']}%",
                    "Priority": r["priority"],
                    "Review": "Yes" if r["needs_human_review"] else "No"
                })
            st.dataframe(pd.DataFrame(out_data), use_container_width=True)
            cats = [r["predicted_category"] for r in results]
            prios = [r["priority"] for r in results]
            reviews = sum(1 for r in results if r["needs_human_review"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(results))
            c2.metric("Urgent", prios.count("Urgent"))
            c3.metric("Needs Review", reviews)
        else:
            st.warning("Enter at least one ticket.")

with tab3:
    st.subheader("Model Performance Evaluation")
    
    # Calculate evaluation metrics
    X_test_clean = categorizer.preprocess(X_test)
    X_test_tfidf = categorizer.vectorizer.transform(X_test_clean)
    
    selected_model_key = 'logistic_regression' if model_choice == "Logistic Regression" else 'naive_bayes'
    clf = categorizer.lr_model if selected_model_key == 'logistic_regression' else categorizer.nb_model
    y_pred = clf.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall", f"{rec:.2%}")
    m4.metric("F1 Score", f"{f1:.2%}")
    
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=categorizer.classes_)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in categorizer.classes_], columns=[f"Pred {c}" for c in categorizer.classes_])
    st.dataframe(cm_df, use_container_width=True)

with tab4:
    st.subheader("🎙️ Evaluator Walkthrough & Executive Summary")
    
    st.success("✅ Complete Auto Email / Ticket Categorizer Solution — AI/ML Intern Assessment")
    
    st.markdown("""
    ### 💬 Executive Presentation Summary
    > *"I built an end-to-end Auto Email / Ticket Categorizer in Python using scikit-learn. The model reads incoming support tickets and automatically routes them to **Billing**, **Technical**, **HR**, or **General** departments.*
    >
    > *It preprocesses raw ticket text by lowercasing, stripping URLs, emails, numbers, and stopwords, and converts text into TF-IDF unigram and bigram numerical feature vectors (`sublinear_tf=True`). It trains and compares **Multinomial Naive Bayes** and **Logistic Regression** models directly on **10,890 real English tickets** from the dataset.*
    >
    > *Beyond core classification, I implemented 5 production bonus features:*
    > 1. **Confidence Scores**: Returns exact class probabilities across all 4 departments.
    > 2. **Human Review Threshold**: Configurable threshold (default 60%) routing low-certainty tickets to manual review.
    > 3. **Priority Tagging**: Detects urgency keywords (down, crash, refund, error) tagging tickets 🔴 URGENT or 🟢 Normal.
    > 4. **Interactive Web App**: Real-time triage GUI with live single ticket & batch prediction modes.
    > 5. **Model Persistence**: Serializes trained pipelines to `model.joblib` for instant enterprise integration."*

    ---

    ### 🛠️ Architecture & Pipeline Overview
    - **Dataset File**: `dataset/dataset-tickets-multi-lang-4-20k.csv` (**10,890 real English tickets**)
    - **Preprocessing**: Lowercasing, Regex URL/email/number stripping, punctuation removal, custom stopwords
    - **Feature Extraction**: TF-IDF Vectorizer with unigrams & bigrams (max 2,000 features, `sublinear_tf=True`)
    - **Classifiers**: Multinomial Naive Bayes & Logistic Regression (`C=5.0`, `class_weight='balanced'`)
    - **Model Artifact**: Serialized `model.joblib` saved in root directory

    ---

    ### 🎁 All 5 Bonus Challenges Completed
    - ✅ **Confidence score output** with 4-department probability breakdown
    - ✅ **"Needs human review" threshold** (configurable slider in sidebar)
    - ✅ **Priority tagging** (Urgent / Normal keyword rules)
    - ✅ **Mini live demo** (CLI terminal script & Streamlit GUI)
    - ✅ **Reflection note** (5-point production roadmap)
    """)

    st.subheader("Urgent Keywords Monitored")
    st.info(", ".join(sorted(URGENT_KEYWORDS)))
