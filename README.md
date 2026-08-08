# 🎫 Auto Email / Ticket Categorizer

**AI/ML Intern Assessment — Fobes Skill Itech Pvt Ltd**  
A lightweight NLP classification engine that reads incoming support tickets (subject + body text) and automatically routes them to the correct department with confidence scoring, human-review thresholding, and priority tagging.

🔗 **GitHub Repository Link**: [https://github.com/SuryaPranav2k5/Ticket_Categorizer.git](https://github.com/SuryaPranav2k5/Ticket_Categorizer.git)

> 📌 **Submission Note**: If submitting a single file, please refer to the GitHub repository link above for full source code (`ticket_categorizer.py`, `app.py`), dataset files, pre-trained model binaries (`model.joblib`), and live Streamlit demo web app.

---

## 🎯 Target Categories
- 💳 **Billing** (invoices, refunds, payment failures, overcharges, credit cards)
- 🛠️ **Technical** (app crashes, API 500 errors, database timeouts, SSO, webhooks)
- 👥 **HR** (leave balance, salary slips, WFH policy, health insurance claims, appraisals)
- 💬 **General** (product feedback, feature requests, partnership inquiries, office address)

---

## 🚀 Key Features & Bonus Objectives Completed

| Feature / Objective | Status | Description |
| :--- | :---: | :--- |
| **Text Preprocessing** | ✅ Built | `clean_text()` lowercases, strips URLs, emails, numbers, punctuation, extra whitespace, and custom stopwords. |
| **Feature Representation** | ✅ Built | `TfidfVectorizer` with unigrams + bigrams (`ngram_range=(1, 2)`), max 2,000 features, `sublinear_tf=True`. |
| **Model Choice Reasoning** | ✅ Built | Evaluates and compares **Multinomial Naive Bayes** (fast sparse baseline) and **Logistic Regression** (`C=5.0`, `class_weight='balanced'`). |
| **Evaluation Metrics** | ✅ Built | Computes Accuracy, Weighted Precision, Recall, F1-Score, Classification Report, and Confusion Matrix. |
| **Real-Time Classifier** | ✅ Built | `predict_single()` classifies single incoming tickets on demand in milliseconds. |
| **Edge-Case Fallback** | ✅ Built | Low-confidence tickets are routed to a manual human-review queue (`needs_human_review: True`). |
| **🎁 Bonus 1: Confidence Scores** | ✅ Built | Outputs top-class confidence percentage + full 4-department probability breakdown. |
| **🎁 Bonus 2: Human Review Threshold** | ✅ Built | Configurable 60% confidence threshold fallback for ambiguous tickets. |
| **🎁 Bonus 3: Priority Tagging** | ✅ Built | Detects urgency signals (`urgent`, `critical`, `crash`, `down`, `refund`) tagging tickets **🔴 URGENT** or **🟢 Normal**. |
| **🎁 Bonus 4: Mini Live Demo** | ✅ Built | Includes **both** a rich CLI console output (`python ticket_categorizer.py`) and an interactive Streamlit Web App (`streamlit run app.py`). |
| **🎁 Bonus 5: Reflection Note** | ✅ Built | 5-point production roadmap covering dataset scaling, transformer fine-tuning, metadata features, threshold tuning, and FastAPI deployment. |

---

## 📊 Dataset & Model Evaluation Performance

- **Dataset Source**: [Kaggle Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) by Tobias Bück.
- **Training Subset**: **10,890 real English support tickets** loaded from `dataset/dataset-tickets-multi-lang-4-20k.csv`.

### 📈 Evaluation Metrics (Tested on 2,178 Unseen Real Test Tickets)

| Model Metric | Naive Bayes | Logistic Regression |
| :--- | :---: | :---: |
| **Accuracy** | **75.21%** | **73.46%** |
| **Weighted Precision** | **81.59%** | **81.05%** |
| **Weighted Recall** | **75.21%** | **73.46%** |
| **Weighted F1-Score** | **76.49%** | **75.95%** |

### 🎯 Department-Level Precision Highlights
- 💳 **Billing Precision**: **93.00%**
- 🛠️ **Technical Precision**: **92.00%**
- 👥 **HR Precision**: **100.00%**

---

## 💻 How to Run

### Environment Setup (Using `uv`)
```powershell
# Create virtual environment
uv venv .venv

# Install dependencies
uv pip install -r requirements.txt
```

### Option 1: Command-Line Evaluation & Live Predictions
```powershell
.\.venv\Scripts\python.exe ticket_categorizer.py
```
*Loads 10,890 real tickets, trains/evaluates models, saves `model.joblib`, runs 5 live test predictions, and displays the reflection note.*

### Option 2: Streamlit Interactive Web Application
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
*Launches an interactive browser GUI featuring:*
- 🚀 **Live Demo Tab**: Type any custom support ticket and see instant category routing, confidence breakdown, and priority flags.
- 📋 **Batch Predict Tab**: Paste multiple tickets to classify in bulk.
- 📊 **Evaluation Metrics Tab**: Live accuracy/precision cards and interactive Confusion Matrix.
- 🎙️ **Evaluator Guide & Summary Tab**: Executive presentation summary and demo walkthrough script.

---

## 📁 Repository Directory Structure

| File / Folder | Description |
| :--- | :--- |
| 📄 `ticket_categorizer.py` | Main Python script with full ML pipeline, dataset loader, model training, evaluation, and CLI predictions. |
| 🌐 `app.py` | Interactive Streamlit web application with 4 tabs and live triage tools. |
| 📁 `dataset/dataset-tickets-multi-lang-4-20k.csv` | Real dataset file containing 10,890 English support tickets. |
| 💾 `model.joblib` | Pre-trained model pipeline artifact serialized with `joblib`. |
| 📋 `requirements.txt` | Dependency requirements (`pandas`, `numpy`, `scikit-learn`, `streamlit`). |
| 📖 `README.md` | Full project documentation and execution guide. |

---

## 🧠 Model Choice & Feature Engineering Reasoning

1. **Feature Representation (TF-IDF)**:
   - TF-IDF (Term Frequency-Inverse Document Frequency) is chosen over simple Bag-of-Words because it down-weights common fluff words while amplifying highly discriminative department terms (e.g., *"invoice"*, *"crash"*, *"leave"*, *"refund"*).
   - Uses `ngram_range=(1, 2)` to capture critical multi-word phrases like *"not working"*, *"credit card"*, or *"password reset"*.
   - Uses `sublinear_tf=True` to dampen word frequency scaling (`1 + log(tf)`).

2. **Model Selection (Naive Bayes vs. Logistic Regression)**:
   - **Multinomial Naive Bayes**: Fast, lightweight baseline ideal for high-dimensional sparse text vectors, achieving **81.59% Weighted Precision**.
   - **Logistic Regression**: Calibrated probabilities (`C=5.0`, `class_weight='balanced'`) providing smooth confidence scores for human review thresholding.
