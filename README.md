# Spam Email Classifier

A simple machine learning project that classifies emails as **spam** or **not spam** using a small dataset and Naive Bayes. Deployable on Streamlit for web usage.

---

## 🚀 Features
- Upload email text or use sample emails
- Predict spam or not spam
- Simple, interactive UI
- Fully deployable on Streamlit Cloud

---

## 🧠 Tech Stack
- Python
- Pandas
- Scikit-learn (Naive Bayes)
- Streamlit

---

## 📂 Dataset
Small sample dataset: spam vs ham (not spam) emails.  
Included in `data/spam.csv`  
Source: [Kaggle – Spam SMS Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🛠 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
