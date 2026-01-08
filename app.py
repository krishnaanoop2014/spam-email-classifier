import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.title("📧 Spam Email Classifier")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/spam.csv")

data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Data preprocessing
X = data['message']
y = data['label']  # 'spam' or 'ham'

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.success(f"{round(accuracy*100, 2)}%")

# User input
st.subheader("Test Your Email")
user_input = st.text_area("Enter email text here:")

if st.button("Predict"):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    st.warning(f"Prediction: **{prediction.upper()}**")
