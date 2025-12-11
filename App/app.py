# app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------- NLTK: download only if missing ----------------
def ensure_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }
    for pkg, loc in resources.items():
        try:
            nltk.data.find(loc)
        except LookupError:
            nltk.download(pkg)

ensure_nltk_resources()

# ---------------- Page config & styling ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
    <style>
        h1, h2, h3 { color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

# ---------------- Locate Data directory (relative paths) ----------------
def find_data_dir():
    """
    Look for a 'Data' directory in the current file directory, parent, or grandparent.
    Returns a Path object (not necessarily existing).
    """
    base = Path(__file__).resolve().parent
    candidates = [base, base.parent, base.parent.parent]
    for c in candidates:
        data_dir = c / "Data"
        if data_dir.exists() and data_dir.is_dir():
            return data_dir
    # fallback to base/Data (useful if running from repo root where Data is present)
    return base / "Data"

DATA_DIR = find_data_dir()

# ---------------- CACHED: load data ----------------
@st.cache_data(show_spinner=False)
def load_csvs(data_dir: Path):
    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        missing = []
        if not fake_path.exists():
            missing.append(str(fake_path))
        if not true_path.exists():
            missing.append(str(true_path))
        raise FileNotFoundError(f"Missing CSV files: {', '.join(missing)}")

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Add labels (0 = fake, 1 = true)
    df_fake['label'] = 0
    df_true['label'] = 1

    return df_fake, df_true

# ---------------- TEXT CLEANING & FEATURES ----------------
def clean_text(text):
    """Clean text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text):
    """Extract simple linguistic features used in the UI."""
    cleaned = clean_text(text)
    words = cleaned.split()

    # ensure stopwords resource available
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    stop_word_count = sum(1 for word in words if word in stop_words)

    try:
        blob = TextBlob(cleaned)
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        sentiment = 0
        subjectivity = 0

    return {
        'Text Length': len(cleaned),
        'Word Count': len(words),
        'Unique Words': len(set(words)),
        'Avg Word Length': round(np.mean([len(w) for w in words]) if words else 0, 2),
        'Stop Word Ratio': round(stop_word_count / len(words) if words else 0, 3),
        'Sentiment': round(sentiment, 3),
        'Subjectivity': round(subjectivity, 3)
    }

# ---------------- CACHED: vectorizer + model ----------------
@st.cache_resource(show_spinner=False)
def get_vectorizer_and_model(texts: pd.Series, labels: pd.Series):
    """
    Train TF-IDF + LogisticRegression once and cache the model and vectorizer.
    """
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, labels)
    return tfidf, model

# ---------------- Main app ----------------
def main():
    st.title("📰 Fake News Detector")
    st.markdown("AI-Powered Misinformation Detection using Data Science")

    # Show where we expect the Data folder (helpful for debugging)
    st.caption(f"Looking for CSVs in: `{DATA_DIR}`")

    # Load CSVs (cached)
    try:
        df_fake, df_true = load_csvs(DATA_DIR)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Put `Fake.csv` and `True.csv` inside a `Data/` folder next to this script or one level up.")
        st.stop()

    # Combine data
    df_fake_copy = df_fake.copy()
    df_true_copy = df_true.copy()

    # Make sure title/text columns exist (defensive)
    for d in (df_fake_copy, df_true_copy):
        if 'title' not in d.columns:
            d['title'] = ""
        if 'text' not in d.columns:
            d['text'] = ""

    df_fake_copy['text_combined'] = df_fake_copy['title'].fillna('') + ' ' + df_fake_copy['text'].fillna('')
    df_true_copy['text_combined'] = df_true_copy['title'].fillna('') + ' ' + df_true_copy['text'].fillna('')

    df = pd.concat([df_fake_copy, df_true_copy], ignore_index=True)

    # Clean texts
    df['text_clean'] = df['text_combined'].apply(clean_text)

    # Train model (cached resource)
    with st.spinner("Training/Loading model..."):
        tfidf, model = get_vectorizer_and_model(df['text_clean'], df['label'])

    # Evaluate once (using train_test_split for metrics display)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'], test_size=0.2, random_state=42
    )
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    # protect metrics in case of degenerate labels
    try:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    except Exception:
        precision = recall = f1 = 0.0

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detector", "📊 Dataset", "📈 Performance", "ℹ About"])

    # ---------------- TAB 1: DETECTOR ----------------
    with tab1:
        st.header("Check News Article")

        user_input = st.text_area(
            "Paste your news article:",
            height=200,
            placeholder="Enter news text here..."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            analyze_btn = st.button("🔍 Analyze News", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear", use_container_width=True)

        if clear_btn:
            st.experimental_rerun()

        if analyze_btn and user_input.strip():
            input_clean = clean_text(user_input)
            input_tfidf = tfidf.transform([input_clean])

            prediction = model.predict(input_tfidf)[0]
            probability = model.predict_proba(input_tfidf)[0]
            confidence = max(probability) * 100

            st.markdown("---")

            # Result
            if prediction == 0:
                st.error("❌ FAKE NEWS DETECTED", icon="🚨")
                st.metric("Confidence (Fake)", f"{probability[0]*100:.2f}%")
            else:
                st.success("✅ REAL NEWS DETECTED", icon="✔")
                st.metric("Confidence (Real)", f"{probability[1]*100:.2f}%")

            # Progress bar
            st.progress(confidence / 100)

            # Show features
            with st.expander("📝 Show Linguistic Features"):
                features = extract_features(user_input)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Text Length", features['Text Length'])
                with c2:
                    st.metric("Word Count", features['Word Count'])
                with c3:
                    st.metric("Unique Words", features['Unique Words'])
                with c4:
                    st.metric("Avg Word Length", features['Avg Word Length'])

            st.info("⚠ Always verify with multiple sources before trusting predictions")

    # ---------------- TAB 2: DATASET ----------------
    with tab2:
        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", f"{len(df):,}")
        with col2:
            st.metric("Fake News", f"{len(df_fake):,}")
        with col3:
            st.metric("True News", f"{len(df_true):,}")

        st.markdown("---")

        # Distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution")
            fig, ax = plt.subplots()
            sizes = [len(df_fake), len(df_true)]
            colors = ['#ef4444', '#22c55e']
            ax.pie(sizes, labels=['Fake', 'True'], colors=colors, autopct='%1.1f%%',
                   textprops={'weight': 'bold'})
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Text Length Distribution")
            fig, ax = plt.subplots()
            df_fake_copy['text_length'] = df_fake_copy['text'].fillna('').apply(len)
            df_true_copy['text_length'] = df_true_copy['text'].fillna('').apply(len)
            ax.hist(df_fake_copy['text_length'], bins=50, alpha=0.7, label='Fake')
            ax.hist(df_true_copy['text_length'], bins=50, alpha=0.7, label='True')
            ax.set_xlabel('Characters')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    # ---------------- TAB 3: PERFORMANCE ----------------
    with tab3:
        st.header("Model Performance")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
        with c2:
            st.metric("Precision", f"{precision*100:.2f}%")
        with c3:
            st.metric("Recall", f"{recall*100:.2f}%")
        with c4:
            st.metric("F1-Score", f"{f1*100:.2f}%")

        st.markdown("---")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax,
                    xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'],
                    cbar_kws={'label': 'Count'})
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- TAB 4: ABOUT ----------------
    with tab4:
        st.header("About This Project")

        st.markdown(f"""
        ### Data Science Methods Used:
        
        *1. Text Preprocessing*
        - Lowercase conversion
        - Special character removal
        - Extra whitespace cleanup
        
        *2. Feature Engineering*
        - TF-IDF Vectorization (5000 features)
        - Unigrams and bigrams (1-2 word combinations)
        - Linguistic features extraction
        
        *3. Machine Learning*
        - Algorithm: Logistic Regression
        - Training: 80% of data ({len(X_train):,} articles)
        - Testing: 20% of data ({len(X_test):,} articles)
        
        *4. Model Performance*
        - Accuracy: {accuracy*100:.2f}%
        - Precision: {precision*100:.2f}%
        - Recall: {recall*100:.2f}%
        - F1-Score: {f1*100:.2f}%
        
        ### Dataset Information
        - Total articles: {len(df):,}
        - Fake news: {len(df_fake):,}
        - True news: {len(df_true):,}
        """)

if __name__ == "__main__":
    main()
