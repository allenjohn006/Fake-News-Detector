# File: app_fixed.py

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
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
        h1, h2, h3 { color: #ef4444; }
    </style>
""", unsafe_allow_html=True)

# ==================== CACHE MODEL ====================
@st.cache_resource
def load_and_train_model():
    """Load your CSV files and return dataframes (labels added)."""
    try:
        # UPDATE THESE PATHS TO YOUR CSV LOCATIONS
        df_fake = pd.read_csv(r"C:\Users\allen\OneDrive\Desktop\Fake-news-detector\Fake.csv")
        df_true = pd.read_csv(r"C:\Users\allen\OneDrive\Desktop\Fake-news-detector\True.csv")

        # Add labels
        df_fake['label'] = 0
        df_true['label'] = 1

        return df_fake, df_true

    except FileNotFoundError as e:
        st.error(f"CSV files not found: {e}")
        st.error("Check the file paths in app.py")
        return None, None

# ==================== TEXT CLEANING ====================
def clean_text(text):
    """Clean text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== FEATURE EXTRACTION ====================
def extract_features(text):
    """Extract linguistic features"""
    cleaned = clean_text(text)
    words = cleaned.split()

    stop_words = set(stopwords.words('english'))
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

# ==================== MAIN APP ====================
def main():
    st.title("📰 Fake News Detector")
    st.markdown("AI-Powered Misinformation Detection using Data Science")

    # Load datasets
    with st.spinner("Loading datasets and training model..."):
        df_fake, df_true = load_and_train_model()

    if df_fake is None:
        st.stop()

    # Combine data
    df_fake_copy = df_fake.copy()
    df_true_copy = df_true.copy()

    df_fake_copy['text_combined'] = df_fake_copy['title'].fillna('') + ' ' + df_fake_copy['text'].fillna('')
    df_true_copy['text_combined'] = df_true_copy['title'].fillna('') + ' ' + df_true_copy['text'].fillna('')

    df = pd.concat([df_fake_copy, df_true_copy], ignore_index=True)

    # Clean texts
    df['text_clean'] = df['text_combined'].apply(clean_text)

    # Train model (show spinner while training)
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test = train_test_split(
            df['text_clean'], df['label'], test_size=0.2, random_state=42
        )

        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_tfidf, y_train)

    # Get metrics
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detector", "📊 Dataset", "📈 Performance", "ℹ About"])

    # ==================== TAB 1: DETECTOR ====================
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
            st.rerun()

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
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Text Length", features['Text Length'])
                with col2:
                    st.metric("Word Count", features['Word Count'])
                with col3:
                    st.metric("Unique Words", features['Unique Words'])
                with col4:
                    st.metric("Avg Word Length", features['Avg Word Length'])

            st.info("⚠ Always verify with multiple sources before trusting predictions")

    # ==================== TAB 2: DATASET ====================
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
            fig, ax = plt.subplots(facecolor='#0f172a')
            ax.set_facecolor('#1e293b')
            sizes = [len(df_fake), len(df_true)]
            colors = ['#ef4444', '#22c55e']
            ax.pie(sizes, labels=['Fake', 'True'], colors=colors, autopct='%1.1f%%',
                   textprops={'color': 'white', 'weight': 'bold'})
            st.pyplot(fig)

        with col2:
            st.subheader("Text Length Distribution")
            fig, ax = plt.subplots(facecolor='#0f172a')
            ax.set_facecolor('#1e293b')
            df_fake_copy['text_length'] = df_fake_copy['text'].apply(len)
            df_true_copy['text_length'] = df_true_copy['text'].apply(len)
            ax.hist(df_fake_copy['text_length'], bins=50, alpha=0.7, label='Fake', color='#ef4444')
            ax.hist(df_true_copy['text_length'], bins=50, alpha=0.7, label='True', color='#22c55e')
            ax.set_xlabel('Characters', color='white')
            ax.set_ylabel('Count', color='white')
            ax.legend()
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            st.pyplot(fig)

    # ==================== TAB 3: PERFORMANCE ====================
    with tab3:
        st.header("Model Performance")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("Precision", f"{precision*100:.2f}%")
        with col3:
            st.metric("Recall", f"{recall*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{f1*100:.2f}%")

        st.markdown("---")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True, ax=ax,
                   xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'],
                   cbar_kws={'label': 'Count'})
        ax.set_ylabel('True Label', color='white')
        ax.set_xlabel('Predicted Label', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

    # ==================== TAB 4: ABOUT ====================
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

# ----- FIXED: correct python entrypoint name -----
if __name__ == "__main__":
    main()
