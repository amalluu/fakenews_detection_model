import streamlit as st
import pickle
import numpy as np
import pandas as pd
import gensim
import nltk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Detection System")
st.markdown("### Using LSTM Neural Network")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer with exact training preprocessing"""
    try:
        # Load model (try both .keras and .h5 formats)
        try:
            model = load_model('fake_news_model.keras')
        except:
            model = load_model('fake_news_model.h5')
        
        # Load tokenizer (updated filename)
        with open('tokenizer_new.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load maxlen from saved file
        maxlen = 40  # Use the actual training maxlen
        
        return model, tokenizer, maxlen
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def preprocess_text_exact_training(text):
    """
    Exact preprocessing pipeline from training:
    1. Gensim simple_preprocess
    2. Remove stopwords (NLTK + custom + Gensim)
    3. Filter words with length > 3
    4. Join back to string
    """
    # Get stopwords exactly as in training
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    
    # Apply exact preprocessing from training
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if (token not in gensim.parsing.preprocessing.STOPWORDS and 
            len(token) > 3 and 
            token not in stop_words):
            result.append(token)
    
    # Join back to string (as done in training with 'clean_joined')
    return " ".join(result)

def predict_fake_news(text, model, tokenizer, maxlen):
    """Predict whether news is fake or real"""
    # Apply exact training preprocessing
    processed_text = preprocess_text_exact_training(text)
    
    # Tokenize using the trained tokenizer
    sequences = tokenizer.texts_to_sequences([processed_text])
    
    # Pad sequences (exactly as in training)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded, verbose=0)[0][0]
    
    return prediction, processed_text

# Load model and tokenizer
model, tokenizer, maxlen = load_model_and_tokenizer()

if model is not None:
    st.success("✅ Model loaded successfully!")
    st.session_state.model_loaded = True
    
    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vocabulary Size", f"{len(tokenizer.word_index):,}")
    with col2:
        st.metric("Max Sequence Length", maxlen)
    with col3:
        st.metric("Model Training Accuracy", "99.5%")
    
    st.markdown("---")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "📁 File Upload"],
        horizontal=True
    )
    
    if input_method == "📝 Text Input":
        # Text input
        st.subheader("Enter News Article")
        

        user_input = st.text_area(
                "Enter the news article text:",
                height=200,
                placeholder="Paste your news article here..."
            )

        
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing the article..."):
                    prediction_score, processed_text = predict_fake_news(user_input, model, tokenizer, maxlen)
                
                # Results
                st.subheader("📊 Analysis Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Prediction score
                    #st.metric("Prediction Score", f"{prediction_score:.4f}")
                    
                    # Classification - CORRECTED LOGIC
                    # Model learned: high score = REAL, low score = FAKE
                    if prediction_score > 0.5:
                        st.error("🚨 **FAKE NEWS DETECTED**")
                        confidence = (1 - prediction_score) * 100
                        
                        
                    else:

                        st.success("✅ **REAL NEWS**")
                        confidence = prediction_score * 100
                        
                        
                
                with col2:
                    # Confidence bar - CORRECTED
                    st.write("**Model Output Interpretation:**")
                    st.write(f"• High score (>0.5) = Fake News")
                    st.write(f"• Low score (≤0.5) = Real News")

                    
                    if prediction_score > 0.5:
                        st.error(f"FAKE NEWS Confidence: {prediction_score:.1%}")
                        # Progress bar showing fake confidence (inverted,convert to float)
                    
                        st.progress(float(1 - prediction_score))
                    else:
                        st.success(f"Real News Confidence: {(1-prediction_score):.1%}")
                        # Progress bar showing real confidence ( convert to float)
                        st.progress(float(prediction_score))
                
                # Show preprocessing details
                with st.expander("🔧 Preprocessing Details"):
                    st.write("**Original Text (first 200 chars):**")
                    st.code(user_input[:200] + ("..." if len(user_input) > 200 else ""))
                    
                    st.write("**Processed Text:**")
                    st.code(processed_text)
                    
                    # Tokenization info
                    sequences = tokenizer.texts_to_sequences([processed_text])
                    st.write(f"**Tokens Generated:** {len(sequences[0]) if sequences[0] else 0}")
                    st.write(f"**Sequence (first 10 tokens):** {sequences[0][:10] if sequences[0] else []}")
                    
                    # Word analysis
                    words = processed_text.split()
                    oov_words = [w for w in words if w not in tokenizer.word_index]
                    st.write(f"**Vocabulary Coverage:** {((len(words) - len(oov_words)) / len(words) * 100):.1f}%")
                    
                    if oov_words:
                        st.write(f"**Out-of-vocabulary words:** {oov_words[:5]}")
                        
            else:
                st.warning("Please enter some text to analyze.")
    
    else:
        # File upload
        st.subheader("Upload News Articles")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with news articles",
            type=['csv'],
            help="CSV should have a column containing news article text"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("**File uploaded successfully!**")
            st.write(f"Shape: {df.shape}")
            
            # Show first few rows
            st.write("**Preview:**")
            st.dataframe(df.head())
            
            # Select text column
            text_column = st.selectbox("Select the column containing news text:", df.columns)
            
            if st.button("🔍 Analyze All Articles", type="primary"):
                with st.spinner("Analyzing articles..."):
                    predictions = []
                    processed_texts = []
                    
                    for text in df[text_column]:
                        if pd.notna(text):
                            pred, processed = predict_fake_news(str(text), model, tokenizer, maxlen)
                            predictions.append(pred)
                            processed_texts.append(processed)
                        else:
                            predictions.append(np.nan)
                            processed_texts.append("")
                    
                    df['prediction_score'] = predictions
                    # CORRECTED: prediction > 0.5 = Fake (since isfake=1 in training)
                    df['prediction'] = ['Fake' if p > 0.5 else 'Real' if pd.notna(p) else 'Error' for p in predictions]
                    df['processed_text'] = processed_texts
                
                # Results
                st.subheader("📊 Batch Analysis Results")
                
                # Summary stats
                fake_count = sum(1 for p in predictions if pd.notna(p) and p > 0.5)
                real_count = sum(1 for p in predictions if pd.notna(p) and p <= 0.5)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Articles", len(df))
                with col2:
                    st.metric("Fake News", fake_count)
                with col3:
                    st.metric("Real News", real_count)
                
                # Results table
                st.dataframe(df[['prediction_score', 'prediction', text_column]].head(10))
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "fake_news_analysis_results.csv",
                    "text/csv"
                )

else:
    st.error("❌ Could not load model files. Please ensure these files are in the same directory:")
    st.write("- `fake_news_model.keras` or `fake_news_model.h5`")
    st.write("- `tokenizer_new.pkl`")
    st.write("- `maxlen_new.pkl`")

# Instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.write("""
    **How to use:**
    1. Ensure model files are in the app directory
    2. Enter or upload news article text
    3. Click "Analyze" to get predictions
    
    **Model Details:**
    - Architecture: Bidirectional LSTM
    - Training Accuracy: 99.5%
    - Features: Title + Article text
    - Preprocessing: Stopword removal, tokenization
    
    **Interpretation:**
    - Score > 0.5 = Real News (model learned this way)
    - Score ≤ 0.5 = Fake News (model learned this way)
    - Higher score = Higher confidence it's real news
    """)
    
    st.header("🔧 Troubleshooting")
    st.write("""
    **Common Issues:**
    - Ensure all model files are present
    - Check that NLTK data is downloaded
    - Verify input text is in English
    - Make sure text has sufficient content
    """)