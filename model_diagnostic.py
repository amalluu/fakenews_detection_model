import streamlit as st
import pickle
import numpy as np
import pandas as pd
import gensim
import nltk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔍 Complete Model Diagnostic")
st.write("Let's find ALL the issues causing 0.999 predictions")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@st.cache_resource
def load_all_files():
    """Load model and tokenizer"""
    try:
        # Try both formats
        try:
            model = load_model('fake_news_model.keras')
            st.success("✅ Loaded fake_news_model.keras")
        except:
            model = load_model('fake_news_model.h5')
            st.success("✅ Loaded fake_news_model.h5")
        
        # Load tokenizer
        with open('tokenizer_new.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        st.success("✅ Loaded tokenizer.pkl")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return None, None

# Load files
model, tokenizer = load_all_files()

if model and tokenizer:
    
    st.header("📊 Model Architecture Analysis")
    
    # Model summary
    with st.expander("Model Summary"):
        model_config = model.get_config()
        st.write("**Model Type:**", type(model).__name__)
        st.write("**Input Shape:**", model.input_shape)
        st.write("**Output Shape:**", model.output_shape)
        
        # Check if it's actually a binary classifier
        output_units = model.layers[-1].units if hasattr(model.layers[-1], 'units') else 'Unknown'
        output_activation = model.layers[-1].activation.__name__ if hasattr(model.layers[-1], 'activation') else 'Unknown'
        
        st.write(f"**Final Layer Units:** {output_units}")
        st.write(f"**Final Layer Activation:** {output_activation}")
        
        if output_units != 1 or output_activation != 'sigmoid':
            st.error("🚨 PROBLEM: Model architecture doesn't match binary classification!")
    
    st.header("🔤 Tokenizer Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vocabulary Size", len(tokenizer.word_index))
    with col2:
        st.metric("Total Words", getattr(tokenizer, 'num_words', 'All'))
    with col3:
        # Check if tokenizer was fitted properly
        sample_words = list(tokenizer.word_index.keys())[:10]
        st.write("**Sample Words:**")
        st.code(str(sample_words))
    
    st.header("🧪 Preprocessing Comparison")
    
    # Test different preprocessing approaches
    test_texts = {
        "Real News": "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. House of Representatives urged the Trump administration on Tuesday to quickly nominate conservative judges.",
        "Fake News": "BREAKING: Scientists discover coffee makes you immortal! Government conspiracy revealed!",
        "Short Text": "Trump wins election.",
        "Empty After Preprocessing": "The a an is was were"
    }
    
    def preprocess_exact_training(text):
        """Exact training preprocessing"""
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if (token not in gensim.parsing.preprocessing.STOPWORDS and 
                len(token) > 3 and 
                token not in stop_words):
                result.append(token)
        
        return " ".join(result)
    
    def preprocess_basic(text):
        """Basic preprocessing"""
        return text.lower()
    
    def preprocess_none(text):
        """No preprocessing"""
        return text
    
    preprocessing_methods = {
        "Training Method": preprocess_exact_training,
        "Basic (lowercase)": preprocess_basic,
        "None (raw)": preprocess_none
    }
    
    results_df = []
    
    for test_name, test_text in test_texts.items():
        st.subheader(f"📝 Testing: {test_name}")
        st.code(test_text[:100] + ("..." if len(test_text) > 100 else ""))
        
        for method_name, preprocess_func in preprocessing_methods.items():
            processed = preprocess_func(test_text)
            
            # Tokenize
            sequences = tokenizer.texts_to_sequences([processed])
            
            # Check if sequence is empty
            if not sequences[0]:
                prediction = "ERROR: Empty sequence"
                seq_len = 0
            else:
                # Pad (try different maxlen values)
                for maxlen in [40, 100, 200]:
                    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
                    pred = model.predict(padded, verbose=0)[0][0]
                    
                    results_df.append({
                        'Text': test_name,
                        'Preprocessing': method_name,
                        'Maxlen': maxlen,
                        'Processed': processed[:50] + "..." if len(processed) > 50 else processed,
                        'Seq_Length': len(sequences[0]),
                        'Prediction': float(pred),
                        'First_10_Tokens': str(sequences[0][:10])
                    })
    
    # Display results
    st.header("📊 Complete Results Analysis")
    
    results_df = pd.DataFrame(results_df)
    
    # Show all results
    st.dataframe(results_df)
    
    # Analyze patterns
    st.subheader("🔍 Pattern Analysis")
    
    # Check if all predictions are similar
    predictions = results_df['Prediction'].values
    unique_predictions = len(set(predictions))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Predictions", unique_predictions)
    with col2:
        st.metric("Min Prediction", f"{min(predictions):.6f}")
    with col3:
        st.metric("Max Prediction", f"{max(predictions):.6f}")
    
    if unique_predictions <= 2:
        st.error("🚨 MAJOR ISSUE: Model gives almost identical predictions for everything!")
        st.write("This suggests:")
        st.write("1. **Model is broken/overfitted**")
        st.write("2. **Severe preprocessing mismatch**") 
        st.write("3. **Wrong model architecture loaded**")
        st.write("4. **Tokenizer corruption**")
    
    # Check for empty sequences
    empty_sequences = results_df[results_df['Seq_Length'] == 0]
    if not empty_sequences.empty:
        st.error("🚨 EMPTY SEQUENCES DETECTED:")
        st.dataframe(empty_sequences)
    
    # Prediction distribution
    st.subheader("📈 Prediction Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(predictions, bins=20, alpha=0.7)
    ax.set_xlabel('Prediction Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of All Predictions')
    st.pyplot(fig)
    
    # Group by preprocessing method
    st.subheader("🔬 By Preprocessing Method")
    for method in results_df['Preprocessing'].unique():
        method_data = results_df[results_df['Preprocessing'] == method]
        method_preds = method_data['Prediction'].values
        
        st.write(f"**{method}:**")
        st.write(f"  - Range: {min(method_preds):.6f} to {max(method_preds):.6f}")
        st.write(f"  - Std Dev: {np.std(method_preds):.6f}")
        st.write(f"  - Unique values: {len(set(method_preds))}")
    
    st.header("🎯 Diagnosis & Recommendations")
    
    # Final diagnosis
    if unique_predictions <= 2:
        st.error("**VERDICT: Model needs retraining or has fundamental issues**")
        st.write("**Recommended actions:**")
        st.write("1. ✅ Check if you loaded the correct model file")
        st.write("2. ✅ Verify the model was actually trained (not just initialized)")
        st.write("3. ✅ Check if tokenizer matches the training tokenizer") 
        st.write("4. ✅ Consider retraining with the exact same preprocessing")
        st.write("5. ✅ Verify training data was balanced and labels correct")
    elif np.std(predictions) > 0.1:
        st.success("**VERDICT: Model shows variation - preprocessing issue likely**")
        st.write("Focus on matching the exact training preprocessing pipeline")
    else:
        st.warning("**VERDICT: Inconclusive - need more investigation**")

else:
    st.error("Cannot load model files. Please ensure files are present:")
    st.write("- fake_news_model.keras (or .h5)")
    st.write("- `tokenizer_new.pkl`")
st.markdown("---")
st.write("💡 **This tool will definitively tell us if the issue is:**")
st.write("- Preprocessing mismatch (fixable)")  
st.write("- Model architecture problem (needs retraining)")
st.write("- Tokenizer corruption (needs retraining)")
st.write("- Model not actually trained (needs retraining)")