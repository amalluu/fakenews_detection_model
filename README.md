# Fake News Detection using Bidirectional LSTM

## Project Overview
A deep learning project demonstrating text classification using Natural Language Processing and Bidirectional LSTM neural networks to classify news articles as **Real** or **Fake**.

## 🎯 Features
- **Bidirectional LSTM Architecture**: Captures context from both directions in text
- **Text Preprocessing Pipeline**: Tokenization, stopword removal, padding
- **Word Embeddings**: 128-dimensional embedding layer for semantic understanding
- **Data Visualization**: Word clouds and distribution analysis
- **Streamlit Web App**: Interactive interface for testing the model
- **Model Persistence**: Saved trained model and tokenizer

## 🎯 Tech Stack
- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural language processing
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Text visualization
- **Streamlit** - Web app framework

## 📁 Dataset & Model Files
Download the required files from [this Google Drive folder](https://drive.google.com/drive/folders/1RZs_mJD-qQvaRKQswSIMRpaIVqoEddJM):

**Required Files:**
- `Fake.csv` - Fake news articles dataset
- `True.csv` - Real news articles dataset  
- `fake_news_model.keras` - Pre-trained model file

**Dataset Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

*Note: Due to GitHub file size limitations, datasets and model files are hosted on Google Drive.*

## 🎯 Model Architecture
```
Sequential Model:
├── Embedding Layer (vocab_size → 128 dimensions)
├── Bidirectional LSTM (128 units)
├── Dense Layer (128 units, ReLU activation)
└── Output Layer (1 unit, Sigmoid activation)

Total Parameters: ~2.8M
Optimizer: Adam
Loss: Binary Crossentropy
```

## 🎯 Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow pandas numpy matplotlib seaborn nltk wordcloud streamlit scikit-learn plotly
```

### Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/amalluu/fakenews_detection_model
cd fakenews_detection_model
```

### 2. Download Required Files
Download `Fake.csv`, `True.csv`, and `fake_news_model.keras` from the Google Drive link above and place them in the project directory.

### 3. Run the Jupyter Notebook
```bash
jupyter notebook FAKENEWSDETECTIONMODELLSTM.ipynb
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

### 5. For Custom Predictions
```python
from tensorflow.keras.models import load_model
import pickle

# Load saved model and tokenizer
model = load_model('fake_news_model.keras')
with open('tokenizer_new.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
```

## 📈 Model Performance
- **Training Performance**: 99.8 on the specific training dataset
- **Validation Split**: 10% of training data
- **Epochs**: 2 (quick convergence)
- **Batch Size**: 64

**Important Limitations:**
- Model is trained on a specific dataset and may not generalize to other news sources
- Preprocessing is aggressive (removes words ≤ 3 characters)
- Uses only first 40 words of articles
- Designed for educational/demonstration purposes

## 📊 What's Included
- **Data Analysis**: Distribution of news by categories
- **Visualizations**: Word clouds for real vs fake news
- **Text Processing**: Length distribution analysis
- **Model Training**: Complete LSTM implementation
- **Web Interface**: Streamlit app for testing

## 📝 Project Structure
```
fakenews_detection_model/
├── FAKENEWSDETECTIONMODELLSTM copy.ipynb  # Main training notebook
├── app.py                                 # Streamlit web app
├── tokenizer_new.pkl                      # Fitted tokenizer
├── maxlen_new.pkl                         # Sequence length parameter
├── model_diagnostic.py                    # Model analysis tools
├── check_maxlen.py                        # Utility script
├── requirements.txt                       # Dependencies
└── README.md                              # This file
```

## 🔍 Educational Value
This project demonstrates:
- Text preprocessing for NLP tasks
- LSTM implementation for sequence classification
- Data visualization techniques
- Model evaluation and persistence
- Building interactive ML applications

**⚠️ Important Note**: This model is designed for educational purposes and works specifically with the dataset used for training. It is not intended for real-world fake news detection applications.

## 🛠️ Future Learning Opportunities
- Experiment with different preprocessing techniques
- Try other neural network architectures (GRU, Transformer)
- Implement attention mechanisms
- Work with larger, more diverse datasets
- Build more robust evaluation metrics

## 📧 Contact
**Amalu Kuruvilla** - amalukuruvilla9496@gmail.com  
**GitHub** - [amalluu](https://github.com/amalluu)

---
⭐ **If you found this educational project helpful, please give it a star!**
