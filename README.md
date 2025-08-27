#  Fake News Detection using Bidirectional LSTM

##  Project Overview
A deep learning solution to combat misinformation by classifying news articles as **Real** or **Fake** using advanced Natural Language Processing and Bidirectional LSTM neural networks.

## Trained Model

The trained model can be downloaded from [this Google Drive folder](https://drive.google.com/drive/folders/1RZs_mJD-qQvaRKQswSIMRpaIVqoEddJM).  
Save the file `fake_news_model.h5` in the project folder before running the code.


## 🎯 Key Results
- **Model Accuracy: 99.8%** on test data
- Successfully processed **44,919 news articles**
- Balanced dataset with equal representation of real and fake news
- Robust text preprocessing pipeline

## 🎯 Features
- **Bidirectional LSTM Architecture**: Captures context from both directions in text
- **Advanced Text Preprocessing**: Tokenization, stopword removal, padding
- **Word Embeddings**: 128-dimensional embedding layer for semantic understanding
- **Data Visualization**: Word clouds and distribution analysis
- **Model Persistence**: Saved trained model and tokenizer for deployment

## 🎯 Tech Stack
- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural language processing
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Text visualization
- **Gensim** - Text processing utilities

## 📁 Dataset
This project uses the **Fake and Real News Dataset** containing:
- **True.csv**: Verified real news articles
- **Fake.csv**: Confirmed fake news articles

**Dataset Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

*Note: Due to size limitations, datasets are not included in this repository. Please download from the Kaggle link above.*

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
pip install tensorflow pandas numpy matplotlib seaborn nltk wordcloud gensim scikit-learn plotly
```

### Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/amalluu/fakenews_detection_model
   ```

2. **Download the dataset** from Kaggle link above and place CSV files in project directory

3. **Run the notebook**
   ```bash
   jupyter notebook FAKENEWSDETECTIONMODELLSTM.ipynb
   ```

4. **For predictions** (after training):
   ```python
   # Load saved model and tokenizer
   from tensorflow.keras.models import load_model
   import pickle
   
   model = load_model('fake_news_model.h5')
   with open('tokenizer.pkl', 'rb') as f:
       tokenizer = pickle.load(f)
   ```

## 📈 Model Performance
- **Training Accuracy**: 99.8%
- **Validation Split**: 10% of training data
- **Test Accuracy**: 99.8%
- **Epochs**: 2 (quick convergence due to effective architecture)
- **Batch Size**: 64


## 🔍 Key Insights
- **Balanced Dataset**: Equal distribution of real vs fake news ensures unbiased learning
- **Text Length Optimization**: Padded sequences to 40 words for optimal performance
- **Effective Preprocessing**: Stopword removal and tokenization significantly improved accuracy
- **Bidirectional Context**: LSTM captures both forward and backward context in news articles

## 📊 Visualizations Included
- Distribution of news by subject categories
- Word clouds for real vs fake news
- Text length distribution analysis
- Confusion matrix for model evaluation

## 🚀 Future Improvements
- [] Real-time news article classification API
- [] Web interface for user input
- [] Multi-language support
- [] Integration with news feeds
- [] Advanced attention mechanisms

## 📝 Project Structure
```
fake-news-detection/
├── FAKENEWSDETECTIONMODELLSTM.ipynb  # Main notebook
├── tokenizer.pkl                     # Fitted tokenizer
├── README.md                         # Project documentation
└── requirements.txt                  # Dependencies
```

##  Contributing
Feel free to fork this project and submit pull requests for improvements!

## 📧 Contact
**Name** - amalukuruvilla9496@gmail.com
**GitHub** - amalluu

---
⭐ **If you found this project helpful, please give it a star!**
