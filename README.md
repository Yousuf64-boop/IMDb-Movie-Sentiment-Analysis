IMDb Movie Sentiment Analysis
📌 Project Overview

This project performs sentiment analysis on IMDb movie reviews using Machine Learning.
It classifies reviews as Positive or Negative based on their textual content.

The model is trained using TF-IDF vectorization and a Random Forest Classifier.

🚀 Features

Text preprocessing (cleaning, stopword removal)

TF-IDF based feature extraction

Machine Learning model (Random Forest)

Model evaluation (Accuracy, Confusion Matrix, Classification Report)

Model saving using pickle

Custom prediction function for new reviews

🛠️ Technologies Used

Python 🐍

NumPy

Pandas

NLTK

Scikit-learn

Pickle

📂 Dataset

Dataset file: movie.csv

Contains:

text → Movie review

label → Sentiment (0 = Negative, 1 = Positive)

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone <your-repo-link>
cd imdb-sentiment-analysis
2️⃣ Install Dependencies
pip install numpy pandas scikit-learn nltk
3️⃣ Download NLTK Stopwords
import nltk
nltk.download('stopwords')
🔄 Workflow
1. Data Loading

Load dataset using Pandas

2. Text Preprocessing

Convert to lowercase

Remove HTML tags

Remove punctuation

Remove stopwords

3. Train-Test Split

80% training, 20% testing

4. Feature Extraction

TF-IDF Vectorizer (max_features = 5000)

5. Model Training

Random Forest Classifier (n_estimators = 100)

6. Evaluation

Accuracy Score

Classification Report

Confusion Matrix

7. Model Saving
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
📊 Model Performance

Accuracy printed after training

Detailed classification metrics included:

Precision

Recall

F1-score

🔮 Prediction Function
def predict_review(review):
    review = clean_text(review)
    review_vector = tfidf.transform([review])
    prediction = model.predict(review_vector)

    if prediction == 1:
        return 'Positive Review'
    else:
        return 'Negative Review'
Example:
predict_review("This movie is amazing!")
📁 Output Files

sentiment_model.pkl → Trained model

tfidf_vectorizer.pkl → TF-IDF vectorizer

🎯 Future Improvements

Use Deep Learning (LSTM / BERT)

Hyperparameter tuning

Deploy as a web app (Flask / Django)

Use larger datasets for better accuracy

👨‍💻 Author

Yousuf Midya
B.Tech CSE (AI & ML)
