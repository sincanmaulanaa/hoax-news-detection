from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)
nltk.download('stopwords')
# Load models and vectorizer

naive_bayes_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))

# random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

data = pd.read_csv('Data_Training.csv')

X_vectorized = vectorizer.fit_transform(data['narasi'])

naive_bayes_model.fit(X_vectorized.toarray(), data['label'])

# random_forest_model.fit(X_vectorized, data['label'])

# Pre-processing teks (contoh: lowercase, menghapus karakter khusus, stopwords)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text


data['narasi'] = data['narasi'].apply(preprocess_text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])

    # Predict using Naive Bayes
    nb_prediction = naive_bayes_model.predict(vectorized_text.toarray())

    # # Predict using Random Forest
    # rf_prediction = random_forest_model.predict(vectorized_text)

    return render_template('result.html', text=text, nb_prediction=nb_prediction)

if __name__ == '__main__':
    app.run(debug=True)
