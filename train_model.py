# model/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/sudhirnl7/FakeNewsClassifier/master/news.csv')
X = df['text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
