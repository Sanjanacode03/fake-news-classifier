from flask import Flask, render_template, request
import pickle
import os

# Create Flask app
app = Flask(__name__)

# Load model and vectorizer from the model/ directory
model_path = os.path.join('model', 'fake_news_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news_text']

    # Transform and predict
    vectorized_text = vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)[0]

    return render_template('index.html', prediction=prediction)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
