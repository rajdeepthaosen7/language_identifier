from flask import Flask, request, jsonify, render_template
import string
import re
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained language detection model
def load_model():
    global lrLangDetectModel
    with open('LRModel.pckl', 'rb') as lrLangDetectFile:
        lrLangDetectModel = pickle.load(lrLangDetectFile)

# Language detection function
def lang_detect(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    
    text = " ".join(text.split())  # Remove extra spaces
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = text.translate(translate_table)  # Remove punctuation
    
    pred = lrLangDetectModel.predict([text])  # Predict language
    prob = lrLangDetectModel.predict_proba([text])  # Get probabilities (optional)
    
    # Print the detected language to the terminal (for debugging)
    print(f"Detected language: {pred[0]}")
    
    return pred[0]

# Initialize the model when the app starts
load_model()

# Define route for language detection
@app.route('/detect_language', methods=['POST'])
def detect_language():
    data = request.get_json()  # Get the JSON data sent by the user
    text = data.get('text', '')  # Extract text from the data
    
    if not text:
        return jsonify({"error": "No text provided"}), 400  # Return error if text is missing
    
    language = lang_detect(text)  # Detect the language
    return jsonify({"language": language})  # Return the detected language as JSON

# Define route for the main page (HTML form)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
