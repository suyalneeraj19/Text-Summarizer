from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

# Download the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences
    sentence_list = sent_tokenize(text)

    # Create a dictionary to keep the frequency of each word
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    for word in word_tokenize(text):
        if word.lower() not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Normalize the frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Score the sentences based on word frequencies
    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    # Get the highest scoring sentences
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Join the sentences to form the summary
    summary = ' '.join(summary_sentences)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' in request.files:
        file = request.files['file']
        text = file.read().decode('utf-8')
    else:
        text = request.form['text']
    num_sentences = int(request.form['num_sentences'])
    summary = summarize_text(text, num_sentences)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
