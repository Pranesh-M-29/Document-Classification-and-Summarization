from flask import Flask, render_template, request, redirect, url_for, session
import os
import joblib
from transformers import pipeline
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ensure a secret key for session management

# Mock user credentials for simplicity (in real apps, use a database)
admin_credentials = {
    "cheenucnu29@gmail.com": "cheenu"
}

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Load the trained model
model = joblib.load('news_classifier.pkl')  # Ensure path is correct to the trained model

# Set folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        # Check if the email and password match admin credentials
        if role == 'administrator' and email in admin_credentials and admin_credentials[email] == password:
            session['email'] = email
            session['role'] = 'administrator'
            return redirect(url_for('admin_dashboard'))  # Redirect to admin dashboard

        # Allow any user with an email and password to log in
        elif email and password:
            session['email'] = email
            session['role'] = 'user'
            return redirect(url_for('upload_page'))  # Redirect to upload page for regular user

        else:
            return "Invalid email or password, please try again."

    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # Get file and text data from the form
        uploaded_file = request.files.get('file')
        entered_text = request.form.get('text')
        action = request.form.get('action')

        # Handle the case where both file and text are empty
        if not uploaded_file and not entered_text:
            return "Please upload a file or enter some text.", 400

        # Process the uploaded file if present
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            # Read content for classification/summarization
            with open(file_path, 'r') as f:
                entered_text = f.read()  # Assuming the file is a text file

        if action == "classify":
            return redirect(url_for('classify', text=entered_text))
        elif action == "summarize":
            return redirect(url_for('summarize', text=entered_text))
        elif action == "both":
            return redirect(url_for('classify_and_summarize', text=entered_text))
        else:
            return "Invalid action selected.", 400

    return render_template('upload.html')

@app.route('/classify', methods=['GET'])
def classify():
    input_text = request.args.get('text')

    if input_text:
        # Directly pass the raw input text to the model pipeline
        prediction = model.predict([input_text])[0]  # Wrap the input_text in a list

        categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                      'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
                      'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                      'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
                      'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

        general_categories = {
            'alt.atheism': 'Religion',
            'comp.graphics': 'Technology',
            'comp.os.ms-windows.misc': 'Technology',
            'comp.sys.ibm.pc.hardware': 'Technology',
            'comp.sys.mac.hardware': 'Technology',
            'comp.windows.x': 'Technology',
            'misc.forsale': 'Commerce',
            'rec.autos': 'Automotive',
            'rec.motorcycles': 'Automotive',
            'rec.sport.baseball': 'Sports',
            'rec.sport.hockey': 'Sports',
            'sci.crypt': 'Science',
            'sci.electronics': 'Science',
            'sci.med': 'Medicine',
            'sci.space': 'Space Science',
            'soc.religion.christian': 'Religion',
            'talk.politics.guns': 'Politics',
            'talk.politics.mideast': 'Politics',
            'talk.politics.misc': 'Politics',
            'talk.religion.misc': 'Religion'
        }

        predicted_category = categories[prediction]
        general_field = general_categories.get(predicted_category, "Unknown")

        return render_template('classify.html', category=general_field)
    else:
        return "No text provided for classification."

# Load the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route('/summarize', methods=['GET'])
def summarize():
    input_text = request.args.get('text')

    if input_text:
        # Perform summarization
        summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        return render_template('summarize.html', original_text=input_text, summary=summary)
    else:
        return "No text provided for summarization."

@app.route('/classify_and_summarize', methods=['GET'])
def classify_and_summarize():
    input_text = request.args.get('text')

    if input_text:
        # Perform classification
        prediction = model.predict([input_text])[0]
        categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                      'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
                      'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                      'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
                      'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

        general_categories = {
            'alt.atheism': 'Religion',
            'comp.graphics': 'Technology',
            'comp.os.ms-windows.misc': 'Technology',
            'comp.sys.ibm.pc.hardware': 'Technology',
            'comp.sys.mac.hardware': 'Technology',
            'comp.windows.x': 'Technology',
            'misc.forsale': 'Commerce',
            'rec.autos': 'Automotive',
            'rec.motorcycles': 'Automotive',
            'rec.sport.baseball': 'Sports',
            'rec.sport.hockey': 'Sports',
            'sci.crypt': 'Science',
            'sci.electronics': 'Science',
            'sci.med': 'Medicine',
            'sci.space': 'Space Science',
            'soc.religion.christian': 'Religion',
            'talk.politics.guns': 'Politics',
            'talk.politics.mideast': 'Politics',
            'talk.politics.misc': 'Politics',
            'talk.religion.misc': 'Religion'
        }

        predicted_category = categories[prediction]
        general_field = general_categories.get(predicted_category, "Unknown")

        # Perform summarization
        summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        return render_template('classify_and_summarize.html', category=general_field, original_text=input_text, summary=summary)
    else:
        return "No text provided for classification and summarization."

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
categories = newsgroups.target_names

# Create a DataFrame with text data and categories
df = pd.DataFrame({
    'Text': newsgroups.data,
    'Category': [categories[i] for i in newsgroups.target]
})

@app.route('/admin_dashboard')
def admin_dashboard():
    # Convert the DataFrame to HTML to pass to the template
    data_html = df.to_html(classes='table table-striped', index=False)
    return render_template('admin_dashboard.html', table=data_html)

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('role', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
