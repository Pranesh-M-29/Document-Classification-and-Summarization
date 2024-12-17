import os
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Step 1: Fetch the dataset
print("Fetching the dataset...")
newsgroups = fetch_20newsgroups(subset='all')
categories = newsgroups.target_names

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.3, random_state=42)

# Step 3: Build the model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Step 4: Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Step 5: Save the model
joblib.dump(model, 'news_classifier.pkl')
print("Model saved as news_classifier.pkl")

# Optional: Evaluate the model on the test set
predicted = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predicted)
print(f"Model Accuracy: {accuracy:.4f}")

# Additional: Save the vectorizer in a 'model/' folder
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the 'model/' directory exists
os.makedirs('models', exist_ok=True)

# Sample text data (replace with your actual dataset)
corpus = ["Sample text data", "Another text example"]

# Train the vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Save the trained vectorizer
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("TF-IDF Vectorizer saved as models/tfidf_vectorizer.pkl")
