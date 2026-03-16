# Multinomial Naive Bayes for Spam Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
spam = pd.read_csv(r"C:\Users\Admin\Downloads\spam.csv")

# Features and target
X = spam['v1']
y = spam['v2']

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
Multinomial_nb = MultinomialNB()
Multinomial_nb.fit(X_train, y_train)

# Predict
predictions = Multinomial_nb.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")