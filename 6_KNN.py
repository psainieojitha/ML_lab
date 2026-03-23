import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = knn.predict(X_train_scaled)
predictions = knn.predict(X_test_scaled)

# Count correct and wrong predictions
correct_predictions = 0
wrong_predictions = 0

for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        print(f"Correct prediction: Predicted {predictions[i]}, Actual {y_test[i]}")
        correct_predictions += 1
    else:
        print(f"Wrong prediction: Predicted {predictions[i]}, Actual {y_test[i]}")
        wrong_predictions += 1

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)

print("Training accuracy:", train_accuracy * 100)
print("Wrong predictions:", wrong_predictions)
print("Correct predictions:", correct_predictions)