import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/CVSN/heart.csv")

# Define features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# List of algorithms to evaluate
algorithms = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Different training and test data sizes to try
data_sizes = [0.6, 0.7, 0.8, 0.9]

# Loop through algorithms
for name, model in algorithms.items():
    print(f"Algorithm: {name}")
    for size in data_sizes:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - size, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training data size: {size:.1%}, Test accuracy: {accuracy:.4f}")
    print()


#ML algos for heart.csv