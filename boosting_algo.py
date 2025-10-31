# Boosting helps improve accuracy by combining weak learners sequentially, focusing on correcting errors made by previous models.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class BoostingClassifier:
    def __init__(self, base_classifier, n_estimators):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.trained_classifiers = []
        self.classifier_weights = []
        
    def fit(self, X, y):
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            classifier = self.base_classifier.__class__()
            classifier.fit(X, y, sample_weight=sample_weights)
            predictions = classifier.predict(X)
            
            # Calculate error and classifier weight
            incorrect = (predictions != y)
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            if error >= 0.5:
                continue
            
            classifier_weight = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.trained_classifiers.append(classifier)
            self.classifier_weights.append(classifier_weight)
            
            # Update sample weights
            sample_weights *= np.exp(-classifier_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)
        
        return self.trained_classifiers, self.classifier_weights
    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        
        for classifier, weight in zip(self.trained_classifiers, self.classifier_weights):
            predictions = classifier.predict(X)
            final_predictions += weight * predictions
            
        return np.sign(final_predictions)
digit = load_digits()
X, y = digit.data, digit.target
# Convert labels to -1 and 1 for binary classification
y = np.where(y % 2 == 0, 1, -1) 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the base classifier
dc = DecisionTreeClassifier(max_depth=1)
model = BoostingClassifier(base_classifier=dc, n_estimators=10)
classifiers, weights = model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)