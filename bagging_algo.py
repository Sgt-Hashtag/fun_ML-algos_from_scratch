# Bagging helps improve accuracy and reduce overfitting especially in models that have high variance.

# BootstrapSampling BaseModelTraining Aggregation 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier:
    def __init__(self,base_classifier,n_estimators):
        self.base_classifier=base_classifier
        self.n_estimators =n_estimators
        self.trained_classifiers = []
        
    def fit(self,X,y):
        for _ in range(self.n_estimators):
            random_idx =np.random.choice(len(X),len(X),replace=True)
            X_sampled = X[random_idx]
            y_sampled = y[random_idx]
            
            classifier = self.base_classifier.__class__()
            classifier.fit(X_sampled,y_sampled)
            
            self.trained_classifiers.append(classifier)
        return self.trained_classifiers
    
    def predict(self,X):
        predictions=[classifier.predict(X) for classifier in self.trained_classifiers]
        
        # aggregate/majorityvote for each column of types
        majority=np.apply_along_axis(lambda x: np.bincount(x).argmax(),axis=0,arr=predictions)
        return majority  
    

digit = load_digits()
X, y = digit.data, digit.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the base classifier
dc = DecisionTreeClassifier()
model = BaggingClassifier(base_classifier=dc, n_estimators=10)
classifiers = model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
