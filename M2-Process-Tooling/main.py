import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = pd.read_csv("iris.csv")           
X = iris.drop("target", axis=1)
y = iris["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define model training function
def train_model(n_estimators, max_depth):
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model with n_estimators={n_estimators} and max_depth={max_depth} has accuracy: {accuracy}")

# Run experiments with different parameters
train_model(10, 2)
train_model(50, 5)
train_model(100, 10)
