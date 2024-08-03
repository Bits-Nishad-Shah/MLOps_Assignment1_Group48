import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = pd.read_csv("iris_full.csv")           
X = iris.drop("target", axis=1)
y = iris["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set the MLflow tracking server
experiment_id = mlflow.create_experiment("Full Dataset")
experiment = mlflow.get_experiment(experiment_id)

# Model 1: Random Forest
with mlflow.start_run(run_name="Random Forest 10 2", experiment_id=experiment.experiment_id):
    n_estimators = 10
    max_depth = 2

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
    mlflow.sklearn.log_model(model, "fd_model1")
    
    print(f"Model with n_estimators={n_estimators} and max_depth={max_depth} has accuracy: {accuracy}")

    # Define the model name
    model_name = "fd_random_forest_model1"
    
    # Register the model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/fd_random_forest_model1"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered with name: {registered_model.name} and version: {registered_model.version}")
    

# Model 2: Random Forest
with mlflow.start_run(run_name="Random Forest 50 5", experiment_id=experiment.experiment_id):
    n_estimators = 50
    max_depth = 5

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
    mlflow.sklearn.log_model(model, "fd_model2")
    
    print(f"Model with n_estimators={n_estimators} and max_depth={max_depth} has accuracy: {accuracy}")



# Model 3: Random Forest
with mlflow.start_run(run_name="Random Forest 100 10", experiment_id=experiment.experiment_id):
    n_estimators = 100
    max_depth = 10

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
    mlflow.sklearn.log_model(model, "fd_model3")
    
    print(f"Model with n_estimators={n_estimators} and max_depth={max_depth} has accuracy: {accuracy}")
