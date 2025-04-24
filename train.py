from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from data import load_data, preprocess_data  # Import from data.py

def train_model(X_train, y_train, C=1.0):
    """Trains a Logistic Regression model on the training data.
    Logs parameters and model to MLflow.
    """
    mlflow.start_run()  # Start MLflow run

    # Log parameters
    mlflow.log_param("C", C)

    print("Training Logistic Regression model with C =", C)
    model = LogisticRegression(C=C, random_state=42, max_iter=1000)  # Increased max_iter
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()  # End MLflow run
    return model

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df) # Use the function
    train_model(X_train, y_train, C=1.0)
    print("Model training complete.")