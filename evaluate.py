from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from data import load_data, preprocess_data # Import from data.py

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on the test data.
    Logs metrics to MLflow.
    """
    mlflow.start_run()  # Start MLflow run (within evaluate.py)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    mlflow.end_run()  # End MLflow run
    return accuracy

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df) # Use the function
    # Load the model.  Assumes the model was logged in the previous training run.
    model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id) #  ADDED THIS LINE
    model = mlflow.sklearn.load_model(model_uri) #  ADDED THIS LINE
    evaluate_model(model, X_test, y_test)