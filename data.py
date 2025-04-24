import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """Loads the Iris dataset."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    df['target'] = iris['target']
    return df

def preprocess_data(df):
    """Preprocesses the Iris dataset:
    * Splits into training and testing sets.
    * Scales the features.
    """
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data loaded and preprocessed.")
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)