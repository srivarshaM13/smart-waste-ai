import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    # Load dataset
    data = pd.read_csv("dataset.csv")

    # Clean column names
    data.columns = data.columns.str.strip()

    print("Dataset Columns:", data.columns)

    # Remove empty rows
    data = data.dropna()

    # Check column exists
    if "Waste_Type" not in data.columns:
        raise Exception("Column 'Waste_Type' not found in dataset!")

    # Separate input and output
    X = data.drop("Waste_Type", axis=1)
    y = data["Waste_Type"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = RandomForestClassifier()

    # Train model
    model.fit(X_train, y_train)

    # Test model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    # Save model
    joblib.dump(model, "waste_model.pkl")
    print("Model saved successfully!")

except Exception as e:
    print("Error occurred:", e)
