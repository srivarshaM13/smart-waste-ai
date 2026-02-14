import joblib
import pandas as pd

# Load trained model
model = joblib.load("waste_model.pkl")

print("===== Waste Classification System =====")

try:
    weight = float(input("Enter Weight: "))
    size = int(input("Enter Size (1=Small, 2=Medium, 3=Large): "))
    texture = int(input("Enter Texture (1=Smooth, 2=Rough): "))
    color = int(input("Enter Color (1=Plastic, 2=Organic, 3=Metal, 4=Paper): "))

    # Create dataframe with column names
    sample = pd.DataFrame([[weight, size, texture, color]],
                          columns=["Weight", "Size", "Texture", "Color"])

    result = model.predict(sample)

    print("Predicted Waste Type:", result[0])

except:
    print("Invalid input. Please enter correct values.")
