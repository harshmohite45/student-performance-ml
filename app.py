import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("==================================================")
print("   STUDENT PERFORMANCE PREDICTOR (AI/ML)   ")
print("==================================================\n")

# 1. Dataset Creation (Features: Study Hours, Attendance %, Past Marks -> Target: Final Grade)
data = {
    "Study_Hours": [2, 4, 5, 7, 8, 3, 6, 9, 1, 5],
    "Attendance": [60, 75, 80, 90, 95, 65, 85, 98, 50, 78],
    "Past_Score": [55, 65, 70, 85, 90, 60, 80, 95, 45, 72],
    "Final_Score": [58, 68, 73, 88, 92, 62, 82, 96, 48, 75],
}

df = pd.DataFrame(data)

# 2. Separate Feature Matrix (X) and Target Vector (y)
X = df[["Study_Hours", "Attendance", "Past_Score"]]
y = df["Final_Score"]

# 3. Train-Test Split (80% training data, 20% test data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model Training using Supervised Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

print("[+] Model Trained Successfully!\n")

# 5. Interactive Demo Inputs
try:
    hours = float(
        input("Enter Daily Study Hours (e.g., 6)     : ")
    )
    attendance = float(
        input("Enter Attendance Percentage (e.g., 85): ")
    )
    past_score = float(
        input("Enter Previous Test Score (e.g., 75) : ")
    )

    # Convert inputs to NumPy array for inference
    sample_input = np.array([[hours, attendance, past_score]])
    predicted_score = model.predict(sample_input)[0]

    print("\n--------------------------------------------------")
    print(f" PREDICTED FINAL SCORE: {predicted_score:.2f}%")
    print("--------------------------------------------------\n")

except ValueError:
    print("\n[!] Please enter valid numerical values.")