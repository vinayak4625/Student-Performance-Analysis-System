# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Ignore warning messages for cleaner output
warnings.filterwarnings('ignore')
print("Libraries imported successfully!")

# ==========================================
# 2. GENERATE SYNTHETIC DATASET
# ==========================================
np.random.seed(42)
n_students = 500

# Generating feature data
data = {
    'Hours_Studied': np.random.uniform(1, 10, n_students),
    'Attendance_Rate': np.random.uniform(50, 100, n_students),
    'Previous_Scores': np.random.uniform(40, 100, n_students),
    'Extracurriculars': np.random.choice(['Yes', 'No'], n_students),
    'Sleep_Hours': np.random.uniform(4, 9, n_students)
}
df = pd.DataFrame(data)

# Creating a realistic target variable (Final_Score) with random noise
df['Final_Score'] = (
    (df['Hours_Studied'] * 2.5) +
    (df['Attendance_Rate'] * 0.3) +
    (df['Previous_Scores'] * 0.4) +
    np.where(df['Extracurriculars'] == 'Yes', 3, 0) +
    (df['Sleep_Hours'] * 1.5) +
    np.random.normal(0, 4, n_students) # Random noise
)

# Ensure scores cap at 100
df['Final_Score'] = np.clip(df['Final_Score'], 0, 100)

print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
plt.figure(figsize=(15, 5))

# Plot 1: Hours Studied vs Final Score
plt.subplot(1, 3, 1)
sns.scatterplot(x='Hours_Studied', y='Final_Score', data=df, alpha=0.6)
plt.title('Study Hours vs Final Score')

# Plot 2: Attendance vs Final Score
plt.subplot(1, 3, 2)
sns.scatterplot(x='Attendance_Rate', y='Final_Score', data=df, color='green', alpha=0.6)
plt.title('Attendance vs Final Score')

# Plot 3: Score Distribution
plt.subplot(1, 3, 3)
sns.histplot(df['Final_Score'], bins=20, kde=True, color='purple')
plt.title('Distribution of Final Scores')

plt.tight_layout()
plt.show()

# ==========================================
# 4. DATA PREPROCESSING
# ==========================================
# Convert 'Yes'/'No' to 1/0 for the Extracurriculars column
le = LabelEncoder()
df['Extracurriculars'] = le.fit_transform(df['Extracurriculars'])

# Define Features (X) and Target (y)
X = df.drop('Final_Score', axis=1)
y = df['Final_Score']

# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Splitting ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ==========================================
# 5. TRAIN THE MACHINE LEARNING MODEL
# ==========================================
# Initialize and train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
print("\nModel Training Complete!")

# ==========================================
# 6. EVALUATE MODEL PERFORMANCE
# ==========================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Visualize Actual vs Predicted scores
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Student Scores')
plt.show()

# Display Feature Importance
importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)
print("\n--- Feature Importances ---")
print(importance)