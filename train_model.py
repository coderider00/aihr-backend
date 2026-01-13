import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import os

# --- CONFIG ---
DATA_FILE = 'data.xls'
MODEL_PATH = 'models/'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def load_or_create_dummy_data():
    """Generates dummy data if file is missing to prevent crash."""
    if os.path.exists(DATA_FILE):
        print(f"Loading {DATA_FILE}...")
        try:
            return pd.read_excel(DATA_FILE)
        except Exception as e:
            print(f"Error reading Excel: {e}. Generating dummy data.")
    else:
        print(f"{DATA_FILE} not found. Generating dummy data for training...")
    
    # Dummy data structure matching standard HR datasets
    data = {
        'Age': np.random.randint(22, 60, 100),
        'Department': np.random.choice(['Sales', 'HR', 'R&D'], 100),
        'DistanceFromHome': np.random.randint(1, 30, 100),
        'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical'], 100),
        'EnvironmentSatisfaction': np.random.randint(1, 5, 100),
        'JobSatisfaction': np.random.randint(1, 5, 100),
        'WorkLifeBalance': np.random.randint(1, 5, 100),
        'YearsAtCompany': np.random.randint(1, 20, 100),
        'PerformanceRating': np.random.randint(2, 5, 100), # Target 1
        'EmpLastSalaryHikePercent': np.random.randint(10, 25, 100) # Target 2
    }
    return pd.DataFrame(data)

# 1. Load Data
df = load_or_create_dummy_data()

# 2. Preprocessing & Encoding
encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Fill missing values if any
df.fillna(df.mean(), inplace=True)

# 3. Split Data
X = df.drop(['PerformanceRating', 'EmpLastSalaryHikePercent'], axis=1)
y_class = df['PerformanceRating'] # Classification Target
y_reg = df['EmpLastSalaryHikePercent'] # Regression Target

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# 4. Train Models
print("Training Classification Model (Random Forest)...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_class_train)

print("Training Regression Model (XGBoost)...")
reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
reg.fit(X_train, y_reg_train)

# 5. Evaluation
y_pred_class = clf.predict(X_test)
acc = accuracy_score(y_class_test, y_pred_class)
print(f"Classification Accuracy: {acc:.2f}")

y_pred_reg = reg.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_pred_reg)
print(f"Regression MAE: {mae:.2f}")

# 6. Save Artifacts
artifacts = {
    'classifier': clf,
    'regressor': reg,
    'encoders': encoders,
    'feature_names': list(X.columns)
}

joblib.dump(artifacts, os.path.join(MODEL_PATH, 'aihr_models.pkl'))
print("Models and encoders saved successfully to 'models/aihr_models.pkl'")
