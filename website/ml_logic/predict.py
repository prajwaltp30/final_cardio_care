import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import mysql.connector
from database.db import get_db_connection
import numpy as np
from scipy.signal import find_peaks

# Load dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Drop unnecessary columns
df1 = df.drop(['Patient ID', 'Income', 'Country', 'Continent', 'Hemisphere', 'Cholesterol'], axis=1)

# Convert 'Sedentary Hours Per Day' to integer
df1['Sedentary Hours Per Day'] = df1['Sedentary Hours Per Day'].astype(int)

# Encode categorical variables
le = LabelEncoder()
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['Diet'] = le.fit_transform(df1['Diet'])

# Split 'Blood Pressure' into 'BP1' and 'BP2'
def split_blood_pressure(blood_pressure):
    return pd.Series(blood_pressure.split('/', 1))

df1[['BP1', 'BP2']] = df1['Blood Pressure'].apply(split_blood_pressure)
df1 = df1.drop(['Blood Pressure', 'Triglycerides', 'Sedentary Hours Per Day'], axis=1)

# Convert 'BP1' and 'BP2' to numeric
df1['BP1'] = pd.to_numeric(df1['BP1'], errors='coerce')
df1['BP2'] = pd.to_numeric(df1['BP2'], errors='coerce')

# Define weights for features
weights = {
    'Age': 0.1,
    'Sex': 0.05,
    'Heart Rate': 0.15,
    'Diabetes': 0.2,
    'Family History': 0.15,
    'Smoking': 0.25,
    'Obesity': 0.2,
    'Alcohol Consumption': 0.1,
    'Exercise Hours Per Week': 0.1,
    'Diet': 0.15,
    'Previous Heart Problems': 0.3,
    'Medication Use': 0.05,
    'Stress Level': 0.2,
    'BMI': 0.15,
    'Physical Activity Days Per Week': 0.1,
    'Sleep Hours Per Day': 0.1,
    'Heart Attack Risk': 0.4,  # This is a composite measure, not a factor itself
    'BP1': 0.25,  # Systolic Blood Pressure
    'BP2': 0.25   # Diastolic Blood Pressure
}

# Modify weights based on conditions
for index, row in df1.iterrows():
    if row['Age'] >= 45:
        weights['Age'] = 0.2
    if row['Sex'] == 0:
        weights['Sex'] = 0.1
    if row['Heart Rate'] < 60:
        weights['Heart Rate'] = 10 + (row['Heart Rate'] - 1) * 0.02
    elif row['Heart Rate'] > 100:
        weights['Heart Rate'] = 0.2 + (row['Heart Rate'] - 100) * 0.02
    if row['BP1'] > 150:
        weights['BP1'] = 0.2 + (row['BP1'] - 150) * 0.02
    if row['BP2'] > 90:
        weights['BP2'] = 0.2 + (row['BP2'] - 90) * 0.02

# Calculate total weighted sum
total_weighted_sum = df1.apply(lambda row: sum(row[col] * weights[col] for col in df1.columns), axis=1)

# Normalize total weighted sum
max_weighted_sum = total_weighted_sum.max()
min_weighted_sum = total_weighted_sum.min()
df1['percentage'] = ((total_weighted_sum - min_weighted_sum) / (max_weighted_sum - min_weighted_sum)) * 100

# Features and target variable
X = df1.drop(columns=['Heart Attack Risk', 'percentage'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_scaled, df1['Heart Attack Risk'])

# Initialize and train Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_scaled, df1['percentage'])

# Function to predict manually
def predict_manually(age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                     exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                     bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2):
    input_data = pd.DataFrame([[age, sex, heart_rate, diabetes, family_history, smoking, obesity, alcohol_consumption,
                                exercise_hours_per_week, diet, previous_heart_problems, medication_use, stress_level,
                                bmi, physical_activity_days_per_week, sleep_hours_per_day, bp1, bp2]],
                              columns=X.columns)
    input_scaled = scaler.transform(input_data)
    predicted_heart_attack_risk = random_forest_classifier.predict(input_scaled)[0]
    predicted_percentage = random_forest_regressor.predict(input_scaled)[0]
    return predicted_heart_attack_risk, predicted_percentage

# Function to get user details from the database
def get_user_details(email):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT age, sex, diabetes, famhistory, smoking, obesity, alcohol, exercise_hours, diet, heart_problem, bmi, physical_activity, sleep_hours, blood_pressure_systolic, blood_pressure_diastolic FROM user_details WHERE email = %s", (email,))
    user_details = cursor.fetchone()
    cursor.close()
    conn.close()
    return user_details

# ------------------ Atherosclerosis Risk (SDPPG) ------------------
def compute_atherosclerosis_risk(rppg_signal, fps: float = 15.0):
    try:
        signal = np.asarray(rppg_signal, dtype=np.float64)
        if signal.size < 60:
            return {"status": "insufficient", "message": "Not enough rPPG samples"}
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        dppg = np.diff(signal)
        sdppg = np.diff(dppg)
        if sdppg.size < 30:
            return {"status": "insufficient", "message": "Not enough SDPPG samples"}
        norm_sd = (sdppg - np.mean(sdppg)) / (np.std(sdppg) + 1e-8)
        peaks_pos, _ = find_peaks(norm_sd, distance=int(0.2*fps), prominence=0.05)
        peaks_neg, _ = find_peaks(-norm_sd, distance=int(0.2*fps), prominence=0.05)
        # Heuristic: choose top 5 by absolute amplitude within central region
        all_peaks = np.concatenate([peaks_pos, peaks_neg])
        if all_peaks.size < 3:
            return {"status": "insufficient", "message": "Insufficient peaks"}
        amps = norm_sd[all_peaks]
        sort_idx = np.argsort(-np.abs(amps))
        top_idx = all_peaks[sort_idx[:5]]
        top_idx = np.sort(top_idx)
        # Assign a,b,c,d,e by order
        a_i = int(top_idx[0])
        # find nearest negative after a for b
        b_candidates = [p for p in top_idx[1:] if norm_sd[p] < 0]
        e_candidates = [p for p in top_idx[1:] if norm_sd[p] > 0]
        if not b_candidates or not e_candidates:
            return {"status": "insufficient", "message": "Missing characteristic peaks"}
        b_i = int(b_candidates[0])
        e_i = int(e_candidates[-1])
        a = float(norm_sd[a_i]); b = float(norm_sd[b_i]); e = float(norm_sd[e_i])
        b_over_a = abs(b) / (abs(a) + 1e-8)
        e_over_a = abs(e) / (abs(a) + 1e-8)
        aging_index = (b - 0 - 0 - e) / (a + 1e-8)
        flags = []
        if b_over_a > 0.5:
            flags.append("High reflected wave (b/a>0.5)")
        if e_over_a < 1.0:
            flags.append("Flattened late waveform (e/a<1.0)")
        risk_level = "low"
        if b_over_a > 0.6 or (b_over_a > 0.5 and e_over_a < 0.9) or aging_index < -0.2:
            risk_level = "elevated"
        if b_over_a > 0.8 or (b_over_a > 0.6 and e_over_a < 0.8):
            risk_level = "high"
        return {
            "status": "ok",
            "b_over_a": round(b_over_a, 3),
            "e_over_a": round(e_over_a, 3),
            "aging_index": round(float(aging_index), 3),
            "risk_level": risk_level,
            "notes": flags
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
