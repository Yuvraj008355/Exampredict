import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load dataset
data = pd.read_csv("StudentPerformanceFactors.csv")

selected_columns = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity", "Gender", "Internet_Access",
    "Motivation_Level", "Family_Income", "School_Type", "Teacher_Quality",
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Exam_Score"
]
data = data[selected_columns].copy()

# -----------------------------------------------
# ✅ STEP 1: NaN clean
# -----------------------------------------------
numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
                "Tutoring_Sessions", "Physical_Activity", "Exam_Score"]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col].fillna(data[col].median(), inplace=True)

cat_cols = ["Gender", "Internet_Access", "Motivation_Level", "Family_Income",
            "School_Type", "Teacher_Quality", "Parental_Involvement",
            "Access_to_Resources", "Extracurricular_Activities"]
for col in cat_cols:
    data[col] = data[col].astype(str).str.strip()
    data[col].replace("nan", np.nan, inplace=True)
    data[col].fillna(data[col].mode()[0], inplace=True)

print(f"✅ Data cleaned. Shape: {data.shape}")
print(f"📊 Score distribution:\n{data['Exam_Score'].describe()}\n")

# -----------------------------------------------
# ✅ STEP 2: Oversample high scorers
# Yeh sabse important fix hai!
# Model ne kabhi 80+ score nahi dekha tha properly
# -----------------------------------------------
low    = data[data["Exam_Score"] < 65]
mid    = data[(data["Exam_Score"] >= 65) & (data["Exam_Score"] < 75)]
high   = data[(data["Exam_Score"] >= 75) & (data["Exam_Score"] < 85)]
vhigh  = data[data["Exam_Score"] >= 85]

print(f"📊 Score groups before oversampling:")
print(f"   Low  (<65)  : {len(low)}")
print(f"   Mid  (65-74): {len(mid)}")
print(f"   High (75-84): {len(high)}")
print(f"   Very High (85+): {len(vhigh)}")

# High scorers ko 4x repeat karo, very high ko 8x
high_over  = pd.concat([high]  * 4, ignore_index=True)
vhigh_over = pd.concat([vhigh] * 8, ignore_index=True)

data_balanced = pd.concat([low, mid, high_over, vhigh_over], ignore_index=True)
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ After oversampling shape: {data_balanced.shape}")
print(f"📊 Score distribution after:\n{data_balanced['Exam_Score'].describe()}\n")

# -----------------------------------------------
# ✅ STEP 3: Feature Engineering
# -----------------------------------------------
support_map    = {"High": 3, "Medium": 2, "Low": 1}
motivation_map = {"High": 3, "Medium": 2, "Low": 1}
income_map     = {"High": 3, "Medium": 2, "Low": 1}

data_balanced["Study_Score"] = (
    data_balanced["Hours_Studied"] * (data_balanced["Attendance"] / 100)
)
data_balanced["Support_Score"] = (
    data_balanced["Parental_Involvement"].map(support_map).fillna(1) +
    data_balanced["Access_to_Resources"].map(support_map).fillna(1) +
    data_balanced["Teacher_Quality"].map(support_map).fillna(1)
)
data_balanced["Motivation_Income"] = (
    data_balanced["Motivation_Level"].map(motivation_map).fillna(1) +
    data_balanced["Family_Income"].map(income_map).fillna(1)
)
data_balanced["Sleep_Quality"] = data_balanced["Sleep_Hours"].apply(
    lambda x: 3 if 7 <= x <= 8 else (2 if 6 <= x <= 9 else 1)
)
data_balanced["High_Achiever"] = (data_balanced["Previous_Scores"] >= 80).astype(int)

# -----------------------------------------------
# ✅ STEP 4: Dummy encoding
# -----------------------------------------------
data_balanced = pd.get_dummies(data_balanced)
cols_to_drop = [c for c in data_balanced.columns if
                c.endswith('_Low') or c.endswith('_No') or
                c.endswith('_Female') or c.endswith('_Public')]
data_balanced = data_balanced.drop(columns=cols_to_drop, errors='ignore')
data_balanced = data_balanced.fillna(0)

print(f"✅ Total columns: {len(data_balanced.columns)}")

# Split
X = data_balanced.drop("Exam_Score", axis=1)
y = data_balanced["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# ✅ STEP 5: Ensemble model
# -----------------------------------------------
gbr1 = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=5,
    min_samples_leaf=2, subsample=0.8, max_features=0.8, random_state=42
)
rfr = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=2,
    max_features=0.7, random_state=42
)
gbr2 = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    min_samples_leaf=1, subsample=0.9, random_state=24
)

ensemble = VotingRegressor(estimators=[
    ("gbr1", gbr1), ("rf", rfr), ("gbr2", gbr2)
])

print("⏳ Training ensemble (1-2 min)...")
ensemble.fit(X_train, y_train)
print("✅ Training done!")

# -----------------------------------------------
# ✅ STEP 6: Calibration
# -----------------------------------------------
y_pred_train = ensemble.predict(X_train)
y_pred_test  = ensemble.predict(X_test)

actual_min = float(y_train.min())
actual_max = float(y_train.max())
pred_min   = float(y_pred_train.min())
pred_max   = float(y_pred_train.max())

def calibrate(preds):
    cal = actual_min + (preds - pred_min) * (actual_max - actual_min) / (pred_max - pred_min)
    return np.clip(cal, actual_min, actual_max)

y_pred_cal = calibrate(y_pred_test)

mae = mean_absolute_error(y_test, y_pred_cal)
r2  = r2_score(y_test, y_pred_cal)

print(f"\n🎯 Model Performance:")
print(f"   MAE             : {mae:.2f}")
print(f"   R²              : {r2:.4f}")
print(f"   Predicted range : {y_pred_cal.min():.1f} – {y_pred_cal.max():.1f}")
print(f"   Actual range    : {y_test.min():.0f} – {y_test.max():.0f}")

# -----------------------------------------------
# Save
# -----------------------------------------------
model_bundle = {
    "model"      : ensemble,
    "pred_min"   : pred_min,
    "pred_max"   : pred_max,
    "actual_min" : actual_min,
    "actual_max" : actual_max,
}

pickle.dump(model_bundle, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))
print("\n✅ Saved! app.py same rehega — koi change nahi chahiye.")