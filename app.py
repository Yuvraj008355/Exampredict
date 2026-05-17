from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# -----------------------------------------------
# ✅ Naya model bundle load karo
# -----------------------------------------------
bundle  = pickle.load(open("model.pkl", "rb"))
model      = bundle["model"]
pred_min   = bundle["pred_min"]
pred_max   = bundle["pred_max"]
actual_min = bundle["actual_min"]
actual_max = bundle["actual_max"]
columns    = pickle.load(open("columns.pkl", "rb"))

print("✅ Model loaded!")
print(f"   Calibration range: {actual_min:.1f} – {actual_max:.1f}")


# -----------------------------------------------
# Calibration helper
# -----------------------------------------------
def calibrate(preds):
    calibrated = actual_min + (preds - pred_min) * (actual_max - actual_min) / (pred_max - pred_min)
    return np.clip(calibrated, actual_min, actual_max)


# -----------------------------------------------
# Input builder - form values se feature dict
# -----------------------------------------------
def build_input(form):
    support_map    = {"High": 3, "Medium": 2, "Low": 1}
    motivation_map = {"High": 3, "Medium": 2, "Low": 1}
    income_map     = {"High": 3, "Medium": 2, "Low": 1}

    hours_studied     = float(form["Hours_Studied"])
    attendance        = float(form["Attendance"])
    sleep_hours       = float(form["Sleep_Hours"])
    previous_scores   = float(form["Previous_Scores"])
    tutoring_sessions = float(form["Tutoring_Sessions"])
    physical_activity = float(form["Physical_Activity"])
    motivation_level  = form["Motivation_Level"]
    family_income     = form["Family_Income"]
    parental_inv      = form["Parental_Involvement"]
    access_resources  = form["Access_to_Resources"]
    teacher_quality   = form["Teacher_Quality"]
    gender            = form["Gender"]
    internet_access   = form["Internet_Access"]
    school_type       = form["School_Type"]
    extracurricular   = form["Extracurricular_Activities"]

    # ✅ Engineered features (model.py se match karna zaroori)
    study_score      = hours_studied * (attendance / 100)
    support_score    = (support_map.get(parental_inv, 1) +
                        support_map.get(access_resources, 1) +
                        support_map.get(teacher_quality, 1))
    motivation_income = (motivation_map.get(motivation_level, 1) +
                         income_map.get(family_income, 1))
    sleep_quality    = 3 if 7 <= sleep_hours <= 8 else (2 if 6 <= sleep_hours <= 9 else 1)
    high_achiever    = 1 if previous_scores >= 80 else 0

    input_dict = {
        "Hours_Studied"    : hours_studied,
        "Attendance"       : attendance,
        "Sleep_Hours"      : sleep_hours,
        "Previous_Scores"  : previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
        "Study_Score"      : study_score,
        "Support_Score"    : support_score,
        "Motivation_Income": motivation_income,
        "Sleep_Quality"    : sleep_quality,
        "High_Achiever"    : high_achiever,
    }

    # ✅ Dummy columns
    if gender == "Male":
        input_dict["Gender_Male"] = 1
    if internet_access == "Yes":
        input_dict["Internet_Access_Yes"] = 1
    if motivation_level == "High":
        input_dict["Motivation_Level_High"] = 1
    elif motivation_level == "Medium":
        input_dict["Motivation_Level_Medium"] = 1
    if family_income == "High":
        input_dict["Family_Income_High"] = 1
    elif family_income == "Medium":
        input_dict["Family_Income_Medium"] = 1
    if school_type == "Private":
        input_dict["School_Type_Private"] = 1
    if teacher_quality == "High":
        input_dict["Teacher_Quality_High"] = 1
    elif teacher_quality == "Medium":
        input_dict["Teacher_Quality_Medium"] = 1
    if parental_inv == "High":
        input_dict["Parental_Involvement_High"] = 1
    elif parental_inv == "Medium":
        input_dict["Parental_Involvement_Medium"] = 1
    if access_resources == "High":
        input_dict["Access_to_Resources_High"] = 1
    elif access_resources == "Medium":
        input_dict["Access_to_Resources_Medium"] = 1
    if extracurricular == "Yes":
        input_dict["Extracurricular_Activities_Yes"] = 1

    # ✅ Reindex to match training columns
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)
    return input_df


# -----------------------------------------------
# Grade helper
# -----------------------------------------------
def get_grade(score):
    if score >= 90: return "A+"
    elif score >= 80: return "A"
    elif score >= 70: return "B+"
    elif score >= 60: return "B"
    elif score >= 50: return "C"
    else: return "F"


# -----------------------------------------------
# ROUTES
# -----------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_df   = build_input(request.form)
        raw_pred   = model.predict(input_df)[0]
        calibrated = calibrate(np.array([raw_pred]))[0]
        score      = round(float(calibrated), 2)
        grade      = get_grade(score)

        return render_template(
            "result.html",
            score=score,
            grade=grade,
            hours=request.form["Hours_Studied"],
            attendance=request.form["Attendance"],
            motivation=request.form["Motivation_Level"],
            previous=request.form["Previous_Scores"],
            tutoring=request.form["Tutoring_Sessions"],
            sleep=request.form["Sleep_Hours"],
            physical=request.form["Physical_Activity"],
            parental=request.form["Parental_Involvement"],
            resources=request.form["Access_to_Resources"],
            teacher=request.form["Teacher_Quality"],
        )

    except Exception as e:
        return render_template("result.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        # JSON data ko form-style dict mein convert karo
        data     = request.get_json()
        input_df = build_input(data)
        raw_pred = model.predict(input_df)[0]
        score    = round(float(calibrate(np.array([raw_pred]))[0]), 2)

        return jsonify({
            "predicted_score": score,
            "grade"          : get_grade(score),
            "status"         : "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400


if __name__ == "__main__":
    app.run(debug=True)