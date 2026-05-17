import pandas as pd
data = pd.read_csv("StudentPerformanceFactors.csv")
print(data["Exam_Score"].max())