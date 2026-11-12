from pathlib import Path

BASE = Path('/opt/airflow')
RAW = BASE / 'data/raw/student-mental-health-6af1a2527d06c522087238.csv'
PROCESSED = BASE / 'data/processed'
MODEL = BASE / 'models/pipeline.joblib'
REPORTS = BASE / 'reports'

TARGET = 'Mental_Health_Risk'
RANDOM_STATE = 42

NUMERIC = ['Age', 'Year_of_Study', 'Sleep_Hours', 'Study_Hours_Per_Week', 'GPA',
           'Social_Media_Hours', 'Screen_Time_Hours', 'Financial_Stress',
           'Family_Support', 'Attendance_Rate']
ORDINAL = ['Physical_Activity', 'Diet_Quality']
ORDINAL_CATEGORIES = [['Low', 'Moderate', 'High'], ['Poor', 'Average', 'Good']]
NOMINAL = ['Gender', 'Part_Time_Job', 'Previous_Mental_Health_History']